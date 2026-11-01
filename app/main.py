from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select, desc
from pypdf import PdfReader
from docx import Document
from .database import Base, engine, get_db
from .models import Candidate, Recruiter, Skill, Resume, ResumeSkill, JobPosting, JobSkill, Application
from .matching import extract_skills, match_score

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(title="Job Matching & Recruitment Platform", version="1.0.0", lifespan=lifespan)

class CandidateIn(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)
    email: EmailStr

class RecruiterIn(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)
    email: EmailStr

class JobIn(BaseModel):
    recruiter_id: int
    title: str
    company: str
    location: str = "Remote"
    description: str
    required_skills: list[str] = []
    min_experience: int = 0

class ApplyIn(BaseModel):
    candidate_id: int
    job_id: int

class StatusIn(BaseModel):
    status: str

ALLOWED_STATUSES = {"Applied", "Shortlisted", "Interview", "Offer", "Rejected"}

def get_or_create_skill(db: Session, name: str) -> Skill:
    name = name.strip().lower()
    skill = db.scalar(select(Skill).where(Skill.name == name))
    if not skill:
        skill = Skill(name=name)
        db.add(skill)
        db.flush()
    return skill

def set_resume_skills(db: Session, resume: Resume, skills: set[str]):
    for name in sorted(skills):
        db.add(ResumeSkill(resume_id=resume.resume_id, skill_id=get_or_create_skill(db, name).skill_id))

def set_job_skills(db: Session, job: JobPosting, skills: set[str]):
    for name in sorted(skills):
        db.add(JobSkill(job_id=job.job_id, skill_id=get_or_create_skill(db, name).skill_id))

def read_resume(filename: str, content: bytes) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if suffix == ".docx":
        doc = Document(BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs).strip()
    if suffix in {".txt", ".md"}:
        return content.decode("utf-8", errors="replace").strip()
    raise HTTPException(400, "Upload a .txt, .pdf, or .docx resume.")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>TalentMatch Demo</title><style>
    body{font-family:system-ui;background:#f3f6fb;color:#182235;margin:0}header{background:#14213d;color:white;padding:28px 5%}
    main{max-width:1000px;margin:28px auto;padding:0 18px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:16px}
    section{background:white;border-radius:14px;padding:20px;box-shadow:0 4px 18px #15213b10}input,textarea,button{box-sizing:border-box;width:100%;padding:10px;margin:5px 0 10px;border:1px solid #cbd5e1;border-radius:8px;font:inherit}
    button{background:#2563eb;color:white;border:0;font-weight:650;cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f8fafc;padding:12px;border-radius:8px}
    small{color:#526078}.result{margin-top:18px}</style></head><body>
    <header><h1>TalentMatch</h1><p>Job Matching & Recruitment Platform · SQL + NLP matching demo</p></header><main>
    <p>Create a candidate and recruiter, post a job, upload a resume, then rank the candidate against jobs. API docs are at <a href="/docs">/docs</a>.</p>
    <div class="grid">
    <section><h2>1. Candidate</h2><input id="cname" placeholder="Full name"><input id="cemail" placeholder="Email"><button onclick="createCandidate()">Create candidate</button><pre id="cresult">Candidate ID appears here</pre></section>
    <section><h2>2. Recruiter</h2><input id="rname" placeholder="Full name"><input id="remail" placeholder="Email"><button onclick="createRecruiter()">Create recruiter</button><pre id="rresult">Recruiter ID appears here</pre></section>
    <section><h2>3. Post a job</h2><input id="rid" placeholder="Recruiter ID (e.g. 1)" type="number"><input id="title" placeholder="Job title"><input id="company" placeholder="Company"><input id="location" placeholder="Location (Remote)"><textarea id="desc" placeholder="Job description including skills" rows="4"></textarea><input id="skills" placeholder="Required skills, comma-separated"><button onclick="createJob()">Create job</button><pre id="jresult">Job ID appears here</pre></section>
    <section><h2>4. Upload resume</h2><input id="cid" type="number" placeholder="Candidate ID"><input id="resume" type="file" accept=".txt,.pdf,.docx"><button onclick="uploadResume()">Upload & extract skills</button><pre id="uresult">Upload result appears here</pre></section>
    <section><h2>5. Match candidates</h2><input id="matchcid" type="number" placeholder="Candidate ID"><button onclick="getMatches()">Rank jobs for candidate</button><pre id="mresult">Ranked results appear here</pre></section>
    <section><h2>6. Recruiter shortlist</h2><input id="matchjid" type="number" placeholder="Job ID"><button onclick="getCandidates()">Rank candidates for job</button><pre id="crresult">Ranked results appear here</pre></section>
    </div><div class="result"><small>Demo only: matching is advisory and must not be used as an automatic hiring/rejection decision.</small></div>
    <script>
    async function call(url,method='GET',body=null){let opts={method,headers:{}};if(body){opts.headers['Content-Type']='application/json';opts.body=JSON.stringify(body)}let r=await fetch(url,opts);let t=await r.text();let data;try{data=JSON.parse(t)}catch{data=t}if(!r.ok)throw Error(JSON.stringify(data));return data}
    function show(id,x){document.getElementById(id).textContent=JSON.stringify(x,null,2)}
    async function run(id,fn){try{show(id,await fn())}catch(e){show(id,{error:e.message})}}
    function createCandidate(){run('cresult',()=>call('/candidates','POST',{full_name:cname.value,email:cemail.value}))}
    function createRecruiter(){run('rresult',()=>call('/recruiters','POST',{full_name:rname.value,email:remail.value}))}
    function createJob(){run('jresult',()=>call('/jobs','POST',{recruiter_id:+rid.value,title:title.value,company:company.value,location:location.value||'Remote',description:desc.value,required_skills:skills.value.split(',').map(x=>x.trim()).filter(Boolean)}))}
    async function uploadResume(){run('uresult',async()=>{if(!resume.files[0])throw Error('Choose a file first');let f=new FormData();f.append('file',resume.files[0]);let r=await fetch('/candidates/'+cid.value+'/resume',{method:'POST',body:f});let d=await r.json();if(!r.ok)throw Error(JSON.stringify(d));return d})}
    function getMatches(){run('mresult',()=>call('/candidates/'+matchcid.value+'/matches'))}
    function getCandidates(){run('crresult',()=>call('/jobs/'+matchjid.value+'/candidates'))}
    </script></main></body></html>
    """

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/candidates")
def create_candidate(payload: CandidateIn, db: Session = Depends(get_db)):
    if db.scalar(select(Candidate).where(Candidate.email == payload.email)):
        raise HTTPException(409, "A candidate with this email already exists.")
    obj = Candidate(full_name=payload.full_name, email=str(payload.email))
    db.add(obj); db.commit(); db.refresh(obj)
    return {"candidate_id": obj.candidate_id, "full_name": obj.full_name, "email": obj.email}

@app.get("/candidates")
def list_candidates(db: Session = Depends(get_db)):
    return [{"candidate_id": c.candidate_id, "full_name": c.full_name, "email": c.email} for c in db.scalars(select(Candidate).order_by(Candidate.candidate_id))]

@app.post("/recruiters")
def create_recruiter(payload: RecruiterIn, db: Session = Depends(get_db)):
    if db.scalar(select(Recruiter).where(Recruiter.email == payload.email)):
        raise HTTPException(409, "A recruiter with this email already exists.")
    obj = Recruiter(full_name=payload.full_name, email=str(payload.email))
    db.add(obj); db.commit(); db.refresh(obj)
    return {"recruiter_id": obj.recruiter_id, "full_name": obj.full_name, "email": obj.email}

@app.post("/jobs")
def create_job(payload: JobIn, db: Session = Depends(get_db)):
    if not db.get(Recruiter, payload.recruiter_id):
        raise HTTPException(404, "Recruiter not found. Create a recruiter first.")
    obj = JobPosting(recruiter_id=payload.recruiter_id, title=payload.title, company=payload.company,
                     location=payload.location, description=payload.description, min_experience=payload.min_experience)
    db.add(obj); db.flush()
    skills = set(s.strip().lower() for s in payload.required_skills if s.strip()) | extract_skills(payload.description)
    set_job_skills(db, obj, skills)
    db.commit(); db.refresh(obj)
    return {"job_id": obj.job_id, "title": obj.title, "required_skills": sorted(skills)}

@app.get("/jobs")
def list_jobs(db: Session = Depends(get_db)):
    jobs = db.scalars(select(JobPosting).options(joinedload(JobPosting.skills).joinedload(JobSkill.skill)).order_by(desc(JobPosting.created_at))).unique()
    return [{"job_id": j.job_id, "title": j.title, "company": j.company, "location": j.location,
             "description": j.description, "required_skills": sorted(x.skill.name for x in j.skills)} for j in jobs]

@app.post("/candidates/{candidate_id}/resume")
async def upload_resume(candidate_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    candidate = db.get(Candidate, candidate_id)
    if not candidate:
        raise HTTPException(404, "Candidate not found.")
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(413, "Resume must be 5 MB or smaller.")
    text = read_resume(file.filename or "resume.txt", content)
    if not text:
        raise HTTPException(400, "No text could be extracted. Scanned PDFs need OCR, which this demo does not include.")
    resume = Resume(candidate_id=candidate_id, filename=(file.filename or "resume")[:255], raw_text=text)
    db.add(resume); db.flush()
    skills = extract_skills(text)
    set_resume_skills(db, resume, skills)
    db.commit()
    return {"resume_id": resume.resume_id, "candidate_id": candidate_id, "filename": resume.filename,
            "extracted_skills": sorted(skills), "note": "Review extracted skills; this demo uses a dictionary-based extractor."}

def candidate_match_rows(db: Session, candidate: Candidate):
    resume = db.scalar(select(Resume).where(Resume.candidate_id == candidate.candidate_id).order_by(desc(Resume.created_at)))
    if not resume:
        raise HTTPException(400, "Upload a resume for this candidate first.")
    resume_skills = {x.skill.name for x in resume.skills}
    jobs = db.scalars(select(JobPosting).options(joinedload(JobPosting.skills).joinedload(JobSkill.skill))).unique().all()
    results = []
    for job in jobs:
        job_skills = {x.skill.name for x in job.skills}
        score, text_similarity = match_score(resume.raw_text, job.title + " " + job.description, resume_skills, job_skills)
        results.append({"job_id": job.job_id, "title": job.title, "company": job.company, "location": job.location,
                        "match_score": score, "text_similarity": text_similarity,
                        "matched_skills": sorted(resume_skills & job_skills),
                        "missing_skills": sorted(job_skills - resume_skills)})
    return sorted(results, key=lambda x: x["match_score"], reverse=True)

@app.get("/candidates/{candidate_id}/matches")
def candidate_matches(candidate_id: int, db: Session = Depends(get_db)):
    candidate = db.get(Candidate, candidate_id)
    if not candidate: raise HTTPException(404, "Candidate not found.")
    return {"candidate_id": candidate_id, "matches": candidate_match_rows(db, candidate)}

@app.get("/jobs/{job_id}/candidates")
def job_candidates(job_id: int, db: Session = Depends(get_db)):
    job = db.get(JobPosting, job_id)
    if not job: raise HTTPException(404, "Job not found.")
    job_skills = {x.skill.name for x in db.scalars(select(JobSkill).where(JobSkill.job_id == job_id)).all()
                  for x in [db.get(Skill, x.skill_id)]}
    candidates = db.scalars(select(Candidate).options(joinedload(Candidate.resumes).joinedload(Resume.skills).joinedload(ResumeSkill.skill))).unique().all()
    results = []
    for c in candidates:
        if not c.resumes: continue
        resume = sorted(c.resumes, key=lambda r: r.resume_id)[-1]
        resume_skills = {x.skill.name for x in resume.skills}
        score, text_similarity = match_score(resume.raw_text, job.title + " " + job.description, resume_skills, job_skills)
        results.append({"candidate_id": c.candidate_id, "full_name": c.full_name, "match_score": score,
                        "text_similarity": text_similarity, "matched_skills": sorted(resume_skills & job_skills),
                        "missing_skills": sorted(job_skills - resume_skills)})
    return {"job_id": job_id, "candidates": sorted(results, key=lambda x: x["match_score"], reverse=True)}

@app.post("/applications")
def apply(payload: ApplyIn, db: Session = Depends(get_db)):
    candidate = db.get(Candidate, payload.candidate_id)
    job = db.get(JobPosting, payload.job_id)
    if not candidate or not job: raise HTTPException(404, "Candidate or job not found.")
    if db.scalar(select(Application).where(Application.candidate_id == payload.candidate_id, Application.job_id == payload.job_id)):
        raise HTTPException(409, "Candidate has already applied to this job.")
    ranked = candidate_match_rows(db, candidate)
    match = next((x for x in ranked if x["job_id"] == job.job_id), None)
    obj = Application(candidate_id=candidate.candidate_id, job_id=job.job_id, match_score=match["match_score"] if match else 0)
    db.add(obj); db.commit(); db.refresh(obj)
    return {"application_id": obj.application_id, "status": obj.status, "match_score": obj.match_score}

@app.get("/applications")
def list_applications(db: Session = Depends(get_db)):
    rows = db.scalars(select(Application).options(joinedload(Application.candidate), joinedload(Application.job)).order_by(desc(Application.applied_at))).unique()
    return [{"application_id": a.application_id, "candidate_id": a.candidate_id, "candidate": a.candidate.full_name,
             "job_id": a.job_id, "job_title": a.job.title, "company": a.job.company,
             "match_score": a.match_score, "status": a.status, "applied_at": a.applied_at.isoformat()} for a in rows]

@app.patch("/applications/{application_id}/status")
def update_status(application_id: int, payload: StatusIn, db: Session = Depends(get_db)):
    if payload.status not in ALLOWED_STATUSES:
        raise HTTPException(400, f"Status must be one of: {', '.join(sorted(ALLOWED_STATUSES))}")
    obj = db.get(Application, application_id)
    if not obj: raise HTTPException(404, "Application not found.")
    obj.status = payload.status
    db.commit()
    return {"application_id": obj.application_id, "status": obj.status}

@app.get("/analytics")
def analytics(db: Session = Depends(get_db)):
    jobs = db.scalars(select(JobPosting)).all()
    apps = db.scalars(select(Application)).all()
    return {"total_candidates": len(db.scalars(select(Candidate)).all()),
            "total_jobs": len(jobs), "total_applications": len(apps),
            "average_match_score": round(sum(a.match_score for a in apps) / len(apps), 1) if apps else 0,
            "applications_by_status": {status: sum(1 for a in apps if a.status == status) for status in sorted(ALLOWED_STATUSES)},
            "applications_by_job": [{"job_id": j.job_id, "title": j.title,
                                     "application_count": sum(1 for a in apps if a.job_id == j.job_id)} for j in jobs]}
