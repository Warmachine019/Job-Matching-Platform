from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select, desc, text
from pypdf import PdfReader
from docx import Document
from .database import Base, engine, get_db
from .models import Candidate, Recruiter, Skill, Resume, ResumeSkill, JobPosting, JobSkill, Application
from .matching import extract_skills
from .research_matching import score_pair

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    # Lightweight forward migration for databases created by Version 2.
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE job_posting ADD COLUMN IF NOT EXISTS preferred_skills TEXT NOT NULL DEFAULT ''"))
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
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
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
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TalentMatch | Recruitment Workspace</title>
<style>
:root{--navy:#14213d;--blue:#315fce;--teal:#15998f;--bg:#f4f6fb;--ink:#1d2939;--muted:#667085;--line:#e5eaf2;--white:#fff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Inter,Segoe UI,Arial,sans-serif}
header{background:linear-gradient(120deg,#14213d,#233f73);color:white;padding:25px max(24px,calc((100% - 1240px)/2))}
header .brand{display:flex;align-items:center;gap:12px}header .logo{width:42px;height:42px;border-radius:12px;background:#ffffff20;display:grid;place-items:center;font-size:22px}
header h1{font-size:23px;margin:0}header p{margin:4px 0 0;color:#d3def1}
main{max-width:1240px;margin:24px auto;padding:0 20px}
.nav{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px}.nav button{width:auto;margin:0;background:#e8edf7;color:#344563;padding:10px 16px}
.nav button.active{background:var(--navy);color:white}
button{border:0;border-radius:9px;padding:10px 14px;background:var(--blue);color:white;font-weight:650;cursor:pointer}
button:hover{filter:brightness(.96)}button.secondary{background:#e9eef8;color:#253858}button.small{padding:6px 9px;font-size:12px}
input,textarea,select{width:100%;padding:10px 11px;border:1px solid #d6deea;border-radius:9px;background:white;color:var(--ink);font:inherit;margin:5px 0 10px}
textarea{min-height:90px;resize:vertical}label{display:block;font-size:12px;color:#526078;font-weight:650}
.panel{display:none}.panel.active{display:block}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:16px}
.card{background:white;border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:0 3px 14px #14213d08}
.stat{grid-column:span 3}.stat .num{font-size:28px;font-weight:750;color:var(--navy)}.stat .label{color:var(--muted)}
.span6{grid-column:span 6}.span12{grid-column:span 12}.card h2{font-size:16px;margin:0 0 14px}.muted{color:var(--muted)}
.formgrid{display:grid;grid-template-columns:1fr 1fr;gap:0 12px}.full{grid-column:1/-1}
.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;min-width:620px}th,td{text-align:left;padding:11px 9px;border-bottom:1px solid var(--line);vertical-align:top}th{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
.pill{display:inline-block;padding:3px 8px;border-radius:99px;background:#edf2ff;color:#315fce;font-size:11px;font-weight:700}
.score{font-weight:800;color:#167c72}.result{background:#f7f9fc;border:1px solid var(--line);padding:12px;border-radius:10px;margin-top:12px}
.notice{padding:11px 13px;border-radius:10px;background:#eef7ff;color:#254b78;margin-bottom:16px}.error{background:#fff0ef;color:#9b2c24}.success{background:#eaf8f2;color:#176b4d}
.actions{display:flex;gap:8px;flex-wrap:wrap}.empty{padding:22px;text-align:center;color:var(--muted)}
@media(max-width:800px){.stat{grid-column:span 6}.span6{grid-column:span 12}}
@media(max-width:480px){.stat{grid-column:span 12}.formgrid{grid-template-columns:1fr}.full{grid-column:auto}}
</style>
</head>
<body>
<header><div class="brand"><div class="logo">✦</div><div><h1>TalentMatch</h1><p>Job matching & recruitment workspace</p></div></div></header>
<main>
<div id="notice" class="notice">Welcome. This academic demo uses skill overlap and TF-IDF similarity. Matching scores are advisory only.</div>
<nav class="nav">
<button class="active" data-tab="dashboard">Overview</button><button data-tab="people">People</button><button data-tab="jobs">Jobs</button><button data-tab="matching">Matching</button><button data-tab="applications">Applications</button>
</nav>

<section id="dashboard" class="panel active">
<div class="grid">
<div class="card stat"><div class="label">Candidates</div><div class="num" id="statCandidates">—</div></div>
<div class="card stat"><div class="label">Recruiters</div><div class="num" id="statRecruiters">—</div></div>
<div class="card stat"><div class="label">Open jobs</div><div class="num" id="statJobs">—</div></div>
<div class="card stat"><div class="label">Applications</div><div class="num" id="statApplications">—</div></div>
<div class="card span6"><h2>Recent jobs</h2><div id="recentJobs" class="tablewrap"></div></div>
<div class="card span6"><h2>Application pipeline</h2><div id="pipeline"></div></div>
<div class="card span12"><div class="actions"><button class="secondary" onclick="refreshAll()">↻ Refresh dashboard</button><a href="/docs" target="_blank"><button class="secondary">Open API docs ↗</button></a></div></div>
</div>
</section>

<section id="people" class="panel">
<div class="grid">
<div class="card span6"><h2>Register candidate</h2><form id="candidateForm" class="formgrid">
<div><label>Full name</label><input name="full_name" required minlength="2" placeholder="e.g. Aarav Sharma"></div>
<div><label>Email</label><input name="email" type="email" required placeholder="candidate@example.com"></div>
<div class="full"><button>Create candidate</button></div></form></div>
<div class="card span6"><h2>Register recruiter</h2><form id="recruiterForm" class="formgrid">
<div><label>Full name</label><input name="full_name" required minlength="2" placeholder="e.g. Demo Recruiter"></div>
<div><label>Email</label><input name="email" type="email" required placeholder="recruiter@example.com"></div>
<div class="full"><button>Create recruiter</button></div></form></div>
<div class="card span12"><h2>Candidate directory</h2><div id="candidateTable" class="tablewrap"></div></div>
<div class="card span12"><h2>Recruiter directory</h2><div id="recruiterTable" class="tablewrap"></div></div>
</div></section>

<section id="jobs" class="panel"><div class="grid">
<div class="card span6"><h2>Post a job</h2><form id="jobForm" class="formgrid">
<div><label>Recruiter ID</label><input name="recruiter_id" type="number" min="1" required></div>
<div><label>Job title</label><input name="title" required placeholder="Junior Python Developer"></div>
<div><label>Company</label><input name="company" required></div><div><label>Location</label><input name="location" value="Remote"></div>
<div><label>Minimum experience (years)</label><input name="min_experience" type="number" min="0" value="0"></div>
<div><label>Required skills (comma separated)</label><input name="required_skills" placeholder="python, sql, fastapi"></div>
<div><label>Preferred skills (comma separated)</label><input name="preferred_skills" placeholder="docker, aws, git"></div>
<div class="full"><label>Job description</label><textarea name="description" required placeholder="Describe responsibilities and requirements..."></textarea></div>
<div class="full"><button>Publish job</button></div></form></div>
<div class="card span6"><h2>Upload candidate resume</h2><form id="resumeForm">
<label>Candidate ID</label><input name="candidate_id" type="number" min="1" required>
<label>Resume file (.txt, .pdf, .docx; max 5 MB)</label><input name="file" type="file" accept=".txt,.pdf,.docx" required>
<button>Upload & extract skills</button></form><div id="resumeResult"></div></div>
<div class="card span12"><h2>Job board</h2><div id="jobTable" class="tablewrap"></div></div>
</div></section>

<section id="matching" class="panel"><div class="grid">
<div class="card span6"><h2>Find jobs for a candidate</h2><form id="candidateMatchForm"><label>Candidate ID</label><input name="candidate_id" type="number" min="1" required><button>Rank matching jobs</button></form><div id="candidateMatches"></div></div>
<div class="card span6"><h2>Find candidates for a job</h2><form id="jobMatchForm"><label>Job ID</label><input name="job_id" type="number" min="1" required><button>Rank candidates</button></form><div id="jobMatches"></div></div>
</div></section>

<section id="applications" class="panel"><div class="grid">
<div class="card span6"><h2>Submit application</h2><form id="applyForm" class="formgrid">
<div><label>Candidate ID</label><input name="candidate_id" type="number" min="1" required></div>
<div><label>Job ID</label><input name="job_id" type="number" min="1" required></div>
<div class="full"><button>Submit application</button></div></form></div>
<div class="card span12"><h2>Application tracker</h2><div id="applicationTable" class="tablewrap"></div></div>
</div></section>
<footer class="muted" style="padding:24px 2px">TalentMatch academic prototype · Use fictional data for demos · Human review is required for hiring decisions.</footer>
</main>
<script>
const $ = id => document.getElementById(id);
async function api(url, options={}) {
  const res = await fetch(url, options);
  const data = await res.json().catch(()=>({detail:"Unexpected server response"}));
  if(!res.ok) throw new Error(data.detail || JSON.stringify(data));
  return data;
}
function notify(message, kind="success") { const n=$("notice"); n.textContent=message; n.className="notice "+(kind==="error"?"error":"success"); }
function esc(v){return String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function table(headers, rows){if(!rows.length)return '<div class="empty">No records yet.</div>';return '<table><thead><tr>'+headers.map(h=>'<th>'+h+'</th>').join('')+'</tr></thead><tbody>'+rows.join('')+'</tbody></table>';}
document.querySelectorAll("[data-tab]").forEach(b=>b.addEventListener("click",()=>{document.querySelectorAll("[data-tab]").forEach(x=>x.classList.toggle("active",x===b));document.querySelectorAll(".panel").forEach(x=>x.classList.toggle("active",x.id===b.dataset.tab));}));
function formObject(form){return Object.fromEntries(new FormData(form).entries());}
async function refreshAll(){
 try{
  const [c,r,j,a,stats]=await Promise.all([api("/candidates"),api("/recruiters"),api("/jobs"),api("/applications"),api("/analytics")]);
  $("statCandidates").textContent=stats.total_candidates;$("statRecruiters").textContent=r.length;$("statJobs").textContent=stats.total_jobs;$("statApplications").textContent=stats.total_applications;
  $("candidateTable").innerHTML=table(["ID","Name","Email"],c.map(x=>`<tr><td>${x.candidate_id}</td><td>${esc(x.full_name)}</td><td>${esc(x.email)}</td></tr>`));
  $("recruiterTable").innerHTML=table(["ID","Name","Email"],r.map(x=>`<tr><td>${x.recruiter_id}</td><td>${esc(x.full_name)}</td><td>${esc(x.email)}</td></tr>`));
  $("jobTable").innerHTML=table(["ID","Role","Company","Location","Skills"],j.map(x=>`<tr><td>${x.job_id}</td><td><b>${esc(x.title)}</b></td><td>${esc(x.company)}</td><td>${esc(x.location)}</td><td>${x.required_skills.map(s=>`<span class="pill">${esc(s)}</span>`).join(" ")}</td></tr>`));
  $("recentJobs").innerHTML=table(["Role","Company","Location"],j.slice(0,5).map(x=>`<tr><td>${esc(x.title)}</td><td>${esc(x.company)}</td><td>${esc(x.location)}</td></tr>`));
  $("pipeline").innerHTML=Object.entries(stats.applications_by_status).map(([k,v])=>`<p style="display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding:5px 0"><span>${esc(k)}</span><b>${v}</b></p>`).join("");
  $("applicationTable").innerHTML=applicationRows(a);
 }catch(e){notify(e.message,"error");}
}
function applicationRows(a){return table(["ID","Candidate","Position","Match","Status","Update"],a.map(x=>`<tr><td>${x.application_id}</td><td>${esc(x.candidate)}</td><td>${esc(x.job_title)} · ${esc(x.company)}</td><td class="score">${x.match_score}%</td><td><span class="pill">${esc(x.status)}</span></td><td><select aria-label="Update status" onchange="updateStatus(${x.application_id},this.value)"><option value="">Change status…</option>${["Applied","Shortlisted","Interview","Offer","Rejected"].map(s=>`<option>${s}</option>`).join("")}</select></td></tr>`));}
$("candidateForm").addEventListener("submit",async e=>{e.preventDefault();try{const d=await api("/candidates",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(formObject(e.target))});notify(`Candidate created. ID: ${d.candidate_id}`);e.target.reset();refreshAll();}catch(x){notify(x.message,"error");}});
$("recruiterForm").addEventListener("submit",async e=>{e.preventDefault();try{const d=await api("/recruiters",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(formObject(e.target))});notify(`Recruiter created. ID: ${d.recruiter_id}`);e.target.reset();refreshAll();}catch(x){notify(x.message,"error");}});
$("jobForm").addEventListener("submit",async e=>{e.preventDefault();try{const f=formObject(e.target);f.recruiter_id=Number(f.recruiter_id);f.min_experience=Number(f.min_experience);f.required_skills=f.required_skills.split(",").map(s=>s.trim()).filter(Boolean);f.preferred_skills=(f.preferred_skills||"").split(",").map(s=>s.trim()).filter(Boolean);const d=await api("/jobs",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(f)});notify(`Job published. ID: ${d.job_id}`);e.target.reset();refreshAll();}catch(x){notify(x.message,"error");}});
$("resumeForm").addEventListener("submit",async e=>{e.preventDefault();try{const f=new FormData(e.target);const id=f.get("candidate_id");const d=await api(`/candidates/${id}/resume`,{method:"POST",body:new FormData(e.target)});$("resumeResult").innerHTML=`<div class="result"><b>Resume processed.</b><p>Extracted skills:</p>${d.extracted_skills.map(s=>`<span class="pill">${esc(s)}</span>`).join(" ")}</div>`;notify("Resume uploaded and parsed.");}catch(x){notify(x.message,"error");}});
$("candidateMatchForm").addEventListener("submit",async e=>{e.preventDefault();try{const d=await api(`/candidates/${formObject(e.target).candidate_id}/matches`);$("candidateMatches").innerHTML='<div class="result">'+(d.matches.length?d.matches.map(m=>`<div style="padding:12px 0;border-bottom:1px solid var(--line)"><b>${esc(m.title)} · ${esc(m.company)}</b><div class="score">${m.match_score}% match</div><p>Matched: ${m.matched_skills.map(esc).join(", ")||"None"}</p><p class="muted">Missing: ${m.missing_skills.map(esc).join(", ")||"None"}</p><p class="muted">${(m.explanation||[]).map(esc).join(" ")}</p>${(m.skill_gap_plan||[]).length?`<p><b>Skill-gap what-if:</b> ${(m.skill_gap_plan||[]).map(g=>`${esc(g.skill)}: ${g.projected_match_score_if_added}% if added (+${g.score_delta})`).join(" · ")}</p>`:""}<button class="small" onclick="quickApply(${d.candidate_id},${m.job_id})">Apply</button></div>`).join(""):"No jobs found.")+'</div>';}catch(x){notify(x.message,"error");}});
$("jobMatchForm").addEventListener("submit",async e=>{e.preventDefault();try{const d=await api(`/jobs/${formObject(e.target).job_id}/candidates`);$("jobMatches").innerHTML='<div class="result">'+(d.candidates.length?d.candidates.map(m=>`<div style="padding:12px 0;border-bottom:1px solid var(--line)"><b>${esc(m.full_name)}</b><div class="score">${m.match_score}% match</div><p>Matched: ${m.matched_skills.map(esc).join(", ")||"None"}</p><p class="muted">Missing: ${m.missing_skills.map(esc).join(", ")||"None"}</p><p class="muted">${(m.explanation||[]).map(esc).join(" ")}</p>${(m.skill_gap_plan||[]).length?`<p><b>Skill-gap what-if:</b> ${(m.skill_gap_plan||[]).map(g=>`${esc(g.skill)}: ${g.projected_match_score_if_added}% if added (+${g.score_delta})`).join(" · ")}</p>`:""}</div>`).join(""):"No candidates with resumes found.")+'</div>';}catch(x){notify(x.message,"error");}});
$("applyForm").addEventListener("submit",async e=>{e.preventDefault();try{const f=formObject(e.target);f.candidate_id=Number(f.candidate_id);f.job_id=Number(f.job_id);const d=await api("/applications",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(f)});notify(`Application created. ID: ${d.application_id}`);e.target.reset();refreshAll();}catch(x){notify(x.message,"error");}});
async function quickApply(candidate_id,job_id){try{const d=await api("/applications",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({candidate_id,job_id})});notify(`Application created. ID: ${d.application_id}`);refreshAll();}catch(x){notify(x.message,"error");}}
async function updateStatus(id,status){if(!status)return;try{await api(`/applications/${id}/status`,{method:"PATCH",headers:{"Content-Type":"application/json"},body:JSON.stringify({status})});notify(`Application ${id} updated to ${status}.`);refreshAll();}catch(x){notify(x.message,"error");}}
refreshAll();
</script></body></html>
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
                     location=payload.location, description=payload.description, preferred_skills=",".join(sorted({s.strip().lower() for s in payload.preferred_skills if s.strip()} - {s.strip().lower() for s in payload.required_skills if s.strip()})), min_experience=payload.min_experience)
    db.add(obj); db.flush()
    skills = {s.strip().lower() for s in payload.required_skills if s.strip()}
    if not skills:
        skills = extract_skills(payload.description)
    set_job_skills(db, obj, skills)
    db.commit(); db.refresh(obj)
    return {"job_id": obj.job_id, "title": obj.title, "required_skills": sorted(skills), "preferred_skills": sorted({s.strip().lower() for s in payload.preferred_skills if s.strip()})}

@app.get("/jobs")
def list_jobs(db: Session = Depends(get_db)):
    jobs = db.scalars(select(JobPosting).options(joinedload(JobPosting.skills).joinedload(JobSkill.skill)).order_by(desc(JobPosting.created_at))).unique()
    return [{"job_id": j.job_id, "title": j.title, "company": j.company, "location": j.location,
             "description": j.description, "required_skills": sorted(x.skill.name for x in j.skills), "preferred_skills": sorted(s for s in (j.preferred_skills or "").split(",") if s)} for j in jobs]

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
        required = {x.skill.name for x in job.skills}
        preferred = {s.strip().lower() for s in (job.preferred_skills or "").split(",") if s.strip()}
        scored = score_pair(resume.raw_text, job.title + " " + job.description, resume_skills, required, preferred)
        results.append({"job_id": job.job_id, "title": job.title, "company": job.company, "location": job.location,
                        **scored})
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
    required = {x.skill.name for x in db.scalars(select(JobSkill).where(JobSkill.job_id == job_id)).all()
                for x in [db.get(Skill, x.skill_id)]}
    preferred = {s.strip().lower() for s in (job.preferred_skills or "").split(",") if s.strip()}
    candidates = db.scalars(select(Candidate).options(joinedload(Candidate.resumes).joinedload(Resume.skills).joinedload(ResumeSkill.skill))).unique().all()
    results = []
    for c in candidates:
        if not c.resumes: continue
        resume = sorted(c.resumes, key=lambda r: r.resume_id)[-1]
        resume_skills = {x.skill.name for x in resume.skills}
        scored = score_pair(resume.raw_text, job.title + " " + job.description, resume_skills, required, preferred)
        results.append({"candidate_id": c.candidate_id, "full_name": c.full_name, **scored})
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


@app.get("/research/method")
def research_method():
    return {
        "name": "ECR-Match v1: Explainable Constraint-aware Reciprocal Matching",
        "weights": {"required_skill_coverage": 0.45, "text_similarity": 0.25, "role_similarity": 0.15, "preferred_skill_coverage": 0.15},
        "reciprocal_aggregation": "harmonic mean of candidate-fit and employer-fit",
        "constraint_rule": "If any required skill is missing, final score is capped at 55/100.",
        "outputs": ["component scores", "matched/missing skills", "human-readable explanation", "skill-gap learning steps"],
        "limitations": ["dictionary-based skill extraction", "TF-IDF is lexical rather than transformer semantics", "role vocabulary is manually curated", "not validated for real hiring decisions"],
    }
