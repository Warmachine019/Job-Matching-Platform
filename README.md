# TalentMatch — Job Matching & Recruitment Platform

An improved, runnable academic MVP based on the supplied project presentation: candidate/recruiter portals, resume parsing, job posting, skill extraction, ranked matches, application tracking, and recruiter analytics backed by a normalized PostgreSQL schema.

## Quick start (Docker recommended)

Install Docker Desktop (Windows/macOS) or Docker Engine + Compose (Linux). From this folder run:

```bash
docker compose up --build
```

Wait until the API starts, then open:
- Demo UI: http://localhost:8000/
- Interactive API: http://localhost:8000/docs
- Health check: http://localhost:8000/health

Stop with `Ctrl+C`; use `docker compose down`. To also delete the demo database, run `docker compose down -v`.

## Demo walkthrough

1. In the UI, create a candidate (e.g. `Demo Candidate`, `candidate@example.com`). Note the candidate ID.
2. Create a recruiter (e.g. `Demo Recruiter`, `recruiter@example.com`). Note the recruiter ID.
3. Post a job, using that recruiter ID. Example title: `Junior Python Developer`; company: `Example Labs`; description: `Build REST API services with Python, FastAPI, PostgreSQL, SQL, Docker and Git.` Required skills: `python, fastapi, postgresql, sql, docker, git`.
4. Upload a `.txt`, `.pdf` or `.docx` resume for the candidate ID. The repository includes `sample_resume.txt`.
5. Click **Rank jobs for candidate** and enter the candidate ID. The result includes a score, matched skills and missing skills.
6. Click **Rank candidates for job** and enter the job ID.
7. Use API docs to create an application:
   `POST /applications` with JSON `{"candidate_id": 1, "job_id": 1}`.
8. See the pipeline at `GET /applications`, update status using `PATCH /applications/1/status` with `{"status":"Shortlisted"}`, and view dashboard metrics at `GET /analytics`.

IDs start at 1 only in a fresh database; use IDs returned by the API/UI.

## Local Python run (without Docker)

Requires Python 3.11+ and PostgreSQL 16 with a database/user matching your `DATABASE_URL`.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
# Set DATABASE_URL for your local PostgreSQL, then:
uvicorn app.main:app --reload
```

## API overview

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| POST / GET | `/candidates` | Create/list candidates |
| POST | `/recruiters` | Create recruiter |
| POST / GET | `/jobs` | Create/list jobs |
| POST | `/candidates/{id}/resume` | Upload and parse resume |
| GET | `/candidates/{id}/matches` | Rank jobs for a candidate |
| GET | `/jobs/{id}/candidates` | Recruiter-side ranked shortlist |
| POST / GET | `/applications` | Apply/list applications |
| PATCH | `/applications/{id}/status` | Update recruitment status |
| GET | `/analytics` | Dashboard metrics |

## Architecture / database concepts

- Normalized tables: `candidate`, `recruiter`, `resume`, `skill`, `resume_skill`, `job_posting`, `job_skill`, `application`.
- Primary keys and foreign keys enforce identity and referential integrity; unique constraints prevent duplicate candidate/job applications.
- Indexes are added to frequently filtered identifiers, candidate email, and job title.
- The baseline matching score is `60% × required-skill coverage + 40% × TF-IDF cosine similarity`. It is transparent and explainable, but is a lexical similarity proxy, not a transformer embedding model.
- The uploaded presentation proposes pgvector embeddings, HNSW/IVFFlat vector indexes, materialized views, partitioning and PgBouncer. Those are **not yet implemented** in this MVP. This version uses TF-IDF to keep setup light and deterministic. See `docs/NEXT_STEPS.md` for a path to the full design.

## Limitations and safety

This is a class/demo MVP, not production recruitment software. It has no login/role enforcement, malware scanning, OCR for scanned PDFs, email alerts, or production privacy controls. Do not upload real personal data. The matching score is decision support only; a human should review candidates. Add authentication, authorization, audit logging, retention/deletion controls, bias evaluation and security review before deployment.


## Version 2 UI improvements

The homepage now has a responsive workspace with:
- Overview cards for candidates, recruiters, jobs and applications
- Candidate and recruiter directories
- Job board with required skill tags
- Resume upload with extracted-skill display
- Candidate-to-job and job-to-candidate match result panels
- One-click application from a candidate match
- Application tracker with status updates
- Inline success/error messages and refreshable dashboard

No frontend build tools are required; the interface is served by FastAPI.

## Updating from an older version

If you use Docker Compose, replace the project files with this version and run:

```powershell
docker compose up --build
```

Your existing Docker database volume is preserved by default. If you want to start with an empty demo database, first back up anything important, then run `docker compose down -v` followed by `docker compose up --build`.
