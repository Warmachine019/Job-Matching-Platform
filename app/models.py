from datetime import datetime, timezone
from sqlalchemy import String, Text, Integer, Float, DateTime, ForeignKey, UniqueConstraint, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .database import Base

def now_utc():
    return datetime.now(timezone.utc)

class Candidate(Base):
    __tablename__ = "candidate"
    candidate_id: Mapped[int] = mapped_column(primary_key=True)
    full_name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    resumes: Mapped[list["Resume"]] = relationship(back_populates="candidate", cascade="all, delete-orphan")
    applications: Mapped[list["Application"]] = relationship(back_populates="candidate")

class Recruiter(Base):
    __tablename__ = "recruiter"
    recruiter_id: Mapped[int] = mapped_column(primary_key=True)
    full_name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    jobs: Mapped[list["JobPosting"]] = relationship(back_populates="recruiter")

class Skill(Base):
    __tablename__ = "skill"
    skill_id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True, index=True, nullable=False)

class Resume(Base):
    __tablename__ = "resume"
    resume_id: Mapped[int] = mapped_column(primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidate.candidate_id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    candidate: Mapped["Candidate"] = relationship(back_populates="resumes")
    skills: Mapped[list["ResumeSkill"]] = relationship(back_populates="resume", cascade="all, delete-orphan")

class ResumeSkill(Base):
    __tablename__ = "resume_skill"
    resume_id: Mapped[int] = mapped_column(ForeignKey("resume.resume_id", ondelete="CASCADE"), primary_key=True)
    skill_id: Mapped[int] = mapped_column(ForeignKey("skill.skill_id", ondelete="CASCADE"), primary_key=True)
    resume: Mapped["Resume"] = relationship(back_populates="skills")
    skill: Mapped["Skill"] = relationship()

class JobPosting(Base):
    __tablename__ = "job_posting"
    job_id: Mapped[int] = mapped_column(primary_key=True)
    recruiter_id: Mapped[int] = mapped_column(ForeignKey("recruiter.recruiter_id"), index=True)
    title: Mapped[str] = mapped_column(String(180), nullable=False, index=True)
    company: Mapped[str] = mapped_column(String(180), nullable=False)
    location: Mapped[str] = mapped_column(String(180), default="Remote")
    description: Mapped[str] = mapped_column(Text, nullable=False)
    min_experience: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    recruiter: Mapped["Recruiter"] = relationship(back_populates="jobs")
    skills: Mapped[list["JobSkill"]] = relationship(back_populates="job", cascade="all, delete-orphan")
    applications: Mapped[list["Application"]] = relationship(back_populates="job")

class JobSkill(Base):
    __tablename__ = "job_skill"
    job_id: Mapped[int] = mapped_column(ForeignKey("job_posting.job_id", ondelete="CASCADE"), primary_key=True)
    skill_id: Mapped[int] = mapped_column(ForeignKey("skill.skill_id", ondelete="CASCADE"), primary_key=True)
    job: Mapped["JobPosting"] = relationship(back_populates="skills")
    skill: Mapped["Skill"] = relationship()

class Application(Base):
    __tablename__ = "application"
    __table_args__ = (
        UniqueConstraint("candidate_id", "job_id", name="uq_candidate_job"),
        CheckConstraint("match_score >= 0 AND match_score <= 100", name="ck_match_score"),
    )
    application_id: Mapped[int] = mapped_column(primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidate.candidate_id"), index=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("job_posting.job_id"), index=True)
    match_score: Mapped[float] = mapped_column(Float, default=0)
    status: Mapped[str] = mapped_column(String(30), default="Applied")
    applied_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    candidate: Mapped["Candidate"] = relationship(back_populates="applications")
    job: Mapped["JobPosting"] = relationship(back_populates="applications")
