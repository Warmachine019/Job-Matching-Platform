"""Research prototype: explainable, constraint-aware, reciprocal job matching.

This is a transparent decision-support baseline, not a validated hiring instrument.
"""
from __future__ import annotations
import re
from typing import Iterable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

WEIGHTS = {
    "required_skill_coverage": 0.45,
    "text_similarity": 0.25,
    "role_similarity": 0.15,
    "preferred_skill_coverage": 0.15,
}

ROLE_TERMS = {
    "software": {"software engineer", "developer", "backend", "frontend", "full stack", "api"},
    "data": {"data analyst", "data scientist", "data engineer", "analytics", "etl", "business intelligence"},
    "ml_ai": {"machine learning", "deep learning", "nlp", "ai engineer", "ml engineer"},
    "devops": {"devops", "cloud engineer", "platform engineer", "site reliability", "kubernetes"},
}

SKILL_LEARNING_PATHS = {
    "python": ["Python fundamentals", "Functions, modules and OOP", "Build a small API or data project"],
    "sql": ["SELECT, filtering and joins", "Aggregation and subqueries", "Window functions and query optimization"],
    "postgresql": ["Relational modelling", "Indexes and query plans", "Transactions and performance tuning"],
    "docker": ["Images and containers", "Dockerfiles and volumes", "Compose a multi-service application"],
    "machine learning": ["Statistics and data preparation", "Supervised learning", "Model evaluation and error analysis"],
    "nlp": ["Text preprocessing", "TF-IDF and embeddings", "Evaluate retrieval and ranking"],
    "react": ["Components and props", "State and hooks", "Build and deploy a small SPA"],
    "aws": ["Cloud fundamentals", "IAM and networking", "Deploy and monitor a small service"],
}


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _cosine(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    try:
        matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform([a, b])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0, 0])
    except ValueError:
        return 0.0


def _role_similarity(candidate_text: str, job_text: str) -> float:
    c, j = _clean(candidate_text), _clean(job_text)
    candidate_roles = {role for role, terms in ROLE_TERMS.items() if any(t in c for t in terms)}
    job_roles = {role for role, terms in ROLE_TERMS.items() if any(t in j for t in terms)}
    if not candidate_roles or not job_roles:
        return 0.0
    return len(candidate_roles & job_roles) / len(job_roles)


def score_pair(
    resume_text: str,
    job_text: str,
    candidate_skills: Iterable[str],
    required_skills: Iterable[str],
    preferred_skills: Iterable[str] = (),
    _include_counterfactual: bool = True,
) -> dict:
    """Return component-level explanations and a reciprocal, constraint-aware score.

    Employer-side fit is required-skill coverage. Candidate-side fit combines semantic
    text, role-family similarity, and preferred-skill coverage. Their harmonic mean
    penalizes one-sided matches. Missing required skills trigger a configurable cap
    (55) so a high text similarity cannot conceal a hard requirement gap.
    """
    cs = {_clean(x) for x in candidate_skills if _clean(x)}
    req = {_clean(x) for x in required_skills if _clean(x)}
    pref = {_clean(x) for x in preferred_skills if _clean(x)} - req
    matched_req = sorted(cs & req)
    missing_req = sorted(req - cs)
    matched_pref = sorted(cs & pref)
    missing_pref = sorted(pref - cs)
    req_cov = len(matched_req) / max(1, len(req)) if req else 1.0
    pref_cov = len(matched_pref) / max(1, len(pref)) if pref else 0.0
    text_sim = _cosine(resume_text, job_text)
    role_sim = _role_similarity(resume_text, job_text)

    candidate_fit = 100 * (
        WEIGHTS["text_similarity"] * text_sim
        + WEIGHTS["role_similarity"] * role_sim
        + WEIGHTS["preferred_skill_coverage"] * pref_cov
        + WEIGHTS["required_skill_coverage"] * req_cov
    )
    employer_fit = 100 * req_cov
    # Harmonic mean discourages a high score on one side masking a low score on the other.
    reciprocal = (2 * candidate_fit * employer_fit / (candidate_fit + employer_fit)) if (candidate_fit + employer_fit) else 0.0
    final_score = reciprocal
    constraint_warning = bool(missing_req)
    if constraint_warning:
        final_score = min(final_score, 55.0)

    explanation = []
    explanation.append(f"Required skills: {len(matched_req)}/{len(req)} matched ({req_cov*100:.0f}%).")
    explanation.append(f"Text relevance: {text_sim*100:.0f}% using TF-IDF cosine similarity.")
    explanation.append(f"Role-family alignment: {role_sim*100:.0f}% using transparent role vocabulary.")
    if pref:
        explanation.append(f"Preferred skills: {len(matched_pref)}/{len(pref)} matched ({pref_cov*100:.0f}%).")
    if missing_req:
        explanation.append("Constraint warning: at least one required skill is missing; score capped at 55.")
    else:
        explanation.append("All listed required skills are present in the extracted skill set.")

    next_skills = []
    for skill in missing_req + missing_pref:
        item = {"skill": skill, "priority": "required" if skill in missing_req else "preferred",
                "learning_steps": SKILL_LEARNING_PATHS.get(skill, [f"Study {skill} fundamentals", f"Complete a practical {skill} exercise", f"Add a portfolio project demonstrating {skill}"])}
        if _include_counterfactual:
            projected = score_pair(resume_text, job_text, cs | {skill}, req, pref, _include_counterfactual=False)
            item["projected_match_score_if_added"] = projected["match_score"]
            item["score_delta"] = round(projected["match_score"] - final_score, 1)
        next_skills.append(item)

    return {
        "match_score": round(max(0.0, min(100.0, final_score)), 1),
        "candidate_fit": round(candidate_fit, 1),
        "employer_fit": round(employer_fit, 1),
        "required_skill_coverage": round(req_cov * 100, 1),
        "preferred_skill_coverage": round(pref_cov * 100, 1),
        "text_similarity": round(text_sim * 100, 1),
        "role_similarity": round(role_sim * 100, 1),
        "matched_skills": matched_req + matched_pref,
        "missing_skills": missing_req + missing_pref,
        "missing_required_skills": missing_req,
        "missing_preferred_skills": missing_pref,
        "constraint_warning": constraint_warning,
        "explanation": explanation,
        "skill_gap_plan": next_skills,
        "method": "ECR-Match v1 (Explainable Constraint-aware Reciprocal Matching)",
    }
