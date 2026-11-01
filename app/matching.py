import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Lightweight demo skill dictionary. Extend for your domain or replace with an NLP model.
SKILL_TERMS = {
    "python", "java", "javascript", "typescript", "sql", "postgresql", "mysql",
    "mongodb", "fastapi", "django", "flask", "react", "node.js", "html", "css",
    "machine learning", "deep learning", "nlp", "natural language processing",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "docker", "kubernetes",
    "aws", "azure", "git", "linux", "excel", "power bi", "tableau", "communication",
    "project management", "data analysis", "data engineering", "rest api", "api",
    "c++", "c#", "leadership", "agile", "statistics", "etl", "redis"
}

def extract_skills(text: str) -> set[str]:
    normalized = re.sub(r"\s+", " ", text.lower())
    found = set()
    for skill in SKILL_TERMS:
        if re.search(r"(?<![a-z0-9+#.])" + re.escape(skill) + r"(?![a-z0-9+#.])", normalized):
            found.add(skill)
    return found

def match_score(resume_text: str, job_text: str, resume_skills: set[str], job_skills: set[str]) -> tuple[float, float]:
    # 60% explicit skill overlap + 40% TF-IDF text similarity; demonstrative baseline, not a hiring decision.
    union = resume_skills | job_skills
    skill_score = len(resume_skills & job_skills) / max(1, len(job_skills))
    try:
        matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform([resume_text, job_text])
        semantic_proxy = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    except ValueError:
        semantic_proxy = 0.0
    score = 100 * (0.60 * skill_score + 0.40 * semantic_proxy)
    return round(min(100.0, score), 1), round(semantic_proxy * 100, 1)
