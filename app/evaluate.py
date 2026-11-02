"""Reproducible synthetic benchmark for ECR-Match and simple baselines.

Run: python -m app.evaluate --output results/evaluation.json
Synthetic labels are illustrative and must not be presented as real-world evidence.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .research_matching import score_pair

RNG = random.Random(20261103)
SKILLS = {
 "software": ["python", "java", "javascript", "typescript", "sql", "postgresql", "fastapi", "react", "docker", "git"],
 "data": ["python", "sql", "pandas", "numpy", "excel", "power bi", "tableau", "statistics", "etl", "postgresql"],
 "ml_ai": ["python", "machine learning", "deep learning", "nlp", "scikit-learn", "pandas", "numpy", "statistics", "sql", "pytorch"],
 "devops": ["linux", "docker", "kubernetes", "aws", "azure", "git", "python", "postgresql", "redis", "terraform"],
}
JOB_TITLES = {"software":"Backend Software Developer", "data":"Data Analyst", "ml_ai":"Machine Learning Engineer", "devops":"DevOps Engineer"}


def build_data(n_candidates=80, n_jobs=24):
    candidates=[]; jobs=[]
    families=list(SKILLS)
    for i in range(n_candidates):
        fam=RNG.choice(families); core=SKILLS[fam]
        picked=set(RNG.sample(core, k=RNG.randint(4, min(8,len(core)))))
        # Include occasional cross-family skills to make matching non-trivial.
        for other in families:
            if other != fam and RNG.random()<0.25: picked.update(RNG.sample(SKILLS[other], 1))
        text=f"{fam} professional experienced with {', '.join(sorted(picked))}. Built practical projects and collaborated on software delivery."
        candidates.append({"id":i,"family":fam,"skills":picked,"text":text})
    for j in range(n_jobs):
        fam=families[j%len(families)]; core=SKILLS[fam]
        req=set(RNG.sample(core, k=RNG.randint(3,5)))
        pref=set(RNG.sample([s for s in core if s not in req], k=2))
        jobs.append({"id":j,"family":fam,"required":req,"preferred":pref,
                     "title":JOB_TITLES[fam],"text":f"{JOB_TITLES[fam]} role. Responsibilities include {', '.join(sorted(req|pref))}."})
    return candidates,jobs


def metrics(rankings, labels, k=5):
    p=[]; r=[]; nd=[]; mrr=[]
    for cid, ranked in rankings.items():
        relevant={jid for (c,jid),v in labels.items() if c==cid and v==1}
        if not relevant: continue
        top=ranked[:k]
        hits=sum(1 for jid in top if jid in relevant)
        p.append(hits/k); r.append(hits/len(relevant))
        dcg=sum((1.0/(__import__('math').log2(i+2))) for i,jid in enumerate(top) if jid in relevant)
        ideal=sum(1.0/(__import__('math').log2(i+2)) for i in range(min(k,len(relevant))))
        nd.append(dcg/ideal if ideal else 0)
        first=next((i+1 for i,jid in enumerate(ranked) if jid in relevant),None)
        mrr.append(1/first if first else 0)
    return {"precision_at_5":round(sum(p)/max(1,len(p)),4),"recall_at_5":round(sum(r)/max(1,len(r)),4),
            "ndcg_at_5":round(sum(nd)/max(1,len(nd)),4),"mrr":round(sum(mrr)/max(1,len(mrr)),4)}


def run():
    candidates,jobs=build_data(); labels={}; scores={"keyword":defaultdict(list),"tfidf":defaultdict(list),"ecr_match":defaultdict(list)}
    for c in candidates:
        for j in jobs:
            overlap=len(c['skills'] & j['required'])/max(1,len(j['required']))
            # Synthetic relevance definition: same broad role family and >=60% required skills.
            rel=int(c['family']==j['family'] and overlap>=0.60)
            labels[(c['id'],j['id'])]=rel
            scores['keyword'][c['id']].append((j['id'],overlap))
            tf=_tfidf(c['text'],j['text'])
            scores['tfidf'][c['id']].append((j['id'],tf))
            ecr=score_pair(c['text'],j['text'],c['skills'],j['required'],j['preferred'])['match_score']
            scores['ecr_match'][c['id']].append((j['id'],ecr))
    rankings={name:{cid:[jid for jid,_ in sorted(rows,key=lambda x:x[1],reverse=True)] for cid,rows in grouped.items()} for name,grouped in scores.items()}
    report={"dataset":{"type":"synthetic","seed":20261103,"candidates":len(candidates),"jobs":len(jobs),"pairs":len(labels),"relevant_pairs":sum(labels.values()),
                       "label_rule":"same broad role family AND at least 60% required-skill coverage"},
            "metrics_at_5":{name:metrics(ranking,labels) for name,ranking in rankings.items()},
            "counterfactual_invariance":{"test":"candidate name/email fields are not inputs to scoring","result":"PASS by design; scorer accepts resume text and skills only"},
            "warning":"Synthetic benchmark only. Do not report these metrics as real-world performance. Validate with consented, representative, human-labelled data before research claims."}
    return report


def _tfidf(a,b):
    try:
        m=TfidfVectorizer(stop_words='english',ngram_range=(1,2)).fit_transform([a,b])
        return float(cosine_similarity(m[0:1],m[1:2])[0,0])
    except ValueError: return 0.0

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='results/evaluation.json'); args=parser.parse_args()
    result=run(); out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
