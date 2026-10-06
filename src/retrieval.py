import re
from collections import Counter
def _tokens(x): return re.findall(r"\b\w+\b",x.lower())
def lexical_overlap(q,t):
    qt=Counter(_tokens(q)); dt=Counter(_tokens(t));
    if not qt:return 0.0
    return sum(min(qt[x],dt[x]) for x in qt)/max(sum(qt.values()),1)
def rerank(query,candidates,top_k=5):
    if not candidates:return []
    m=max(abs(x.get("score",0.0)) for x in candidates) or 1.0; out=[]
    for x in candidates:
        y=dict(x); y["dense_score"]=x.get("score",0.0); y["lexical_score"]=lexical_overlap(query,x["text"]); y["rerank_score"]=0.75*(y["dense_score"]/m)+0.25*y["lexical_score"]; out.append(y)
    out.sort(key=lambda z:z["rerank_score"],reverse=True); return out[:top_k]
