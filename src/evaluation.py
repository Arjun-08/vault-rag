import re
def normalize_answer(s): return " ".join(re.sub(r"\b(a|an|the)\b"," ",re.sub(r"[^a-z0-9\s]"," ",s.lower())).split())
def exact_match(pred,refs): return float(any(normalize_answer(pred)==normalize_answer(r) for r in refs))
def token_f1(pred,refs):
    p=normalize_answer(pred).split(); best=0.0
    for r in refs:
        q=normalize_answer(r).split(); common=sum(min(p.count(t),q.count(t)) for t in set(p)&set(q))
        score=1.0 if not p and not q else 0.0 if not p or not q or common==0 else 2*(common/len(p))*(common/len(q))/((common/len(p))+(common/len(q)))
        best=max(best,score)
    return best
def retrieval_metrics(results,answers,k):
    hits=[]; precisions=[]; rrs=[]
    for ret,refs in zip(results,answers):
        refs=[normalize_answer(x) for x in refs]; flags=[any(a in normalize_answer(x["text"]) for a in refs if a) for x in ret[:k]]
        hits.append(float(any(flags))); precisions.append(sum(flags)/max(len(flags),1)); rr=0.0
        for rank,f in enumerate(flags,1):
            if f: rr=1/rank; break
        rrs.append(rr)
    return {f"hit_rate@{k}":sum(hits)/len(hits),f"precision@{k}":sum(precisions)/len(precisions),"mrr":sum(rrs)/len(rrs)}
