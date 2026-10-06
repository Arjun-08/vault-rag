import json,time
from pathlib import Path
from .config import *
from .ingestion import load_directory
from .chunking import chunk_records
from .embeddings import Embedder
from .vector_store import VectorStore
from .retrieval import rerank
from .llm import LocalLLM
from .dataset import load_squad,build_squad_records
from .evaluation import retrieval_metrics,exact_match,token_f1

def build_index(records):
    chunks=chunk_records(records,CHUNK_SIZE_WORDS,CHUNK_OVERLAP_WORDS); emb=Embedder(EMBEDDING_MODEL); print("[INDEX] Generating embeddings"); vec=emb.encode([x["text"] for x in chunks]); store=VectorStore(vec.shape[1]); store.add(vec,chunks); store.save(VECTOR_DIR); print(f"[INDEX] Saved {len(chunks)} chunks"); return store,emb

def build_public_index(limit=300):
    ds=load_squad(limit); return build_index(build_squad_records(ds))
def build_local_index():
    records=load_directory(RAW_DIR)
    if not records: raise RuntimeError("No PDF/DOCX/TXT/MD files found in data/raw/")
    return build_index(records)
def load_components(): return VectorStore.load(VECTOR_DIR),Embedder(EMBEDDING_MODEL)
def ask(question,use_reranker=True,load_llm=True):
    store,emb=load_components(); t=time.perf_counter(); qv=emb.encode([question]); candidates=store.search(qv,CANDIDATE_K if use_reranker else TOP_K); retrieval_ms=(time.perf_counter()-t)*1000; final=rerank(question,candidates,TOP_K) if use_reranker else candidates[:TOP_K]; answer=None; generation_ms=None
    if load_llm:
        llm=LocalLLM(LLM_MODEL); t=time.perf_counter(); answer=llm.answer(question,final); generation_ms=(time.perf_counter()-t)*1000
    return {"answer":answer,"sources":final,"retrieval_ms":retrieval_ms,"generation_ms":generation_ms}
def evaluate(limit=300):
    ds=load_squad(limit); records=build_squad_records(ds); store,emb=build_index(records); results=[]; answers=[]; questions=[]
    for i,row in enumerate(ds,1):
        if i%25==0: print(f"[EVAL] {i}/{len(ds)}")
        q=row["question"]; results.append(rerank(q,store.search(emb.encode([q],batch_size=1),CANDIDATE_K),TOP_K)); answers.append(row["answers"]["text"]); questions.append(q)
    metrics=retrieval_metrics(results,answers,TOP_K); print("[EVAL] Retrieval metrics",json.dumps(metrics,indent=2)); llm=LocalLLM(LLM_MODEL); em=[]; f1=[]; total=0.0
    for i in range(min(20,len(ds))):
        t=time.perf_counter(); pred=llm.answer(questions[i],results[i]); total+=(time.perf_counter()-t)*1000; em.append(exact_match(pred,answers[i])); f1.append(token_f1(pred,answers[i])); print(f"[EVAL-LLM] {i+1}/{min(20,len(ds))}")
    metrics.update(answer_exact_match=sum(em)/len(em),answer_token_f1=sum(f1)/len(f1),avg_generation_ms=total/len(em)); (OUTPUT_DIR/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8"); print(json.dumps(metrics,indent=2)); return metrics
