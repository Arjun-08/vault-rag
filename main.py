import argparse
from src.pipeline import build_public_index,build_local_index,ask,evaluate
p=argparse.ArgumentParser(); p.add_argument("--mode",choices=["build","index-local","evaluate","ask"],required=True); p.add_argument("--eval-limit",type=int,default=300); p.add_argument("--question")
a=p.parse_args()
if a.mode=="build": build_public_index(a.eval_limit)
elif a.mode=="index-local": build_local_index()
elif a.mode=="evaluate": evaluate(a.eval_limit)
elif a.mode=="ask":
    if not a.question: raise ValueError("--question is required")
    r=ask(a.question); print("\nANSWER\n",r["answer"]); print("\nSOURCES")
    for i,s in enumerate(r["sources"],1): print(f"[{i}] {s['document_name']} | score={s['score']:.4f}")
    print(f"Retrieval latency: {r['retrieval_ms']:.2f} ms"); print(f"Generation latency: {r['generation_ms']:.2f} ms")
