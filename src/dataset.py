from datasets import load_dataset
def load_squad(limit=None):
    print("[DATA] Loading SQuAD validation split from Hugging Face")
    ds=load_dataset("rajpurkar/squad",split="validation")
    if limit: ds=ds.select(range(min(limit,len(ds))))
    print(f"[DATA] Evaluation examples: {len(ds)}"); return ds
def build_squad_records(ds):
    records=[]; seen=set()
    for row in ds:
        key=(row["title"],row["context"])
        if key in seen: continue
        seen.add(key); records.append({"document_id":str(row["id"]),"document_name":f"SQuAD_Wikipedia_{row['title']}.txt","page":None,"text":row["context"]})
    print(f"[DATA] Unique contexts: {len(records)}"); return records
