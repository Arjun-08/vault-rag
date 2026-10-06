from pathlib import Path
import json, faiss
class VectorStore:
    def __init__(self,dimension): self.index=faiss.IndexFlatIP(dimension); self.metadata=[]
    def add(self,embeddings,metadata): self.index.add(embeddings); self.metadata.extend(metadata)
    def search(self,q,k=5):
        k=min(k,self.index.ntotal)
        if k==0:return []
        scores,idx=self.index.search(q,k); out=[]
        for score,i in zip(scores[0],idx[0]):
            if i<0:continue
            x=dict(self.metadata[int(i)]); x["score"]=float(score); out.append(x)
        return out
    def save(self,directory):
        directory.mkdir(parents=True,exist_ok=True); faiss.write_index(self.index,str(directory/"index.faiss")); (directory/"metadata.json").write_text(json.dumps(self.metadata,ensure_ascii=False),encoding="utf-8")
    @classmethod
    def load(cls,directory):
        index=faiss.read_index(str(directory/"index.faiss")); obj=cls(index.d); obj.index=index; obj.metadata=json.loads((directory/"metadata.json").read_text(encoding="utf-8")); return obj
