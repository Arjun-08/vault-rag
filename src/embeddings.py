import numpy as np
from sentence_transformers import SentenceTransformer
class Embedder:
    def __init__(self,model_name):
        print(f"[EMBED] Loading {model_name}"); self.model=SentenceTransformer(model_name)
    def encode(self,texts,batch_size=32):
        x=self.model.encode(texts,batch_size=batch_size,show_progress_bar=True,normalize_embeddings=True,convert_to_numpy=True)
        return x.astype("float32")
