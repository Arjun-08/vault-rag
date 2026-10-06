from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "evaluation"
OUTPUT_DIR = ROOT / "outputs"
VECTOR_DIR = ROOT / "vector_store"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "Qwen/Qwen3-0.6B"
CHUNK_SIZE_WORDS = 650
CHUNK_OVERLAP_WORDS = 100
TOP_K = 5
CANDIDATE_K = 10
for path in [RAW_DIR, PROCESSED_DIR, EVAL_DIR, OUTPUT_DIR, VECTOR_DIR]: path.mkdir(parents=True, exist_ok=True)
