# VaultRAG

A local Retrieval-Augmented Generation system for question answering over private documents.

## Stack
- Embeddings: BAAI/bge-small-en-v1.5
- Vector search: FAISS
- Local LLM: Qwen/Qwen3-0.6B
- UI: Streamlit
- Evaluation corpus: SQuAD validation set

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Build reproducible public index

```powershell
python main.py --mode build --eval-limit 300
```

## Evaluate

```powershell
python main.py --mode evaluate --eval-limit 300
```

## Use your own documents

Put PDF, DOCX, TXT or MD files in `data/raw/`, then:

```powershell
python main.py --mode index-local
python main.py --mode ask --question "What does the document say about ...?"
```

## Streamlit

```powershell
streamlit run app.py
```

Local document contents, embeddings, retrieval and generation are intended to stay on the local machine. No paid LLM API is required. Local execution should not be represented as enterprise-grade security without additional controls.
