import time
import streamlit as st
from src.config import RAW_DIR,VECTOR_DIR,EMBEDDING_MODEL,LLM_MODEL
from src.pipeline import build_local_index,ask
st.set_page_config(page_title="VaultRAG",layout="wide")
st.title("VaultRAG"); st.caption("Local document question answering with Retrieval-Augmented Generation")
with st.sidebar:
    st.header("Documents")
    uploads=st.file_uploader("Upload PDF, DOCX, TXT or Markdown",type=["pdf","docx","txt","md"],accept_multiple_files=True)
    if st.button("Save uploaded documents",use_container_width=True):
        if not uploads: st.warning("Upload at least one document.")
        for f in uploads: (RAW_DIR/f.name).write_bytes(f.getbuffer()); st.success(f"Saved {f.name}")
    if st.button("Build / rebuild local index",use_container_width=True):
        with st.spinner("Indexing documents..."):
            try: build_local_index(); st.success("Index built successfully.")
            except Exception as e: st.error(str(e))
    st.divider(); st.write(f"Embeddings: `{EMBEDDING_MODEL}`"); st.write(f"LLM: `{LLM_MODEL}`")
    docs=[p.name for p in RAW_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".pdf",".docx",".txt",".md"}]
    for d in docs: st.write(f"- {d}")
if not (VECTOR_DIR/"index.faiss").exists(): st.warning("No local index found. Upload documents and build the index first.")
else:
    q=st.text_area("Ask a question about your documents",height=120); rr=st.checkbox("Use second-stage reranking",True)
    if st.button("Ask",type="primary"):
        if not q.strip(): st.warning("Enter a question.")
        else:
            with st.spinner("Retrieving and generating..."):
                t=time.perf_counter()
                try: r=ask(q.strip(),rr,True)
                except Exception as e: st.error(str(e)); st.stop()
                elapsed=(time.perf_counter()-t)*1000
            st.subheader("Answer"); st.write(r["answer"]); c1,c2,c3=st.columns(3); c1.metric("Sources",len(r["sources"])); c2.metric("Retrieval",f"{r['retrieval_ms']:.0f} ms"); c3.metric("End-to-end",f"{elapsed:.0f} ms")
            st.subheader("Sources")
            for i,s in enumerate(r["sources"],1):
                loc=s["document_name"]+(f" — page {s['page']}" if s.get("page") is not None else "")
                with st.expander(f"[SOURCE {i}] {loc} | score={s['score']:.4f}"): st.write(s["text"])
