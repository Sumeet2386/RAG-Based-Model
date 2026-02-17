"""
Streamlit UI patched to be robust:
- Loads embeddings.joblib and embeddings_meta.json
- Uses OLLAMA_HOST /api/embed for query embedding
- Checks dimension mismatch and shows helpful error with remediation
"""
import streamlit as st
from pathlib import Path
import os, joblib, json
import numpy as np
import requests

PROJECT_ROOT = Path(os.environ.get("RAG_PROJECT_ROOT") or Path(__file__).resolve().parent)
st.sidebar.markdown(f"**Project root:** `{PROJECT_ROOT}`")

EMBED_JOBLIB = PROJECT_ROOT / "embeddings.joblib"
EMBED_META = PROJECT_ROOT / "embeddings_meta.json"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip('/')
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")

st.sidebar.text_input("Ollama host URL", value=OLLAMA_HOST, key="ollama_host")
st.sidebar.text_input("Embedding model", value=EMBED_MODEL, key="embed_model")

st.title("Ask a question")

question = st.text_area("Enter your question", height=150)
k = st.slider("Top K results", 1, 10, 3)

def load_embeddings():
    if not EMBED_JOBLIB.exists():
        return None, None
    data = joblib.load(EMBED_JOBLIB)
    # data expected as list of dicts with key 'embedding'
    if isinstance(data, dict):
        # maybe a dataframe-like object serialized
        recs = data.get("records", [])
    else:
        recs = data
    embeddings = []
    texts = []
    for r in recs:
        emb = r.get("embedding") if isinstance(r, dict) else None
        embeddings.append(emb)
        texts.append(r.get("text","") if isinstance(r, dict) else "")
    arr = np.array(embeddings, dtype=float)
    meta = {}
    if EMBED_META.exists():
        with EMBED_META.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
    return recs, arr, meta

def get_query_embedding(text):
    url = f"{OLLAMA_HOST}/api/embed"
    r = requests.post(url, json={"model": EMBED_MODEL, "input": [text]}, timeout=30)
    r.raise_for_status()
    data = r.json()
    embs = data.get("embeddings") or data.get("embedding")
    if not embs:
        raise RuntimeError("No embeddings returned: " + str(data))
    emb = embs[0] if isinstance(embs, list) else embs
    return np.array(emb, dtype=float)

if st.button("Search"):
    if not question or question.strip()=="":
        st.warning("Please enter a question.")
    else:
        loaded = load_embeddings()
        if loaded[0] is None:
            st.error("No embeddings found. Run read_chunks.py to create embeddings.joblib and embeddings_meta.json in project root.")
        else:
            recs, mat, meta = loaded
            st.success(f"Loaded {len(recs)} chunks from project")
            try:
                q = get_query_embedding(question)
                st.write("Query embedding length:", q.shape[0])
                stored_dim = None
                if meta and "embedding_dim" in meta:
                    stored_dim = int(meta["embedding_dim"]) if meta["embedding_dim"] else None
                # check dims
                if mat.ndim != 2:
                    st.error(f"Stored embeddings array is unexpected shape: {mat.shape}")
                elif stored_dim and stored_dim != q.shape[0]:
                    st.error(f"Dimension mismatch: saved embeddings dim {stored_dim} != query dim {q.shape[0]}. Recreate embeddings with the same model.")
                elif mat.shape[1] != q.shape[0]:
                    st.error(f"Shapes misaligned: stored dim {mat.shape[1]} != query dim {q.shape[0]}.")
                else:
                    # cosine similarity
                    norms = np.linalg.norm(mat, axis=1)
                    qn = np.linalg.norm(q)
                    sims = (mat @ q) / (norms * qn + 1e-12)
                    idx = np.argsort(-sims)[:k]
                    for i in idx:
                        st.markdown(f"**Score {sims[i]:.4f}**")
                        st.write(recs[i].get("text",""))
            except Exception as e:
                st.error(f"Search failed: {e}")
