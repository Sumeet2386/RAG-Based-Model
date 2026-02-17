"""
Robust read_chunks.py
- Reads JSON chunk files from <project_root>/jsons
- Calls Ollama /api/embed in batches
- Saves embeddings to embeddings.joblib (list of dicts) and embeddings_meta.json with {"embedding_dim": N}
- Respects environment variables:
    RAG_PROJECT_ROOT - path to project root
    OLLAMA_HOST - embedding server (default http://localhost:11434)
    EMBED_MODEL - embedding model (default bge-m3)
"""
import os, json, math
from pathlib import Path
import requests
import joblib

PROJECT_ROOT = Path(os.environ.get("RAG_PROJECT_ROOT") or Path(__file__).resolve().parent)
JSON_DIR = PROJECT_ROOT / "jsons"
OUT_JOBLIB = PROJECT_ROOT / "embeddings.joblib"
OUT_META = PROJECT_ROOT / "embeddings_meta.json"

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip('/')
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")

BATCH_SIZE = int(os.environ.get("EMBED_BATCH", "64"))

def create_embeddings(batch_texts):
    if not batch_texts:
        return []
    url = f"{OLLAMA}/api/embed"
    resp = requests.post(url, json={"model": EMBED_MODEL, "input": batch_texts}, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    # Support both "embedding"/"embeddings" key variants
    if "embeddings" in data:
        embs = data["embeddings"]
    elif "embedding" in data:
        embs = data["embedding"]
    else:
        raise RuntimeError(f"No embeddings in response: {data}")
    return embs

def main():
    if not JSON_DIR.exists():
        print("JSON directory not found:", JSON_DIR)
        return
    files = sorted([p for p in JSON_DIR.glob("*.json")])
    if not files:
        print("No json files found in", JSON_DIR)
        return
    all_chunks = []
    chunk_id = 0
    embed_dim = None

    for f in files:
        print("Reading", f)
        with f.open("r", encoding="utf-8", errors="replace") as fh:
            content = json.load(fh)
        chunks = content.get("chunks", [])
        texts = [c.get("text", "") for c in chunks]
        # request in batches
        embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            print(f"Requesting embeddings for batch {i}..{i+len(batch)}")
            embs = create_embeddings(batch)
            embeddings.extend(embs)
        if len(embeddings) != len(chunks):
            raise RuntimeError(f"Embeddings count mismatch for file {f}: {len(embeddings)} != {len(chunks)}")
        for i, c in enumerate(chunks):
            c["chunk_id"] = chunk_id
            c["embedding"] = embeddings[i]
            all_chunks.append(c)
            chunk_id += 1
            if embed_dim is None and hasattr(embeddings[i], "__len__"):
                embed_dim = len(embeddings[i])
    # Save
    print(f"Saving {len(all_chunks)} chunks to {OUT_JOBLIB}")
    joblib.dump(all_chunks, OUT_JOBLIB)
    meta = {"embedding_dim": int(embed_dim) if embed_dim is not None else None, "model": EMBED_MODEL}
    print("Saving meta:", meta)
    with OUT_META.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    print("Done")

if __name__ == '__main__':
    main()
