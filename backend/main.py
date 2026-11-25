"""FastAPI backend for the Real-Time Video & PDF RAG Chatbot."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_engine import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_WHISPER_MODEL,
    OllamaClient,
    VectorStore,
    build_rag_messages,
    compute_file_hash,
    format_location,
    ingest_file,
)
from rag_engine.config import PROJECT_ROOT
from rag_engine.ollama_client import LOCAL_EMBED_MODEL

app = FastAPI(title="Video & PDF RAG Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = VectorStore()
store.load()
ollama = OllamaClient(DEFAULT_OLLAMA_HOST)


class ChatRequest(BaseModel):
    query: str
    chat_history: list[dict[str, str]] = []
    doc_ids: list[str] | None = None
    top_k: int = 5
    temperature: float = 0.2
    embed_model: str = DEFAULT_EMBED_MODEL
    chat_model: str = DEFAULT_CHAT_MODEL
    ollama_host: str = DEFAULT_OLLAMA_HOST
    retrieval_only: bool = False


@app.get("/api/health")
def get_health(ollama_host: str = DEFAULT_OLLAMA_HOST) -> dict[str, Any]:
    ollama.set_host(ollama_host)
    healthy, models, err = ollama.check_health()
    embed_models = [
        m for m in models if any(k in m.lower() for k in ("embed", "bge", "minilm", "e5"))
    ]
    embed_options = embed_models + [LOCAL_EMBED_MODEL] + [m for m in models if m not in embed_models]
    recommended_embed = embed_models[0] if embed_models else LOCAL_EMBED_MODEL
    has_sample = (PROJECT_ROOT / "embeddings.joblib").exists() or (
        PROJECT_ROOT / "RAG" / "embeddings.joblib"
    ).exists()
    return {
        "ollama_healthy": healthy,
        "ollama_error": err,
        "models": models,
        "embed_models": embed_options,
        "defaults": {
            "ollama_host": DEFAULT_OLLAMA_HOST,
            "embed_model": recommended_embed,
            "chat_model": models[0] if models else DEFAULT_CHAT_MODEL,
            "whisper_model": DEFAULT_WHISPER_MODEL,
        },
        "store": {
            "num_documents": len(store.documents),
            "num_chunks": len(store.records),
            "embed_model": store.embed_model,
            "has_sample": has_sample,
        },
    }


@app.get("/api/documents")
def list_documents() -> dict[str, Any]:
    return {"documents": list(store.documents.values())}


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    embed_model: str = Form(DEFAULT_EMBED_MODEL),
    whisper_model: str = Form(DEFAULT_WHISPER_MODEL),
    language: str = Form("auto"),
    ollama_host: str = Form(DEFAULT_OLLAMA_HOST),
) -> dict[str, Any]:
    ollama.set_host(ollama_host)
    healthy, _, err = ollama.check_health()
    if not healthy:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama is not reachable at {ollama_host}: {err}",
        )

    filename = file.filename or "uploaded_file"
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    doc_id = compute_file_hash(file_bytes)
    if store.has_document(doc_id):
        return {
            "status": "already_indexed",
            "document": store.documents[doc_id],
        }

    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        source_type, chunks, extra_meta = ingest_file(
            tmp_path,
            whisper_model=whisper_model,
            language=language,
        )
        texts = [c["text"] for c in chunks]
        embs = ollama.embed_texts(texts, model=embed_model)
        store.add_document(
            doc_id=doc_id,
            source_name=filename,
            source_type=source_type,
            chunks=chunks,
            embeddings=embs,
            embed_model=embed_model,
            extra_meta=extra_meta,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "status": "indexed",
        "document": store.documents[doc_id],
    }


@app.post("/api/documents/import-sample")
def import_sample() -> dict[str, Any]:
    candidates = [
        PROJECT_ROOT / "embeddings.joblib",
        PROJECT_ROOT / "RAG" / "embeddings.joblib",
    ]
    for candidate in candidates:
        if candidate.exists():
            legacy = VectorStore(store_path=candidate)
            if legacy.load() and legacy.records:
                store.records = legacy.records
                store.matrix = legacy.matrix
                store.documents = legacy.documents
                store.embed_dim = legacy.embed_dim
                store.embed_model = legacy.embed_model or DEFAULT_EMBED_MODEL
                store.save()
                return {"status": "imported", "documents": list(store.documents.values())}
    raise HTTPException(status_code=404, detail="No sample embeddings.joblib found.")


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str) -> dict[str, Any]:
    if not store.has_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    store.remove_document(doc_id)
    return {"status": "deleted", "documents": list(store.documents.values())}


@app.delete("/api/documents")
def clear_documents() -> dict[str, Any]:
    store.clear()
    return {"status": "cleared", "documents": []}


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/chat")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    if not store.records:
        raise HTTPException(status_code=400, detail="No documents indexed yet.")

    ollama.set_host(req.ollama_host)
    doc_filter = set(req.doc_ids) if req.doc_ids is not None else None

    def event_generator():
        try:
            q_emb = ollama.embed_query(req.query, model=req.embed_model)
            results = store.search(q_emb, top_k=req.top_k, doc_ids=doc_filter)
            sources = [
                {
                    "source_name": rec.get("source_name", "Unknown"),
                    "source_type": rec.get("source_type", "text"),
                    "location": format_location(rec),
                    "score": round(score, 4),
                    "text": rec.get("text", ""),
                }
                for rec, score in results
            ]
            yield _sse("sources", sources)

            if not results:
                yield _sse("token", "No matching content found in the selected documents.")
                yield _sse("done", {})
                return

            if req.retrieval_only:
                lines = ["**Top matching excerpts:**\n"]
                for s in sources:
                    lines.append(
                        f"- **{s['source_name']}** (`{s['location']}`, score `{s['score']:.3f}`): {s['text']}"
                    )
                yield _sse("token", "\n\n".join(lines))
                yield _sse("done", {})
                return

            messages = build_rag_messages(
                query=req.query,
                retrieved=results,
                chat_history=req.chat_history,
            )
            for token in ollama.stream_chat(
                messages=messages,
                model=req.chat_model,
                temperature=req.temperature,
            ):
                yield _sse("token", token)
            yield _sse("done", {})
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Serve built React frontend if present
DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="frontend")

