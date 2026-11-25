"""High-efficiency vector store with pre-normalized float32 embeddings and metadata filtering."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .config import STORE_META_PATH, STORE_PATH


@dataclass
class ChunkRecord:
    chunk_id: int
    doc_id: str
    source_name: str
    source_type: str  # "video", "audio", "pdf", "text"
    text: str
    page: int | None = None
    start: float | None = None
    end: float | None = None

    def format_location(self) -> str:
        if self.source_type in {"video", "audio"} and self.start is not None:
            return f"{_fmt_time(self.start)} - {_fmt_time(self.end or self.start)}"
        if self.source_type == "pdf" and self.page is not None:
            return f"Page {self.page}"
        return "Document"


def _fmt_time(seconds: float) -> str:
    total_sec = max(0, int(round(seconds)))
    hrs, rem = divmod(total_sec, 3600)
    mins, secs = divmod(rem, 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (mat / norms).astype(np.float32, copy=False)


def _l2_normalize_vec(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec.astype(np.float32, copy=False)
    return (vec / norm).astype(np.float32, copy=False)


class VectorStore:
    """In-memory + joblib-persisted vector store using unit-normalized float32 matrix."""

    def __init__(
        self,
        store_path: Path = STORE_PATH,
        meta_path: Path = STORE_META_PATH,
    ) -> None:
        self.store_path = Path(store_path)
        self.meta_path = Path(meta_path)
        self.records: list[dict[str, Any]] = []
        self.matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.embed_model: str | None = None
        self.embed_dim: int | None = None
        self.documents: dict[str, dict[str, Any]] = {}

    def load(self) -> bool:
        """Load persisted store if present. Returns True if loaded."""
        if not self.store_path.exists():
            return False
        payload = joblib.load(self.store_path)
        if isinstance(payload, dict) and "records" in payload and "matrix" in payload:
            self.records = payload["records"]
            self.matrix = np.asarray(payload["matrix"], dtype=np.float32)
            self.documents = payload.get("documents", {})
        elif isinstance(payload, list):
            # Legacy list[dict] format
            self._load_legacy_records(payload)
        elif hasattr(payload, "to_dict"):
            # Legacy pandas.DataFrame format
            self._load_legacy_records(payload.to_dict(orient="records"))
        else:
            return False

        if self.meta_path.exists():
            with self.meta_path.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
            self.embed_model = meta.get("model")
            self.embed_dim = meta.get("embedding_dim")
        elif self.matrix.ndim == 2 and self.matrix.shape[0] > 0:
            self.embed_dim = int(self.matrix.shape[1])

        if not self.documents and self.records:
            self._rebuild_documents_index()
        return True

    def _load_legacy_records(self, raw_records: list[dict[str, Any]]) -> None:
        embs = []
        cleaned = []
        for idx, rec in enumerate(raw_records):
            emb = rec.get("embedding")
            if emb is None:
                continue
            embs.append(emb)
            title = rec.get("title") or "Legacy Transcript"
            cleaned.append(
                asdict(
                    ChunkRecord(
                        chunk_id=rec.get("chunk_id", idx),
                        doc_id="legacy",
                        source_name=str(title),
                        source_type="video" if "start" in rec else "text",
                        text=str(rec.get("text", "")),
                        start=rec.get("start"),
                        end=rec.get("end"),
                        page=rec.get("page"),
                    )
                )
            )
        self.records = cleaned
        self.matrix = _l2_normalize_rows(np.asarray(embs, dtype=np.float32)) if embs else np.empty((0, 0), dtype=np.float32)
        self._rebuild_documents_index()

    def _rebuild_documents_index(self) -> None:
        docs: dict[str, dict[str, Any]] = {}
        for r in self.records:
            doc_id = r["doc_id"]
            if doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "source_name": r["source_name"],
                    "source_type": r["source_type"],
                    "chunks": 0,
                }
            docs[doc_id]["chunks"] += 1
        self.documents = docs

    def save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "records": self.records,
                "matrix": self.matrix,
                "documents": self.documents,
            },
            self.store_path,
            compress=3,
        )
        meta = {
            "embedding_dim": int(self.embed_dim) if self.embed_dim else None,
            "model": self.embed_model,
            "num_chunks": len(self.records),
            "num_documents": len(self.documents),
        }
        with self.meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    def has_document(self, doc_id: str) -> bool:
        return doc_id in self.documents

    def add_document(
        self,
        doc_id: str,
        source_name: str,
        source_type: str,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
        embed_model: str,
        extra_meta: dict[str, Any] | None = None,
    ) -> int:
        """Add a document's chunks and embeddings. Returns number of chunks added."""
        if len(chunks) == 0 or embeddings.size == 0:
            return 0
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Chunks and embeddings count mismatch")

        normed = _l2_normalize_rows(embeddings)
        dim = int(normed.shape[1])

        if self.embed_dim is not None and self.embed_dim != dim and len(self.records) > 0:
            raise ValueError(
                f"Embedding dimension mismatch: store has {self.embed_dim}, new embeddings have {dim}. "
                f"Use the same embedding model ({self.embed_model}) or clear the store."
            )

        if self.has_document(doc_id):
            self.remove_document(doc_id, save=False)

        start_id = len(self.records)
        new_records: list[dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            rec = ChunkRecord(
                chunk_id=start_id + i,
                doc_id=doc_id,
                source_name=source_name,
                source_type=source_type,
                text=chunk["text"],
                page=chunk.get("page"),
                start=chunk.get("start"),
                end=chunk.get("end"),
            )
            new_records.append(asdict(rec))

        self.records.extend(new_records)
        if self.matrix.size == 0:
            self.matrix = normed
        else:
            self.matrix = np.vstack([self.matrix, normed])

        self.embed_dim = dim
        self.embed_model = embed_model
        doc_info = {
            "doc_id": doc_id,
            "source_name": source_name,
            "source_type": source_type,
            "chunks": len(new_records),
        }
        if extra_meta:
            doc_info.update(extra_meta)
        self.documents[doc_id] = doc_info
        self.save()
        return len(new_records)

    def remove_document(self, doc_id: str, save: bool = True) -> None:
        if doc_id not in self.documents:
            return
        keep_indices = [i for i, r in enumerate(self.records) if r["doc_id"] != doc_id]
        self.records = [self.records[i] for i in keep_indices]
        for idx, rec in enumerate(self.records):
            rec["chunk_id"] = idx
        if keep_indices and self.matrix.size > 0:
            self.matrix = self.matrix[keep_indices]
        else:
            self.matrix = np.empty((0, 0), dtype=np.float32)
            self.embed_dim = None
        self.documents.pop(doc_id, None)
        if save:
            self.save()

    def clear(self) -> None:
        self.records = []
        self.matrix = np.empty((0, 0), dtype=np.float32)
        self.documents = {}
        self.embed_dim = None
        self.embed_model = None
        if self.store_path.exists():
            self.store_path.unlink()
        if self.meta_path.exists():
            self.meta_path.unlink()

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        doc_ids: set[str] | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        """Return top_k (record_dict, cosine_score) pairs."""
        if not self.records or self.matrix.size == 0:
            return []
        q = _l2_normalize_vec(query_embedding)
        if q.shape[0] != self.matrix.shape[1]:
            raise ValueError(
                f"Query embedding dim {q.shape[0]} != store dim {self.matrix.shape[1]}"
            )

        if doc_ids is not None:
            mask_idx = [i for i, r in enumerate(self.records) if r["doc_id"] in doc_ids]
            if not mask_idx:
                return []
            sub_mat = self.matrix[mask_idx]
            scores = sub_mat @ q
            k = min(top_k, len(mask_idx))
            top_local = np.argpartition(-scores, k - 1)[:k]
            top_local = top_local[np.argsort(-scores[top_local])]
            return [(self.records[mask_idx[i]], float(scores[i])) for i in top_local]

        scores = self.matrix @ q
        k = min(top_k, len(self.records))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self.records[i], float(scores[i])) for i in top_idx]

