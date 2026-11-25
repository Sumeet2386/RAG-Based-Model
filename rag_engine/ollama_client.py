"""Pooled Ollama HTTP client for embeddings, chat streaming, and model discovery."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Callable, Generator, Iterable

import numpy as np
import requests


LOCAL_EMBED_MODEL = "local-fast"


def local_embed_texts(texts: list[str], dim: int = 512) -> np.ndarray:
    """Fast deterministic feature-hashing embeddings (word + char n-grams) when no Ollama embed model is pulled."""
    if not texts:
        return np.empty((0, dim), dtype=np.float32)
    mat = np.zeros((len(texts), dim), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            continue
        features: list[str] = list(tokens)
        # Add word bigrams for phrase matching
        features.extend(f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1))
        for feat in features:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            val = int.from_bytes(digest, "little")
            col = val % dim
            sign = 1.0 if ((val >> 63) & 1) == 0 else -1.0
            mat[row_idx, col] += sign
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (mat / norms).astype(np.float32, copy=False)


class OllamaClient:
    """Thread-safe / session-pooled client for a local or remote Ollama server."""

    def __init__(self, host: str = "http://localhost:11434") -> None:
        self.host = host.rstrip("/")
        self._session = requests.Session()

    def set_host(self, host: str) -> None:
        self.host = host.rstrip("/")

    def check_health(self, timeout: float = 3.0) -> tuple[bool, list[str], str | None]:
        """Return (is_healthy, installed_model_names, error_message)."""
        try:
            resp = self._session.get(f"{self.host}/api/tags", timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
            return True, models, None
        except requests.RequestException as exc:
            return False, [], str(exc)

    def embed_texts(
        self,
        texts: list[str],
        model: str = "bge-m3",
        batch_size: int = 64,
        timeout: float = 120.0,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Embed a list of texts in batches and return a float32 numpy array (N, D)."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        if model == LOCAL_EMBED_MODEL:
            arr = local_embed_texts(texts)
            if progress_cb:
                progress_cb(len(texts), len(texts))
            return arr

        all_embeddings: list[list[float]] = []
        total = len(texts)
        url = f"{self.host}/api/embed"

        for start in range(0, total, batch_size):
            batch = texts[start : start + batch_size]
            try:
                resp = self._session.post(
                    url,
                    json={"model": model, "input": batch},
                    timeout=timeout,
                )
                resp.raise_for_status()
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code in {404, 501}:
                    arr = local_embed_texts(texts)
                    if progress_cb:
                        progress_cb(total, total)
                    return arr
                raise
            data = resp.json()
            embs = data.get("embeddings") or data.get("embedding")
            if not embs:
                raise RuntimeError(f"Ollama returned no embeddings: {data}")
            if isinstance(embs[0], (int, float)):
                embs = [embs]
            if len(embs) != len(batch):
                raise RuntimeError(
                    f"Embedding count mismatch: expected {len(batch)}, got {len(embs)}"
                )
            all_embeddings.extend(embs)
            if progress_cb:
                progress_cb(min(start + len(batch), total), total)

        return np.asarray(all_embeddings, dtype=np.float32)

    def embed_query(self, text: str, model: str = "bge-m3", timeout: float = 30.0) -> np.ndarray:
        """Embed a single query string and return a 1D float32 numpy array."""
        arr = self.embed_texts([text], model=model, batch_size=1, timeout=timeout)
        return arr[0]

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: str = "llama3.2",
        temperature: float = 0.2,
        timeout: float = 120.0,
    ) -> Generator[str, None, None]:
        """Stream assistant tokens from Ollama /api/chat."""
        url = f"{self.host}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }
        with self._session.post(url, json=payload, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                chunk = json.loads(raw_line)
                if "error" in chunk:
                    raise RuntimeError(chunk["error"])
                msg = chunk.get("message") or {}
                token = msg.get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break

