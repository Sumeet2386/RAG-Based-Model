"""Pooled Ollama HTTP client for embeddings, chat streaming, and model discovery."""

from __future__ import annotations

import json
from typing import Callable, Generator, Iterable

import numpy as np
import requests


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

        all_embeddings: list[list[float]] = []
        total = len(texts)
        url = f"{self.host}/api/embed"

        for start in range(0, total, batch_size):
            batch = texts[start : start + batch_size]
            resp = self._session.post(
                url,
                json={"model": model, "input": batch},
                timeout=timeout,
            )
            resp.raise_for_status()
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

