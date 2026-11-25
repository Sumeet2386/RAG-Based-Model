"""Modular RAG engine for Video, Audio, and PDF ingestion and chat."""

from .config import (
    AUDIO_EXTENSIONS,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_WHISPER_MODEL,
    PDF_EXTENSIONS,
    TEXT_EXTENSIONS,
    VIDEO_EXTENSIONS,
)
from .ollama_client import OllamaClient
from .vector_store import VectorStore, ChunkRecord
from .ingestion import ingest_file, compute_file_hash
from .chatbot import build_rag_messages, format_location

__all__ = [
    "AUDIO_EXTENSIONS",
    "ChunkRecord",
    "DEFAULT_CHAT_MODEL",
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_OLLAMA_HOST",
    "DEFAULT_WHISPER_MODEL",
    "OllamaClient",
    "PDF_EXTENSIONS",
    "TEXT_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "VectorStore",
    "build_rag_messages",
    "compute_file_hash",
    "format_location",
    "ingest_file",
]

