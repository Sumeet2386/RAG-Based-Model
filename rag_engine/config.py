"""Centralized configuration defaults for the Video & PDF RAG Chatbot."""

from pathlib import Path
import os

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("RAG_PROJECT_ROOT") or PACKAGE_DIR.parent)

DATA_DIR = Path("/tmp/rag_data") if os.environ.get("VERCEL") else (PROJECT_ROOT / "data")
UPLOAD_DIR = DATA_DIR / "uploads"
STORE_PATH = DATA_DIR / "vector_store.joblib"
STORE_META_PATH = DATA_DIR / "vector_store_meta.json"

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
DEFAULT_CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3.2")
DEFAULT_WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")

EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH", "64"))

# Chunking defaults
PDF_CHUNK_CHARS = 1000
PDF_CHUNK_OVERLAP = 150
VIDEO_CHUNK_MAX_WORDS = 220
VIDEO_CHUNK_MAX_SECONDS = 45.0

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
PDF_EXTENSIONS = {".pdf"}
TEXT_EXTENSIONS = {".txt", ".md", ".json"}

