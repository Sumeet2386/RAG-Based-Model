"""Ingestion pipeline for PDF, Video, Audio, and Text/Transcript files."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from pypdf import PdfReader

from .config import (
    AUDIO_EXTENSIONS,
    PDF_CHUNK_CHARS,
    PDF_CHUNK_OVERLAP,
    PDF_EXTENSIONS,
    TEXT_EXTENSIONS,
    VIDEO_CHUNK_MAX_SECONDS,
    VIDEO_CHUNK_MAX_WORDS,
    VIDEO_EXTENSIONS,
)


def compute_file_hash(data: bytes) -> str:
    """Return short SHA-256 hex digest to deduplicate uploaded files."""
    return hashlib.sha256(data).hexdigest()[:16]


def _get_ffmpeg_exe() -> str:
    """Resolve system ffmpeg or bundled imageio-ffmpeg binary."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg is required for video/audio processing. "
            "Install imageio-ffmpeg (`pip install imageio-ffmpeg`) or system ffmpeg."
        ) from exc


def extract_audio_wav(media_path: Path, out_wav: Path) -> Path:
    """Extract 16kHz mono PCM WAV from video or audio using ffmpeg."""
    ffmpeg = _get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(out_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {proc.stderr[-500:]}")
    return out_wav


def _split_text_sliding(
    text: str,
    chunk_chars: int = PDF_CHUNK_CHARS,
    overlap: int = PDF_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks, preferring sentence/whitespace boundaries."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_chars:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    n = len(cleaned)
    while start < n:
        end = min(start + chunk_chars, n)
        if end < n:
            # Prefer breaking on sentence end or space near the end of window
            window = cleaned[start:end]
            split_at = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
            if split_at > chunk_chars // 2:
                end = start + split_at + 1
            else:
                space_at = window.rfind(" ")
                if space_at > chunk_chars // 2:
                    end = start + space_at
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def extract_pdf_chunks(
    pdf_path: Path,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract text from a PDF page-by-page into overlapping chunks with page metadata."""
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    chunks: list[dict[str, Any]] = []

    for page_idx, page in enumerate(reader.pages, start=1):
        if progress_cb:
            progress_cb(f"Extracting PDF page {page_idx}/{total_pages}...")
        raw_text = page.extract_text() or ""
        for piece in _split_text_sliding(raw_text):
            chunks.append({"text": piece, "page": page_idx})

    if not chunks:
        raise RuntimeError(
            "No selectable text could be extracted from this PDF. "
            "If it is a scanned image PDF, run OCR before uploading."
        )
    return chunks, {"pages": total_pages}


def merge_transcript_segments(
    segments: list[dict[str, Any]],
    max_words: int = VIDEO_CHUNK_MAX_WORDS,
    max_seconds: float = VIDEO_CHUNK_MAX_SECONDS,
) -> list[dict[str, Any]]:
    """Merge short Whisper segments into coherent RAG chunks while keeping timestamps."""
    merged: list[dict[str, Any]] = []
    buf_texts: list[str] = []
    buf_words = 0
    buf_start: float | None = None
    buf_end: float | None = None

    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        words = len(text.split())

        if buf_start is None:
            buf_start = seg_start
        buf_end = seg_end
        buf_texts.append(text)
        buf_words += words

        duration = buf_end - buf_start
        if buf_words >= max_words or duration >= max_seconds:
            merged.append(
                {
                    "text": " ".join(buf_texts),
                    "start": round(buf_start, 2),
                    "end": round(buf_end, 2),
                }
            )
            # Keep last segment as 1-segment overlap for context continuity
            buf_texts = [text]
            buf_words = words
            buf_start = seg_start
            buf_end = seg_end

    if buf_texts and buf_start is not None:
        last_text = " ".join(buf_texts)
        if not merged or merged[-1]["text"] != last_text:
            merged.append(
                {
                    "text": last_text,
                    "start": round(buf_start, 2),
                    "end": round(buf_end or buf_start, 2),
                }
            )
    return merged


def _load_wav_float32(wav_path: Path):
    """Load a 16kHz mono PCM16 WAV file as a float32 numpy array in [-1, 1]."""
    import wave
    import numpy as np

    with wave.open(str(wav_path), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def _transcribe_wav(
    wav_path: Path,
    whisper_model: str = "base",
    language: str | None = None,
) -> list[dict[str, Any]]:
    """Transcribe a 16kHz WAV file using the fastest available local Whisper backend."""
    # 1. faster-whisper (CTranslate2, fastest on CPU)
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(whisper_model, device="cpu", compute_type="int8")
        lang_arg = language if language and language != "auto" else None
        segments_iter, _info = model.transcribe(str(wav_path), language=lang_arg, vad_filter=True)
        return [
            {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
            for s in segments_iter
            if s.text.strip()
        ]
    except ImportError:
        pass

    # 2. mlx-whisper (Apple Silicon Metal acceleration)
    try:
        import mlx_whisper

        lang_arg = language if language and language != "auto" else None
        audio_arr = _load_wav_float32(wav_path)
        result = mlx_whisper.transcribe(
            audio_arr,
            path_or_hf_repo=f"mlx-community/whisper-{whisper_model}-mlx",
            language=lang_arg,
        )
        return [
            {"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()}
            for s in result.get("segments", [])
            if s.get("text", "").strip()
        ]
    except ImportError:
        pass

    # 3. openai-whisper (PyTorch)
    try:
        import whisper

        model = whisper.load_model(whisper_model)
        kwargs: dict[str, Any] = {"word_timestamps": False}
        if language and language != "auto":
            kwargs["language"] = language
        audio_arr = _load_wav_float32(wav_path)
        result = model.transcribe(audio_arr, **kwargs)
        return [
            {"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()}
            for s in result.get("segments", [])
            if s.get("text", "").strip()
        ]
    except ImportError:
        pass

    # 4. whisper-cli (whisper.cpp via Homebrew)
    whisper_cli = shutil.which("whisper-cli") or shutil.which("whisper-cpp")
    if whisper_cli:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_prefix = Path(tmpdir) / "transcript"
            cmd = [whisper_cli, "-f", str(wav_path), "-oj", "-of", str(out_prefix)]
            if language and language != "auto":
                cmd.extend(["-l", language])
            proc = subprocess.run(cmd, capture_output=True, text=True)
            json_file = Path(f"{out_prefix}.json")
            if proc.returncode == 0 and json_file.exists():
                data = json.loads(json_file.read_text(encoding="utf-8"))
                return [
                    {
                        "start": item.get("offsets", {}).get("from", 0) / 1000.0,
                        "end": item.get("offsets", {}).get("to", 0) / 1000.0,
                        "text": item.get("text", "").strip(),
                    }
                    for item in data.get("transcription", [])
                    if item.get("text", "").strip()
                ]

    raise RuntimeError(
        "No local Whisper transcription backend is installed in this Python environment. "
        "Install `faster-whisper` or `openai-whisper` (recommended on Python 3.11–3.13), "
        "`mlx-whisper` (Apple Silicon), or `brew install whisper-cpp`. "
        "You can still upload PDFs, TXT/MD files, or Whisper JSON transcripts right now."
    )


def extract_media_chunks(
    media_path: Path,
    whisper_model: str = "base",
    language: str | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract audio from video/audio and transcribe into timestamped chunks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "audio.wav"
        if progress_cb:
            progress_cb("Extracting 16kHz audio track with ffmpeg...")
        extract_audio_wav(media_path, wav_path)

        if progress_cb:
            progress_cb(f"Transcribing audio with Whisper ({whisper_model})...")
        segments = _transcribe_wav(wav_path, whisper_model=whisper_model, language=language)

    if not segments:
        raise RuntimeError("No speech was detected in the uploaded media file.")

    chunks = merge_transcript_segments(segments)
    duration = segments[-1]["end"]
    return chunks, {"duration_seconds": round(duration, 1), "raw_segments": len(segments)}


def extract_text_or_json_chunks(file_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load plain text, markdown, or pre-generated Whisper JSON transcripts."""
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(file_path.read_text(encoding="utf-8", errors="replace"))
        raw_chunks = data.get("chunks") or data.get("segments") or []
        if raw_chunks and isinstance(raw_chunks, list):
            if "start" in raw_chunks[0]:
                chunks = merge_transcript_segments(raw_chunks)
                return chunks, {"raw_segments": len(raw_chunks)}
            return [
                {"text": str(c.get("text", "")).strip(), "page": c.get("page")}
                for c in raw_chunks
                if str(c.get("text", "")).strip()
            ], {}
        full_text = str(data.get("text", ""))
        return [{"text": p} for p in _split_text_sliding(full_text)], {}

    raw = file_path.read_text(encoding="utf-8", errors="replace")
    chunks = [{"text": p} for p in _split_text_sliding(raw)]
    if not chunks:
        raise RuntimeError("Uploaded text file is empty.")
    return chunks, {}


def ingest_file(
    file_path: Path,
    whisper_model: str = "base",
    language: str | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Dispatch ingestion by file extension. Returns (source_type, chunks, extra_meta)."""
    ext = file_path.suffix.lower()
    if ext in PDF_EXTENSIONS:
        chunks, meta = extract_pdf_chunks(file_path, progress_cb=progress_cb)
        return "pdf", chunks, meta
    if ext in VIDEO_EXTENSIONS:
        chunks, meta = extract_media_chunks(
            file_path, whisper_model=whisper_model, language=language, progress_cb=progress_cb
        )
        return "video", chunks, meta
    if ext in AUDIO_EXTENSIONS:
        chunks, meta = extract_media_chunks(
            file_path, whisper_model=whisper_model, language=language, progress_cb=progress_cb
        )
        return "audio", chunks, meta
    if ext in TEXT_EXTENSIONS:
        chunks, meta = extract_text_or_json_chunks(file_path)
        return "text", chunks, meta
    raise ValueError(f"Unsupported file type: {ext}")

