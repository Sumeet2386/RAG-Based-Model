"""Verification tests for the Video & PDF RAG engine."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_engine import (
    VectorStore,
    build_rag_messages,
    compute_file_hash,
    format_location,
    ingest_file,
)
from rag_engine.ingestion import extract_audio_wav, merge_transcript_segments


def _make_minimal_pdf(text: str) -> bytes:
    """Build a valid single-page PDF containing `text` without external deps."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
    objs = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        (
            b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        ),
        f"4 0 obj << /Length {len(stream)} >> stream\n".encode("latin-1")
        + stream
        + b"\nendstream endobj\n",
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for obj in objs:
        offsets.append(len(out))
        out.extend(obj)
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode("latin-1"))
    for off in offsets:
        out.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
    out.extend(
        f"trailer << /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode(
            "latin-1"
        )
    )
    return bytes(out)


class TestRagEngine(unittest.TestCase):
    def test_vector_store_crud_and_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            store = VectorStore(store_path=tmp / "vs.joblib", meta_path=tmp / "meta.json")

            chunks1 = [
                {"text": "Python variables and loops", "page": 1},
                {"text": "Neural networks and backpropagation", "page": 2},
            ]
            embs1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            added = store.add_document("doc1", "notes.pdf", "pdf", chunks1, embs1, "bge-m3")
            self.assertEqual(added, 2)

            chunks2 = [{"text": "Installing ffmpeg on macOS", "start": 10.0, "end": 25.0}]
            embs2 = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
            store.add_document("doc2", "setup.mp4", "video", chunks2, embs2, "bge-m3")

            # Query closest to doc1 chunk 0
            res = store.search(np.array([0.9, 0.1, 0.0], dtype=np.float32), top_k=2)
            self.assertEqual(res[0][0]["text"], "Python variables and loops")
            self.assertGreater(res[0][1], 0.9)

            # Filter to doc2 only
            res_filtered = store.search(
                np.array([0.9, 0.1, 0.0], dtype=np.float32), top_k=2, doc_ids={"doc2"}
            )
            self.assertEqual(len(res_filtered), 1)
            self.assertEqual(res_filtered[0][0]["doc_id"], "doc2")
            self.assertEqual(format_location(res_filtered[0][0]), "00:10–00:25")

            # Reload from disk
            store2 = VectorStore(store_path=tmp / "vs.joblib", meta_path=tmp / "meta.json")
            self.assertTrue(store2.load())
            self.assertEqual(len(store2.records), 3)
            self.assertEqual(len(store2.documents), 2)

            # Remove doc1
            store2.remove_document("doc1")
            self.assertEqual(len(store2.records), 1)
            self.assertNotIn("doc1", store2.documents)

    def test_pdf_ingestion(self) -> None:
        pdf_bytes = _make_minimal_pdf("Hello RAG world from PDF page one.")
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "sample.pdf"
            pdf_path.write_bytes(pdf_bytes)
            stype, chunks, meta = ingest_file(pdf_path)
            self.assertEqual(stype, "pdf")
            self.assertEqual(meta["pages"], 1)
            self.assertEqual(len(chunks), 1)
            self.assertIn("Hello RAG world", chunks[0]["text"])
            self.assertEqual(chunks[0]["page"], 1)

    def test_video_audio_extraction_and_legacy_json(self) -> None:
        video_files = list((PROJECT_ROOT / "RAG" / "video").glob("*.mp4"))
        if video_files:
            with tempfile.TemporaryDirectory() as tmpdir:
                wav_out = Path(tmpdir) / "out.wav"
                extract_audio_wav(video_files[0], wav_out)
                self.assertTrue(wav_out.exists())
                self.assertGreater(wav_out.stat().st_size, 10000)

        json_files = list((PROJECT_ROOT / "RAG" / "jsons").glob("*.json"))
        if json_files:
            stype, chunks, meta = ingest_file(json_files[0])
            self.assertEqual(stype, "text")
            self.assertGreater(len(chunks), 0)
            self.assertIn("start", chunks[0])

    def test_legacy_embeddings_load(self) -> None:
        legacy_path = PROJECT_ROOT / "RAG" / "embeddings.joblib"
        if legacy_path.exists():
            store = VectorStore(store_path=legacy_path)
            self.assertTrue(store.load())
            self.assertGreater(len(store.records), 0)
            self.assertEqual(store.matrix.dtype, np.float32)

    def test_build_rag_messages(self) -> None:
        rec = {
            "source_name": "lecture.mp4",
            "source_type": "video",
            "start": 65.0,
            "end": 95.0,
            "text": "Functions in Python are defined with def.",
        }
        msgs = build_rag_messages(
            "How do I define a function?",
            [(rec, 0.88)],
            chat_history=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
        )
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[-1]["role"], "user")
        self.assertIn("01:05–01:35", msgs[-1]["content"])
        self.assertEqual(len(compute_file_hash(b"abc")), 16)


if __name__ == "__main__":
    unittest.main(verbosity=2)

