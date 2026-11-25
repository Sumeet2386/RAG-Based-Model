# Video & PDF Local RAG Chatbot

A real-time, full-stack **Retrieval-Augmented Generation (RAG)** application that lets users upload **Videos**, **Audio files**, **PDFs**, or **Text/Transcripts** and chat with them using a **React + FastAPI** interface powered by **Ollama** and **Whisper**.

---

## Key Features

- **Real-Time Multi-Format Ingestion**
  - **Videos & Audio** (`.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`, `.mp3`, `.wav`, `.m4a`): Extracts 16kHz mono PCM audio via `ffmpeg` (`imageio-ffmpeg`), transcribes locally with Whisper (`mlx-whisper` on Apple Silicon, `faster-whisper`, `openai-whisper`, or `whisper-cli`), and merges segments into coherent timestamped chunks.
  - **PDFs** (`.pdf`): Extracts text page-by-page using `pypdf` with sentence-aware overlapping chunks and exact page metadata.
  - **Text & Transcripts** (`.txt`, `.md`, `.json`): Direct ingestion of plain text or pre-generated Whisper JSON transcripts.
- **High-Efficiency Vector Search**
  - Stores pre-normalized `float32` unit vectors in memory (`data/vector_store.joblib`) for single-operation cosine similarity (`matrix @ q`) and `np.argpartition` top-K retrieval.
  - Supports SHA-256 file deduplication, per-document filtering, and legacy `embeddings.joblib` import.
- **Resilient Embedding Pipeline**
  - Uses pooled HTTP calls to Ollama (`/api/embed`) for models like `bge-m3`.
  - Automatically falls back to built-in deterministic `local-fast` feature-hashing embeddings if no Ollama embedding model is pulled.
- **Modern React + FastAPI Streaming UI**
  - Drag-and-drop multi-file upload, live knowledge base management, model selectors, and **Server-Sent Events (SSE)** token streaming with expandable source citations (`MM:SS–MM:SS` for videos, `Page X` for PDFs).

---

## Architecture

```text
├── frontend/                 # React + Vite UI (drag-and-drop upload, SSE chat, citations)
│   ├── src/App.jsx
│   ├── src/index.css
│   └── vite.config.js        # Proxies /api -> http://127.0.0.1:8000 in dev
├── backend/
│   └── main.py               # FastAPI REST + SSE streaming server
├── api/
│   └── index.py              # Vercel Serverless entrypoint
├── rag_engine/               # Core modular RAG engine
│   ├── config.py             # Paths, model defaults, chunking parameters
│   ├── ingestion.py          # PDF, Video/Audio, and Text/JSON extractors
│   ├── ollama_client.py      # Pooled Ollama client + local-fast embedding fallback
│   ├── vector_store.py       # Pre-normalized float32 vector store
│   └── chatbot.py            # Grounded multi-turn prompt builder & citation formatter
├── tests/
│   └── test_rag_engine.py    # Unit & integration tests
├── vercel.json               # Vercel build & /api rewrite config
├── requirements.txt          # Python dependencies
└── app.py                    # Optional standalone Streamlit UI
```

---

## Local Setup & Run

### 1. Prerequisites
- **Python 3.10+**
- **Node.js 18+** & `npm`
- **Ollama** installed and running locally

### 2. Install Python & Frontend Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd frontend
npm install
cd ..
```

### 3. Start Ollama & Pull Models
```bash
ollama serve
ollama pull llama3.2
# Recommended for dense semantic embeddings:
ollama pull bge-m3
```

### 4. Run Backend & Frontend
In terminal 1 (FastAPI backend):
```bash
.venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In terminal 2 (React Vite dev server):
```bash
cd frontend
npm run dev
```

Open **http://localhost:5173/** in your browser.

---

## Deploying to Vercel

This repository includes [vercel.json](vercel.json) and [api/index.py](api/index.py) configured to build `frontend/dist` and route `/api/*` to the FastAPI backend:

```bash
npx vercel --prod
```

> **Note on Cloud Deployments:** Vercel Serverless Functions cannot reach a local `http://localhost:11434` Ollama instance and use ephemeral `/tmp` storage. For cloud usage, set **Ollama Host** in the UI settings to a publicly reachable Ollama/tunnel URL, or use **Retrieval-only mode** with built-in `local-fast` embeddings.

---

## Running Tests

```bash
.venv/bin/python tests/test_rag_engine.py
```
