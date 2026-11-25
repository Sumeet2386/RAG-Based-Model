# Video & PDF Local RAG Chatbot

A real-time, 100% local Retrieval-Augmented Generation (RAG) chatbot where any user can upload **Videos** (`.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`), **Audio** (`.mp3`, `.wav`, `.m4a`), **PDFs** (`.pdf`), or **Text/Transcripts** (`.txt`, `.md`, `.json`) and chat with them using **Ollama** and **Whisper**.

## Features
- **Real-Time Upload & Indexing**: Drag-and-drop videos or PDFs directly in the sidebar with live progress tracking.
- **Fast Video/Audio Pipeline**: Extracts 16kHz mono audio via `ffmpeg` (`imageio-ffmpeg`), transcribes with Whisper (`mlx-whisper` / `faster-whisper` / `openai-whisper`), and merges short segments into coherent timestamped chunks.
- **PDF Ingestion**: Extracts text page-by-page using `pypdf` with sentence-aware overlapping chunks and exact page citations.
- **Efficient Vector Search**: Stores pre-normalized `float32` unit vectors in memory (`data/vector_store.joblib`) for instant cosine similarity via a single matrix-vector dot product, with SHA-256 deduplication and per-document filtering.
- **Streaming Chatbot**: Multi-turn chat interface with streaming responses from Ollama (`/api/chat`) and expandable source citations (`MM:SS–MM:SS` for videos, `Page X` for PDFs).

## Quickstart

### 1. Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Ollama & Pull Models
```bash
ollama serve
ollama pull bge-m3
ollama pull llama3.2
```

### 3. Run the Chatbot
```bash
streamlit run app.py
```
