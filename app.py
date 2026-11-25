"""Real-Time Video & PDF RAG Chatbot powered by Streamlit, Whisper, and Ollama."""

from __future__ import annotations

from pathlib import Path
import tempfile

import streamlit as st

from rag_engine import (
    AUDIO_EXTENSIONS,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_WHISPER_MODEL,
    OllamaClient,
    PDF_EXTENSIONS,
    TEXT_EXTENSIONS,
    VIDEO_EXTENSIONS,
    VectorStore,
    build_rag_messages,
    compute_file_hash,
    format_location,
    ingest_file,
)
from rag_engine.config import PROJECT_ROOT, STORE_PATH

st.set_page_config(
    page_title="Video & PDF RAG Chatbot",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

SUPPORTED_EXTS = sorted(
    ext.lstrip(".")
    for ext in (VIDEO_EXTENSIONS | AUDIO_EXTENSIONS | PDF_EXTENSIONS | TEXT_EXTENSIONS)
)


@st.cache_resource(show_spinner=False)
def get_vector_store() -> VectorStore:
    store = VectorStore()
    store.load()
    return store


@st.cache_resource(show_spinner=False)
def get_ollama_client() -> OllamaClient:
    return OllamaClient(DEFAULT_OLLAMA_HOST)


def _auto_import_legacy(store: VectorStore) -> bool:
    """Import legacy embeddings.joblib if present and new store is empty."""
    legacy_candidates = [
        PROJECT_ROOT / "embeddings.joblib",
        PROJECT_ROOT / "RAG" / "embeddings.joblib",
    ]
    for candidate in legacy_candidates:
        if candidate.exists():
            legacy_store = VectorStore(store_path=candidate)
            if legacy_store.load() and legacy_store.records:
                store.records = legacy_store.records
                store.matrix = legacy_store.matrix
                store.documents = legacy_store.documents
                store.embed_dim = legacy_store.embed_dim
                store.embed_model = legacy_store.embed_model or DEFAULT_EMBED_MODEL
                store.save()
                return True
    return False


def main() -> None:
    store = get_vector_store()
    client = get_ollama_client()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Sidebar: Settings, Upload & Knowledge Base ───────────────────────────
    with st.sidebar:
        st.title("⚙️ Settings & Files")

        ollama_host = st.text_input("Ollama Host", value=DEFAULT_OLLAMA_HOST)
        client.set_host(ollama_host)

        healthy, installed_models, health_err = client.check_health()
        if healthy:
            st.success(f"Ollama connected ({len(installed_models)} models)")
        else:
            st.error("Ollama not reachable. Start it with `ollama serve`.")
            if health_err:
                st.caption(health_err)

        embed_options = installed_models if installed_models else [DEFAULT_EMBED_MODEL]
        default_embed_idx = (
            embed_options.index(DEFAULT_EMBED_MODEL)
            if DEFAULT_EMBED_MODEL in embed_options
            else 0
        )
        embed_model = st.selectbox(
            "Embedding Model",
            options=embed_options,
            index=default_embed_idx,
            help="Must match the model used to index your documents.",
        )

        chat_options = installed_models if installed_models else [DEFAULT_CHAT_MODEL]
        default_chat_idx = (
            chat_options.index(DEFAULT_CHAT_MODEL)
            if DEFAULT_CHAT_MODEL in chat_options
            else 0
        )
        chat_model = st.selectbox("Chat Model", options=chat_options, index=default_chat_idx)

        with st.expander("Advanced Options", expanded=False):
            whisper_model = st.selectbox(
                "Whisper Model (Video/Audio)",
                options=["tiny", "base", "small", "medium", "large-v3"],
                index=["tiny", "base", "small", "medium", "large-v3"].index(DEFAULT_WHISPER_MODEL)
                if DEFAULT_WHISPER_MODEL in ["tiny", "base", "small", "medium", "large-v3"]
                else 1,
            )
            language = st.text_input(
                "Spoken Language (optional)",
                value="auto",
                help="Use 'auto' for auto-detection, or e.g. 'en', 'hi'.",
            )
            top_k = st.slider("Top-K Context Chunks", min_value=1, max_value=12, value=5)
            temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
            retrieval_only = st.checkbox(
                "Retrieval-only mode (no LLM)",
                value=False,
                help="Return matching chunks directly without calling a chat LLM.",
            )

        st.divider()
        st.subheader("📤 Upload Video or PDF")
        uploaded_files = st.file_uploader(
            "Drop files here to index in real time",
            type=SUPPORTED_EXTS,
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("🚀 Process & Index Files", type="primary", use_container_width=True):
            if not healthy:
                st.error("Cannot generate embeddings: Ollama server is not running.")
            else:
                for up_file in uploaded_files:
                    file_bytes = up_file.getvalue()
                    doc_id = compute_file_hash(file_bytes)
                    if store.has_document(doc_id):
                        st.info(f"Already indexed: `{up_file.name}`")
                        continue

                    with st.status(f"Processing `{up_file.name}`...", expanded=True) as status:
                        try:
                            suffix = Path(up_file.name).suffix
                            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                tmp.write(file_bytes)
                                tmp_path = Path(tmp.name)

                            try:
                                source_type, chunks, extra_meta = ingest_file(
                                    tmp_path,
                                    whisper_model=whisper_model,
                                    language=language,
                                    progress_cb=lambda msg: status.write(msg),
                                )
                            finally:
                                tmp_path.unlink(missing_ok=True)

                            status.write(f"Generating embeddings for {len(chunks)} chunks...")
                            texts = [c["text"] for c in chunks]
                            prog = st.progress(0.0)
                            embs = client.embed_texts(
                                texts,
                                model=embed_model,
                                progress_cb=lambda done, tot: prog.progress(done / max(tot, 1)),
                            )
                            added = store.add_document(
                                doc_id=doc_id,
                                source_name=up_file.name,
                                source_type=source_type,
                                chunks=chunks,
                                embeddings=embs,
                                embed_model=embed_model,
                                extra_meta=extra_meta,
                            )
                            status.update(
                                label=f"Indexed `{up_file.name}` ({added} chunks)",
                                state="complete",
                                expanded=False,
                            )
                        except Exception as exc:
                            status.update(label=f"Failed `{up_file.name}`", state="error")
                            st.error(str(exc))

        st.divider()
        st.subheader(f"📚 Knowledge Base ({len(store.documents)} files)")

        if not store.documents and (
            (PROJECT_ROOT / "embeddings.joblib").exists()
            or (PROJECT_ROOT / "RAG" / "embeddings.joblib").exists()
        ):
            if st.button("Load Sample Pre-Indexed Lecture", use_container_width=True):
                if _auto_import_legacy(store):
                    st.rerun()

        selected_doc_ids: set[str] | None = None
        if store.documents:
            doc_choices = {
                doc_id: f"{info['source_name']} ({info['source_type'].upper()}, {info['chunks']} chunks)"
                for doc_id, info in store.documents.items()
            }
            active_ids = st.multiselect(
                "Filter sources for chat",
                options=list(doc_choices.keys()),
                default=list(doc_choices.keys()),
                format_func=lambda did: doc_choices[did],
            )
            selected_doc_ids = set(active_ids)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear Chat", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()
            with col2:
                if st.button("Reset Index", use_container_width=True):
                    store.clear()
                    st.session_state.messages = []
                    st.rerun()
        else:
            st.caption("Upload a Video, Audio, or PDF in the uploader above to get started.")

    # ── Main Chat Area ───────────────────────────────────────────────────────
    st.title("🎬📄 Video & PDF RAG Chatbot")
    st.caption(
        "Ask questions across your uploaded videos, audio lectures, and PDFs. "
        "Answers are grounded in your files with exact timestamps and page citations."
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"View {len(msg['sources'])} retrieved sources"):
                    for s in msg["sources"]:
                        st.markdown(
                            f"**{s['source_name']}** · `{s['location']}` · Score `{s['score']:.3f}`"
                        )
                        st.caption(s["text"])

    prompt = st.chat_input("Ask a question about your uploaded videos or PDFs...")
    if prompt:
        if not store.records:
            st.warning("Please upload and index at least one Video or PDF in the sidebar first.")
            st.stop()
        if not healthy:
            st.error("Ollama is not reachable. Start Ollama (`ollama serve`) and try again.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                q_emb = client.embed_query(prompt, model=embed_model)
                results = store.search(q_emb, top_k=top_k, doc_ids=selected_doc_ids)

                sources_payload = [
                    {
                        "source_name": rec.get("source_name", "Unknown"),
                        "location": format_location(rec),
                        "score": score,
                        "text": rec.get("text", ""),
                    }
                    for rec, score in results
                ]

                if not results:
                    answer = "No matching content found in the selected documents."
                    st.markdown(answer)
                elif retrieval_only:
                    lines = ["**Top matching excerpts:**\n"]
                    for s in sources_payload:
                        lines.append(f"- **{s['source_name']}** (`{s['location']}`, score `{s['score']:.3f}`): {s['text']}")
                    answer = "\n\n".join(lines)
                    st.markdown(answer)
                else:
                    rag_messages = build_rag_messages(
                        query=prompt,
                        retrieved=results,
                        chat_history=st.session_state.messages[:-1],
                    )
                    answer = st.write_stream(
                        client.stream_chat(
                            messages=rag_messages,
                            model=chat_model,
                            temperature=temperature,
                        )
                    )

                if sources_payload:
                    with st.expander(f"View {len(sources_payload)} retrieved sources"):
                        for s in sources_payload:
                            st.markdown(
                                f"**{s['source_name']}** · `{s['location']}` · Score `{s['score']:.3f}`"
                            )
                            st.caption(s["text"])

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources_payload}
                )
            except Exception as exc:
                st.error(f"Error answering question: {exc}")


if __name__ == "__main__":
    main()

