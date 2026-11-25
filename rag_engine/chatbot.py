"""Prompt construction and citation formatting for the RAG chatbot."""

from __future__ import annotations

from typing import Any


def format_location(record: dict[str, Any]) -> str:
    """Format human-readable source location (timestamp range or PDF page)."""
    stype = record.get("source_type", "text")
    if stype in {"video", "audio"} and record.get("start") is not None:
        start = _fmt_time(float(record["start"]))
        end = _fmt_time(float(record.get("end") or record["start"]))
        return f"{start}–{end}"
    if stype == "pdf" and record.get("page") is not None:
        return f"Page {record['page']}"
    return "Document"


def _fmt_time(seconds: float) -> str:
    total_sec = max(0, int(round(seconds)))
    hrs, rem = divmod(total_sec, 3600)
    mins, secs = divmod(rem, 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def build_rag_messages(
    query: str,
    retrieved: list[tuple[dict[str, Any], float]],
    chat_history: list[dict[str, str]] | None = None,
    max_history_turns: int = 4,
) -> list[dict[str, str]]:
    """Build system + recent history + grounded user prompt for /api/chat."""
    context_blocks: list[str] = []
    for idx, (rec, score) in enumerate(retrieved, start=1):
        loc = format_location(rec)
        src = rec.get("source_name", "Unknown")
        context_blocks.append(
            f"[{idx}] Source: {src} ({loc}, relevance={score:.2f})\n{rec.get('text', '').strip()}"
        )

    context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "No context retrieved."

    system_prompt = (
        "You are a helpful, accurate assistant that answers questions strictly based on the "
        "provided document and video/audio transcript excerpts.\n"
        "Rules:\n"
        "1. Answer clearly and concisely using ONLY the provided context.\n"
        "2. Cite your sources inline using the source name and timestamp or page number "
        "(e.g., `[video.mp4, 01:20–01:55]` or `[notes.pdf, Page 3]`).\n"
        "3. If the answer is not present in the provided context, say clearly that the uploaded "
        "files do not contain that information rather than guessing."
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if chat_history:
        recent = chat_history[-(max_history_turns * 2) :]
        for msg in recent:
            if msg.get("role") in {"user", "assistant"} and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})

    user_prompt = (
        f"Context excerpts:\n\n{context_text}\n\n"
        f"---\nQuestion: {query}"
    )
    messages.append({"role": "user", "content": user_prompt})
    return messages

