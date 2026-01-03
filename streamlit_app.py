"""Streamlit UI for the RAG app.

Features:
- Upload and ingest PDFs (runs the same pipeline as the backend ChatService).
- Ask questions against the vector store and view retrieved context.
- Lightweight logging configured via backend settings/env.
"""

from pathlib import Path
from typing import List, Tuple

import streamlit as st
from langchain.schema import Document

from backend.core.dependency import (
    get_docstore,
    get_model,
    get_settings,
    get_vector_store,
)
from backend.servies.chat_service import ChatService
from backend.servies.file_service import PDFFileService
from backend.servies.model_service import ModelService
from backend.utils.logging import configure_logging


@st.cache_resource(show_spinner=False)
def bootstrap_services() -> Tuple[ChatService, Path]:
    """Initialize settings, logging, and core services once per session."""
    cfg = get_settings()
    cfg.ensure_dirs()
    configure_logging(cfg)

    vector_store = get_vector_store(cfg)
    docstore = get_docstore(cfg)
    model_service: ModelService = get_model(cfg)
    file_service = PDFFileService()

    chat_service = ChatService(
        cfg=cfg,
        vector_store=vector_store,
        file_service=file_service,
        model_service=model_service,
        docstore=docstore,
    )
    return chat_service, cfg.upload_dir


def _persist_upload(upload_dir: Path, uploaded_file) -> Path:
    """Save uploaded file to disk and return the path."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / uploaded_file.name
    with target.open("wb") as f:
        f.write(uploaded_file.getbuffer())
    return target


def _metric(label: str, value: str) -> None:
    """Render a compact metric widget."""
    st.metric(label, value)


def main() -> None:
    st.set_page_config(page_title="RAG QA Studio", layout="wide")
    st.title("RAG QA Studio")
    st.caption("Upload a PDF, ingest it, and ask questions over the indexed content.")

    chat_service, upload_dir = bootstrap_services()

    # Sidebar: ingestion controls
    st.sidebar.header("Ingest")
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    ingest_btn = st.sidebar.button("Ingest file", type="primary", disabled=uploaded_file is None)

    ingest_status = st.sidebar.empty()

    if ingest_btn and uploaded_file:
        try:
            ingest_status.info("Saving and ingesting...")
            with st.spinner("Saving and ingesting..."):
                pdf_path = _persist_upload(upload_dir, uploaded_file)
                result = chat_service.ingest(str(pdf_path))
            ingest_status.success(
                f"Ingested {uploaded_file.name}: pages={result.processed_pages}, chunks={result.chunks_indexed}"
            )
        except Exception as exc:  # pragma: no cover - UI flow
            ingest_status.error(f"Failed to ingest: {exc}")

    # Main: QA area
    st.subheader("Ask a question")
    col_q, col_k = st.columns([3, 1])
    with col_q:
        question = st.text_input("Question", placeholder="e.g., What is the model architecture?")
    with col_k:
        top_k = st.slider("Top-K", min_value=1, max_value=10, value=4, step=1)

    ask_btn = st.button("Ask", type="primary", disabled=not question.strip())

    if "history" not in st.session_state:
        st.session_state.history = []  # type: ignore

    if ask_btn and question.strip():
        with st.spinner("Retrieving and generating..."):
            response = chat_service.answer(question.strip(), k=top_k)
        st.session_state.history.append(
            {"question": question.strip(), "answer": response.answer, "context": response.context}
        )

    # Display chat history
    for idx, turn in enumerate(reversed(st.session_state.get("history", []))):
        st.markdown(f"**Q{len(st.session_state.history)-idx}:** {turn['question']}")
        st.markdown(turn["answer"])

        with st.expander("Show retrieved context", expanded=False):
            context: List = turn["context"]
            for i, ctx in enumerate(context, start=1):
                page = f"p.{ctx.page_number}" if ctx.page_number is not None else "n/a"
                st.markdown(f"- **Chunk {i} ({page})** — {ctx.text}")

        st.divider()

    # Footer metrics (quick debug)
    with st.sidebar:
        st.subheader("Vector store")
        try:
            ntotal = getattr(chat_service.vector_store.index, "ntotal", 0)
        except Exception:
            ntotal = "unknown"
        _metric("Indexed chunks", str(ntotal))
        _metric("Top-K default", str(chat_service.cfg.search_k))


if __name__ == "__main__":
    main()
