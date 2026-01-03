Short Project Overview
- End-to-end RAG stack: FastAPI backend for PDF ingestion, chunking, FAISS indexing, and retrieval-augmented QA with Ollama models; Streamlit UI for uploading, ingesting, and chatting over indexed content; CLI helper for headless ingestion.

Features
- PDF ingestion with chunking, table/image handling, and vector indexing (FAISS).
- Chat endpoint that retrieves top-k chunks and answers via Ollama LLM.
- Health check endpoint for readiness info.
- Streamlit UI for upload + QA and a CLI helper to ingest without running the server.

Setup
1) Python ≥3.10, install deps: `pip install -e .` (or `uv pip install -e .`).
2) Ensure Ollama is running with required models pulled.
3) Optional `.env` (examples): `CHAT_MODEL=gemma3`, `EMBEDDING_MODEL=embeddinggemma:300m`, `OLLAMA_BASE_URL=http://localhost:11434`, `LOG_LEVEL=INFO`.

Configurations
- Runtime settings live in `backend/core/config.py` (env-driven).
- Vector/doc stores default under `storage/`; reset behavior controlled in `backend/core/dependency.py`.
- Logging controlled by `LOG_LEVEL`, `LOG_TO_FILE`, `LOG_FILE`.

API Documentation
- Base URL: `/api`
- Health: `GET /api/health`
- Ingest (existing file): `POST /api/ingest?file_path=/full/path/to.pdf`
- Ingest (upload): `POST /api/ingest` with form file `file=@/path/to.pdf`
- Chat: `POST /api/chat` with JSON `{"question":"...","k":4}`
Example curls:
- `curl http://localhost:8000/api/health`
- `curl -X POST "http://localhost:8000/api/ingest?file_path=/full/path/to.pdf"`
- `curl -X POST -F "file=@/full/path/to.pdf" http://localhost:8000/api/ingest`
- `curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d "{\"question\":\"What is the model architecture?\",\"k\":4}"`

Usage
- API: `uvicorn backend.app:app --reload`
- Streamlit UI: `streamlit run streamlit_app.py`
- CLI ingest: `python -m backend.utils.process_pdf path/to/file.pdf`

```bash
Project structure (key files)
├── backend/
│   ├── app.py                 # FastAPI entrypoint registering routers
│   ├── api/
│   │   ├── chat.py            # ingest/chat routes
│   │   └── health.py          # health endpoint
│   ├── core/
│   │   ├── config.py          # env/settings
│   │   └── dependency.py      # DI helpers, vector/doc store wiring
│   ├── models/
│   │   └── schemas.py         # Pydantic schemas
│   ├── servies/               # (typo kept for compatibility)
│   │   ├── interface/
│   │   │   ├── chat_interface.py
│   │   │   ├── file_interface.py
│   │   │   └── model_interface.py
│   │   ├── chat_service.py    # retrieval + QA orchestration
│   │   ├── file_service.py    # PDF parsing/chunking
│   │   └── model_service.py   # Ollama chat/embedding wrappers
│   ├── system_prompts/
│   │   └── prompt_v1.py
│   └── utils/
│       ├── json_docstore.py   # persisted parent docs
│       ├── logging.py         # logging setup
│       └── process_pdf.py     # CLI helper to ingest a PDF
├── streamlit_app.py           # Streamlit UI for upload/chat
├── notebook.py                # exploratory notebook logic
├── main.py                    # placeholder
├── pyproject.toml             # dependencies
```
Tech stack
- FastAPI, Uvicorn
- LangChain, FAISS
- Ollama (chat + embedding models)
- Unstructured (PDF parsing)
- Streamlit (UI)
- Pydantic, Python-dotenv, logging helpers
