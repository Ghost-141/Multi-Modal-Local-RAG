# Multi-Modal Local RAG


End-to-end RAG (Retrieval-Augmented Generation) system with FastAPI backend for PDF ingestion, chunking, FAISS vector indexing, and retrieval-augmented QA using Ollama models. Features a Streamlit UI for document upload and chat interface, plus CLI tools for headless processing.

## Features
- **PDF Processing**: Advanced PDF ingestion with text chunking, table extraction, and image handling
- **Vector Search**: FAISS-based vector indexing and similarity search
- **Multi-Modal Support**: Handles text, tables, and images from PDF documents
- **Chat Interface**: Retrieval-augmented question answering with configurable top-k results
- **Health Monitoring**: Comprehensive health checks for models and vector store
- **Persistent Storage**: JSON-based document store with vector persistence

## Setup

### Prerequisites
- Python ≥3.10
- Ollama running locally with required models

### Installation
```bash
# Install dependencies
pip install uv
# or with uv
uv sync
```

### Ollama Models
Ensure Ollama is running and pull required models:
```bash
ollama pull gemma3
ollama pull embeddinggemma:300m
```

## Configuration

### Environment Variables (.env)
```env
APP_ENV=local
DATA_DIR=./storage
EMBEDDING_MODEL=embeddinggemma:300m
CHAT_MODEL=gemma3
OLLAMA_BASE_URL=http://localhost:11434
SEARCH_K=4
LOG_LEVEL=INFO
LOG_TO_FILE=false
# LOG_FILE=./storage/logs/app.log  # Optional custom log file path
```

### Configuration Details
- **Runtime Settings**: Managed in `backend/core/config.py` with environment variable overrides
- **Storage**: Vector store, uploads, and logs default to `./storage/` directory
- **Dependency Injection**: Service wiring handled in `backend/core/dependency.py`
- **Logging**: Configurable via `LOG_LEVEL`, `LOG_TO_FILE`, and `LOG_FILE` variables

## API Documentation

### Base URL
`/api`

### Endpoints

#### Health Check
```http
GET /api/health
```
Returns system status, model readiness, and vector store statistics.

#### Document Ingestion
```http
# Upload file
POST /api/ingest
Content-Type: multipart/form-data

# Existing file path
POST /api/ingest?file_path=/path/to/document.pdf
```

#### Chat/Query
```http
POST /api/chat
Content-Type: application/json

{
  "question": "Your question here",
  "k": 4
}
```

### Example Usage
```bash
# Health check
curl http://localhost:8000/api/health

# Ingest via file path
curl -X POST "http://localhost:8000/api/ingest?file_path=/full/path/to.pdf"

# Ingest via upload
curl -X POST -F "file=@/path/to/document.pdf" http://localhost:8000/api/ingest

# Chat query
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the main topic?","k":4}'
```

## Usage

### Start API Server
```bash
uvicorn backend.main:app --reload
```

### Launch Streamlit UI
```bash
streamlit run streamlit_app.py
```

### CLI Document Processing
```bash
python -m backend.utils.process_pdf /path/to/document.pdf
```

## Project Structure

```
├── backend/
│   ├── app.py                 # FastAPI application entrypoint
│   ├── api/
│   │   ├── chat.py            # Ingest and chat endpoints
│   │   └── health.py          # Health check endpoint
│   ├── core/
│   │   ├── config.py          # Environment-driven configuration
│   │   └── dependency.py      # Dependency injection setup
│   ├── models/
│   │   └── schemas.py         # Pydantic request/response models
│   ├── servies/               # Business logic services
│   │   ├── interface/         # Service interfaces
│   │   │   ├── chat_interface.py
│   │   │   ├── file_interface.py
│   │   │   └── model_interface.py
│   │   ├── chat_service.py    # RAG orchestration service
│   │   ├── file_service.py    # PDF processing service
│   │   ├── model_service.py   # Ollama model wrappers
│   │   └── types.py           # Type definitions
│   ├── system_prompts/
│   │   ├── notebook_prompts.py
│   │   └── prompt_v1.py       # System prompts for chat
│   └── utils/
│       ├── json_docstore.py   # Document persistence
│       ├── logging.py         # Logging configuration
│       ├── parent_store.py    # Parent document storage
│       └── process_pdf.py     # CLI PDF processing
├── streamlit_app.py           # Streamlit web interface
├── prompts.py                 # Additional prompt utilities
├── pyproject.toml             # Project dependencies
└── .env                       # Environment configuration
```

## Tech Stack

- **Backend**: FastAPI, Uvicorn
- **ML/AI**: LangChain, Ollama, Transformers, PyTorch
- **Vector Store**: FAISS
- **Document Processing**: Unstructured, PDF2Image
- **UI**: Streamlit
- **Utilities**: Pydantic, Python-dotenv, TQDM

## License

See [LICENSE](LICENSE) file for details.
