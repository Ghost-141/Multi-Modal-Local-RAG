Backend layout (key files)
├── backend/
│   ├── api/
│   │   ├── chat.py          # ingest/chat routes
│   │   └── health.py        # health endpoint
│   ├── app.py               # FastAPI entrypoint registering routers
│   ├── core/
│   │   ├── config.py
│   │   └── dependency.py
│   ├── models/
│   │   └── schemas.py
│   ├── servies/
│   │   ├── interface/
│   │   │   ├── chat_interface.py
│   │   │   ├── file_interface.py
│   │   │   └── model_interface.py
│   │   ├── chat_service.py
│   │   ├── file_service.py
│   │   └── model_service.py
│   ├── system_prompts/
│   │   └── prompt_v1.py
│   └── utils/
│       ├── logging.py
│       └── process_pdf.py   # CLI helper to ingest a PDF
