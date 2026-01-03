from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, UploadFile

from backend.core.config import Settings
from backend.core.dependency import (
    get_settings,
    get_vector_store,
    get_model,
    get_docstore,
)
from backend.models.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
)
from backend.servies.chat_service import ChatService
from backend.servies.file_service import PDFFileService
from backend.servies.model_service import ModelService

router = APIRouter(prefix="/api")

def get_service(
    cfg: Settings = Depends(get_settings),
) -> ChatService:
    vector_store = get_vector_store(cfg)
    docstore = get_docstore(cfg)
    file_service = PDFFileService()
    model_service: ModelService = get_model(cfg)
    return ChatService(
        cfg=cfg,
        vector_store=vector_store,
        file_service=file_service,
        model_service=model_service,
        docstore=docstore,
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    payload: IngestRequest = Depends(),
    file: Optional[UploadFile] = File(None),
    svc: ChatService = Depends(get_service),
):
    target_path: Optional[Path] = None
    cfg = svc.cfg

    if file:
        target_path = cfg.upload_dir / file.filename
        with target_path.open("wb") as f:
            f.write(await file.read())
    elif payload.file_path:
        target_path = Path(payload.file_path)

    if target_path is None:
        raise HTTPException(
            status_code=400, detail="Provide a PDF via file upload or file_path."
        )

    try:
        return svc.ingest(str(target_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, svc: ChatService = Depends(get_service)):
    try:
        return svc.answer(req.question, req.k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
