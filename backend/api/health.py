from fastapi import APIRouter, Depends

from backend.core.config import Settings
from backend.core.dependency import get_settings, get_vector_store, get_model
from backend.models.schemas import HealthResponse

router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
def health(cfg: Settings = Depends(get_settings)):
    model_service = get_model(cfg)
    store = get_vector_store(cfg)
    ntotal = getattr(store, "index", None)
    count = ntotal.ntotal if ntotal is not None else 0
    embedding_ready = False
    chat_ready = False
    chat_model_name = None
    try:
        chat_model = model_service.get_chat_model()
        chat_model_name = getattr(
            chat_model, "model", None
        ) or cfg.embedding_model.replace("embedding", "chat")
        chat_ready = True
    except Exception:
        chat_ready = False
    try:
        embedder = model_service.get_embedder()
        embedder.embed_query("health check")
        embedding_ready = True
    except Exception:
        embedding_ready = False
    return HealthResponse(
        status="ok",
        config={**cfg.model_dump(), "vectors": count, "chat_model": chat_model_name},
        embedding_ready=embedding_ready,
        chat_ready=chat_ready,
    )
