from functools import lru_cache
import shutil
from pathlib import Path
from typing import Optional

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from backend.core.config import Settings, settings
from backend.servies.model_service import ModelService, get_model_service
from backend.utils.json_docstore import JsonDocStore


@lru_cache
def get_settings() -> Settings:
    settings.ensure_dirs()
    return settings


@lru_cache
def get_model(cfg: Optional[Settings] = None) -> ModelService:
    cfg = cfg or get_settings()
    return get_model_service(cfg)


def _vector_dim(model_service: ModelService) -> int:
    sample = model_service.get_embedder().embed_query("dimension check")
    return len(sample)


def _build_new_store(cfg: Settings, model_service: ModelService) -> FAISS:
    dim = _vector_dim(model_service)
    index = faiss.IndexFlatIP(dim)
    return FAISS(
        embedding_function=model_service.get_embedder(),
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
        normalize_L2=True,
    )


_STORES_RESET = False


def _reset_stores(cfg: Settings) -> None:
    """Clear persisted vector_store and docstore for a fresh run."""
    if cfg.vector_store_path.exists():
        shutil.rmtree(cfg.vector_store_path, ignore_errors=True)
    if cfg.docstore_path.exists():
        try:
            cfg.docstore_path.unlink()
        except FileNotFoundError:
            pass
    cfg.ensure_dirs()


def get_vector_store(cfg: Optional[Settings] = None) -> FAISS:
    cfg = cfg or get_settings()
    model_service = get_model(cfg)
    store_path: Path = cfg.vector_store_path

    global _STORES_RESET
    if not _STORES_RESET:
        _reset_stores(cfg)
        _STORES_RESET = True

    if store_path.exists() and any(store_path.iterdir()):
        return FAISS.load_local(
            str(store_path),
            model_service.get_embedder(),
            allow_dangerous_deserialization=True,
        )

    return _build_new_store(cfg, model_service)


def persist_vector_store(store: FAISS, cfg: Optional[Settings] = None) -> None:
    cfg = cfg or get_settings()
    cfg.ensure_dirs()
    store.save_local(str(cfg.vector_store_path))


@lru_cache
def get_docstore(cfg: Optional[Settings] = None) -> JsonDocStore:
    cfg = cfg or get_settings()
    global _STORES_RESET
    if not _STORES_RESET:
        _reset_stores(cfg)
        _STORES_RESET = True
    return JsonDocStore(Path(cfg.docstore_path))
