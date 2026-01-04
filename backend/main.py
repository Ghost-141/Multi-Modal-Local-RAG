"""FastAPI application entrypoint that registers API routers."""

from fastapi import FastAPI

from backend.api.chat import router as chat_router
from backend.api.health import router as health_router


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Backend", version="0.1.0")
    app.include_router(chat_router)
    app.include_router(health_router)
    return app


app = create_app()
