from functools import lru_cache
from typing import Optional

from langchain_ollama import ChatOllama, OllamaEmbeddings

from backend.core.config import Settings
from backend.servies.interface.model_interface import ModelInterface


class ModelService(ModelInterface):
    """Centralized model factory for embeddings and chat LLMs."""

    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self._embedder: Optional[OllamaEmbeddings] = None
        self._chat: Optional[ChatOllama] = None

    def get_embedder(self) -> OllamaEmbeddings:
        if self._embedder is None:
            self._embedder = OllamaEmbeddings(
                model=self.cfg.embedding_model,
                base_url=self.cfg.ollama_base_url,
            )
        return self._embedder

    def get_chat_model(self) -> ChatOllama:
        if self._chat is None:
            # Derive chat model name if using an embedding model naming convention.
            chat_model = self.cfg.chat_model or self.cfg.embedding_model.replace("embedding", "chat")
            self._chat = ChatOllama(
                model=chat_model,
                base_url=self.cfg.ollama_base_url,
                temperature=0,
            )
        return self._chat

    def generate(self, prompt: str) -> str:
        model = self.get_chat_model()
        return model.invoke(prompt)


@lru_cache
def get_model_service(cfg: Settings) -> ModelService:
    return ModelService(cfg)
