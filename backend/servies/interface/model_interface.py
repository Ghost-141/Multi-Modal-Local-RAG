from typing import Protocol

from langchain_ollama import ChatOllama, OllamaEmbeddings


class ModelInterface(Protocol):
    """Abstraction for model lifecycle and generation."""

    def get_embedder(self) -> OllamaEmbeddings:
        """Return (and cache) an embedding model."""

    def get_chat_model(self) -> ChatOllama:
        """Return (and cache) a chat model for answering questions."""

    def generate(self, prompt: str) -> str:
        """Generate a completion for the given prompt."""
