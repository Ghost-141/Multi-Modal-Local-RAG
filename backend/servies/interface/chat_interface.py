from typing import Protocol

from langchain_core.documents import Document

from backend.models.schemas import ChatResponse


class ChatInterface(Protocol):
    def ingest(self, file_path: str) -> int:
        """Process and index a file, returning the number of chunks."""

    def answer(self, question: str, k: int = 4) -> ChatResponse:
        """Run retrieval over the persisted vector store."""
