from typing import List, Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    file_path: Optional[str] = Field(
        None,
        description="Path to an already uploaded PDF; optional when using file upload.",
    )


class IngestResponse(BaseModel):
    processed_pages: int
    chunks_indexed: int
    vector_store_path: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question to run retrieval against.")
    k: int = Field(4, description="Number of chunks to fetch from the vector store.")


class ContextChunk(BaseModel):
    text: str
    page_number: Optional[int] = None
    source: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    context: List[ContextChunk]


class HealthResponse(BaseModel):
    status: str
    config: dict
    embedding_ready: bool
    chat_ready: bool
