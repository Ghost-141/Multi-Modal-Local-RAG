import logging
import time
import uuid
from typing import List, Optional, Any, Dict

from langchain.schema import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.vectorstores import VectorStore
from langchain.retrievers.multi_vector import MultiVectorRetriever

from backend.core.config import Settings
from backend.core.dependency import persist_vector_store
from backend.models.schemas import ChatRequest, ChatResponse, ContextChunk, IngestResponse
from backend.servies.interface.chat_interface import ChatInterface
from backend.servies.interface.file_interface import FileInterface
from backend.servies.model_service import ModelService
from backend.system_prompts.prompt_v1 import PROMPT, TEXT_SUMMARY_PROMPT, IMAGE_DESCRIPTION_PROMPT
from backend.utils.json_docstore import JsonDocStore


DEFAULT_PROMPT = ChatPromptTemplate.from_template(PROMPT)
SUMMARY_PROMPT = ChatPromptTemplate.from_template(TEXT_SUMMARY_PROMPT)
IMAGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            [
                {"type": "text", "text": IMAGE_DESCRIPTION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": "{image_url}"},
                },
            ],
        )
    ]
)


class ChatService(ChatInterface):
    def __init__(
        self,
        cfg: Settings,
        vector_store: VectorStore,
        file_service: FileInterface,
        model_service: ModelService,
        docstore: JsonDocStore,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.cfg = cfg
        self.vector_store = vector_store
        self.file_service = file_service
        self.model_service = model_service
        self.docstore = docstore
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vector_store,
            docstore=self.docstore,
            id_key="doc_id",
            search_kwargs={"k": self.cfg.search_k},
        )

    def ingest(self, file_path: str) -> IngestResponse:
        self.logger.info("Ingest started for %s", file_path)
        t_ingest = time.perf_counter()

        modal = self.file_service.load(file_path)
        if not (modal.texts or modal.tables or modal.images):
            self.logger.warning("No chunks extracted from %s", file_path)
            return IngestResponse(
                processed_pages=0,
                chunks_indexed=0,
                vector_store_path=str(self.cfg.vector_store_path),
            )

        self.logger.info(
            "Modal counts for %s -> text=%d tables=%d images=%d",
            file_path,
            len(modal.texts),
            len(modal.tables),
            len(modal.images),
        )

        llm = self.model_service.get_chat_model()
        parser = StrOutputParser()

        # Summaries for text/table
        t_summary = time.perf_counter()
        text_summaries = (
            SUMMARY_PROMPT | llm | parser
        ).batch([{"element": d.page_content} for d in modal.texts]) if modal.texts else []

        table_summaries = (
            SUMMARY_PROMPT | llm | parser
        ).batch([{"element": d.page_content} for d in modal.tables]) if modal.tables else []
        self.logger.info(
            "Summaries complete for %s in %.2fs (text=%d, tables=%d)",
            file_path,
            time.perf_counter() - t_summary,
            len(text_summaries),
            len(table_summaries),
        )

        # Summaries for images (using image_url data URI)
        image_summaries: List[str] = []
        if modal.images:
            t_img = time.perf_counter()
            image_inputs = [
                {"image_url": f"data:image/jpeg;base64,{d.page_content}"}
                for d in modal.images
            ]
            image_summaries = (IMAGE_PROMPT | llm | parser).batch(image_inputs)
            self.logger.info(
                "Image summaries complete for %s in %.2fs (images=%d)",
                file_path,
                time.perf_counter() - t_img,
                len(image_summaries),
            )

        # Build child docs with doc_ids
        child_docs: List[Document] = []
        parents: List[tuple[str, Document]] = []

        def add_docs(chunks: List[Document], summaries: List[str], modality: str) -> None:
            for idx, d in enumerate(chunks):
                if idx < len(summaries) and summaries[idx]:
                    summary = summaries[idx]
                elif modality == "image":
                    summary = f"Image from {d.metadata.get('source')} page {d.metadata.get('page_number')}"
                else:
                    summary = d.page_content
                doc_id = f"{modality}-{uuid.uuid4()}"
                child_docs.append(
                    Document(
                        page_content=summary,
                        metadata={
                            "doc_id": doc_id,
                            "modality": modality,
                            "source": d.metadata.get("source"),
                            "page_number": d.metadata.get("page_number"),
                        },
                    )
                )
                parents.append(
                    (
                        doc_id,
                        Document(
                            page_content=d.page_content,
                            metadata={
                                "modality": modality,
                                "source": d.metadata.get("source"),
                                "page_number": d.metadata.get("page_number"),
                            },
                        ),
                    )
                )

        add_docs(modal.texts, text_summaries, "text")
        add_docs(modal.tables, table_summaries, "table")
        add_docs(modal.images, image_summaries, "image")
        self.logger.info(
            "Prepared %d child docs for indexing (%d parents)",
            len(child_docs),
            len(parents),
        )

        ids = self.vector_store.add_documents(child_docs) if child_docs else []
        if child_docs:
            persist_vector_store(self.vector_store, self.cfg)
            self.docstore.mset(parents)
            self.logger.info(
                "Persisted vector store and docstore for %s (indexed %d docs)",
                file_path,
                len(ids),
            )

        processed_pages = len(
            {
                d.metadata.get("page_number")
                for d in (modal.texts + modal.tables + modal.images)
                if d.metadata.get("page_number") is not None
            }
        )
        self.logger.info(
            "Ingest finished for %s in %.2fs (pages=%d, chunks_indexed=%d)",
            file_path,
            time.perf_counter() - t_ingest,
            processed_pages,
            len(ids),
        )

        return IngestResponse(
            processed_pages=processed_pages,
            chunks_indexed=len(ids),
            vector_store_path=str(self.cfg.vector_store_path),
        )

    def _format_context(self, docs: List[Document]) -> List[ContextChunk]:
        return [
            ContextChunk(
                text=d.page_content,
                page_number=d.metadata.get("page_number"),
                source=d.metadata.get("source"),
            )
            for d in docs
        ]

    @staticmethod
    def _to_text(doc: Any) -> str:
        if isinstance(doc, Document):
            return doc.page_content or ""
        if hasattr(doc, "text"):
            return getattr(doc, "text") or ""
        return str(doc) if doc is not None else ""

    def _parse_docs(self, docs: List[Document]) -> Dict[str, List[str]]:
        images: List[str] = []
        texts: List[str] = []
        for d in docs:
            content = self._to_text(d).strip()
            if not content:
                continue
            # If you're not using images right now, keep everything as text
            texts.append(content)
        return {"images": images, "texts": texts}

    def _build_prompt(self, payload: Dict[str, Any]) -> List[HumanMessage]:
        ctx = payload["context"]
        question = payload["question"]
        context_text = "\n\n".join(ctx.get("texts", []))

        return [
            HumanMessage(
                content="""
You are a Q/A assistant that answers questions using ONLY the provided context.
If the context does not contain enough information, say exactly: "I don't know based on the provided context."

Rules:
- Start your answer immediately. Do NOT write any preface or lead-in (e.g., "Okay", "Let's break down", "Sure", "Here is", "Here's").
- Do not use outside knowledge or guess missing details.
- If the question is ambiguous, ask ONE short clarifying question.
- Keep the answer concise but include all meaningful details from the context relevant to the question.
- Don't answer anything outside the retrieved context. No citations or source tags.

Context:
{context_text}

Question:
{question}

Answer format:
- Use bullet points, one bullet per distinct fact from the context (still use a single bullet if only one fact).
- Do not add sources, citations, or page numbers.
""".strip().format(context_text=context_text, question=question)
            )
        ]

    def _generate_answer(self, docs: List[Document], question: str) -> str:
        """Generate an answer using a retrieval-aware chain."""
        try:
            llm: ChatOllama = self.model_service.get_chat_model()
        except Exception:
            return " ".join([d.page_content for d in docs])

        chain = (
            {
                "context": RunnableLambda(lambda _q: docs) | RunnableLambda(self._parse_docs),
                "question": RunnablePassthrough(),
            }
            | RunnablePassthrough().assign(
                response=RunnableLambda(self._build_prompt) | llm | StrOutputParser()
            )
        )

        result = chain.invoke(question)
        return result["response"]

    def answer(self, question: str, k: int = 4) -> ChatResponse:
        t_answer = time.perf_counter()
        self.retriever.search_kwargs = {"k": k or self.cfg.search_k}
        docs: List[Document] = self.retriever.invoke(question)
        self.logger.info(
            "Retrieved %d docs for question (k=%d): %.120s",
            len(docs),
            k or self.cfg.search_k,
            question,
        )
        answer = self._generate_answer(docs, question)
        self.logger.info(
            "Answer generated in %.2fs for question: %.120s",
            time.perf_counter() - t_answer,
            question,
        )
        return ChatResponse(answer=answer, context=self._format_context(docs))

    def show_context(self, question: str, k: int = 4) -> List[ContextChunk]:
        """Helper to fetch and return the context that would be used for a question."""
        self.retriever.search_kwargs = {"k": k or self.cfg.search_k}
        docs: List[Document] = self.retriever.invoke(question)
        return self._format_context(docs)
