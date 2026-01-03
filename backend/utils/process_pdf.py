from pathlib import Path

from backend.core.dependency import get_settings, get_vector_store, get_model, get_docstore
from backend.servies.chat_service import ChatService
from backend.servies.file_service import PDFFileService
from backend.utils.logging import configure_logging


def process_pdf(file_path: str) -> None:
    cfg = get_settings()
    configure_logging(cfg)
    vector_store = get_vector_store(cfg)
    model_service = get_model(cfg)
    docstore = get_docstore(cfg)
    service = ChatService(
        cfg=cfg,
        vector_store=vector_store,
        file_service=PDFFileService(),
        model_service=model_service,
        docstore=docstore,
    )
    result = service.ingest(file_path)
    print(f"Ingested {file_path}: {result.chunks_indexed} chunks -> {result.vector_store_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest a PDF into the local vector store.")
    parser.add_argument("file", type=str, help="Path to the PDF file")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"File not found: {args.file}")

    process_pdf(str(path))
