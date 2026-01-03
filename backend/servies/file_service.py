import base64
import logging
import time
from pathlib import Path
from typing import Iterable, List, Optional

from langchain.schema import Document
from unstructured.partition.pdf import partition_pdf

from backend.servies.interface.file_interface import FileInterface
from backend.servies.types import ModalChunks


class PDFFileService(FileInterface):
    """Parse PDFs into modal chunks (text, tables, images) using notebook-style extraction."""

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        chunking_strategy: str = "by_title",
        max_characters: int = 2000,
        combine_text_under_n_chars: int = 1500,
        new_after_n_chars: int = 5000,
    ) -> None:
        if combine_text_under_n_chars > max_characters:
            raise ValueError(
                f"combine_text_under_n_chars ({combine_text_under_n_chars}) "
                f"cannot exceed max_characters ({max_characters})"
            )
        self.chunking_strategy = chunking_strategy
        self.max_characters = max_characters
        self.combine_text_under_n_chars = combine_text_under_n_chars
        self.new_after_n_chars = new_after_n_chars

    def _custom_chunk(self, elements: Iterable) -> Iterable:
        """Placeholder for future custom chunking (tables/images, etc.)."""
        return elements

    @staticmethod
    def _meta_obj(obj: object) -> object:
        """Return raw metadata object or dict if present."""
        return getattr(obj, "metadata", None)

    @classmethod
    def _meta_value(cls, obj: object, key: str) -> Optional[object]:
        """Fetch a metadata attribute from metadata obj/dict/to_dict."""
        meta = cls._meta_obj(obj)
        if meta is None:
            return None
        if hasattr(meta, key):
            return getattr(meta, key)
        if isinstance(meta, dict):
            return meta.get(key)
        to_dict = getattr(meta, "to_dict", None)
        if callable(to_dict):
            try:
                return to_dict().get(key)
            except Exception:
                return None
        return None

    @classmethod
    def _get_page_number(cls, chunk: object) -> Optional[int]:
        return cls._meta_value(chunk, "page_number")

    @classmethod
    def _get_orig_elements(cls, chunk: object) -> list:
        meta = cls._meta_obj(chunk)
        if meta is None:
            return []
        orig = getattr(meta, "orig_elements", None)
        if orig:
            return orig
        if isinstance(meta, dict):
            return meta.get("orig_elements", []) or []
        to_dict = getattr(meta, "to_dict", None)
        if callable(to_dict):
            try:
                data = to_dict()
                return data.get("orig_elements", []) or []
            except Exception:
                return []
        return []

    @classmethod
    def _iter_chunk_elements(cls, chunk: object) -> list:
        """Yield orig_elements when present, plus the chunk itself as a fallback."""
        orig = cls._get_orig_elements(chunk)
        return (orig or []) + [chunk]

    @classmethod
    def _page_number_from(cls, *objs: object) -> Optional[int]:
        for obj in objs:
            page = cls._get_page_number(obj)
            if page is not None:
                return page
        return None

    def _extract_tables(self, elements: Iterable, source: str) -> List[Document]:
        """Match notebook get_table: iterate orig_elements and grab table HTML."""
        tables: List[Document] = []
        for chunk in elements:
            orig_elements = self._get_orig_elements(chunk)
            for el in orig_elements:
                if "Table" in str(type(el)):
                    html = self._meta_value(el, "text_as_html") or getattr(el, "text", "")
                    if not html:
                        continue
                    tables.append(
                        Document(
                            page_content=str(html),
                            metadata={
                                "source": source,
                                "page_number": self._page_number_from(el, chunk),
                                "type": "table",
                            },
                        )
                    )
        return tables

    def _extract_texts(self, elements: Iterable, source: str) -> List[Document]:
        """Match notebook save_texts: keep CompositeElement chunks as text docs."""
        texts: List[Document] = []
        for chunk in elements:
            if "CompositeElement" in str(type(chunk)):
                text = getattr(chunk, "text", "") or ""
                cleaned = text.strip()
                if cleaned:
                    texts.append(
                        Document(
                            page_content=cleaned,
                            metadata={
                                "source": source,
                                "page_number": self._page_number_from(chunk),
                                "type": "text",
                            },
                        )
                    )
        return texts

    def _extract_images(self, elements: Iterable, source: str) -> List[Document]:
        """Match notebook get_image_base64: pull base64 from orig_elements or disk."""
        images: List[Document] = []
        for chunk in elements:
            orig_elements = self._get_orig_elements(chunk)
            for el in orig_elements:
                if "Image" not in str(type(el)):
                    continue
                img_b64 = self._meta_value(el, "image_base64")
                if not img_b64:
                    img_path = self._meta_value(el, "image_path")
                    if img_path:
                        try:
                            data = Path(img_path).read_bytes()
                            img_b64 = base64.b64encode(data).decode("utf-8")
                        except Exception:
                            img_b64 = None
                if img_b64:
                    images.append(
                        Document(
                            page_content=img_b64,
                            metadata={
                                "source": source,
                                "page_number": self._page_number_from(el, chunk),
                                "type": "image",
                            },
                        )
                    )
        return images

    def load(self, file_path: str) -> ModalChunks:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found at {file_path}")

        start = time.perf_counter()
        self.logger.info(
            "Parsing PDF %s with chunking=%s max_chars=%d combine_under=%d new_after=%d",
            path.name,
            self.chunking_strategy,
            self.max_characters,
            self.combine_text_under_n_chars,
            self.new_after_n_chars,
        )

        elements = partition_pdf(
            filename=str(path),
            strategy="hi_res",
            infer_table_structure=True,
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            extract_image_block_output_dir="figures",
            chunking_strategy=self.chunking_strategy,
            max_characters=self.max_characters,
            combine_text_under_n_chars=self.combine_text_under_n_chars,
            new_after_n_chars=self.new_after_n_chars,
        )
        self.logger.info(
            "Partitioned %s into %d elements in %.2fs",
            path.name,
            len(elements),
            time.perf_counter() - start,
        )

        chunked = self._custom_chunk(elements)

        texts = self._extract_texts(chunked, str(path.name))
        tables = self._extract_tables(chunked, str(path.name))
        images = self._extract_images(chunked, str(path.name))
        self.logger.info(
            "Extracted %d text chunks, %d tables, %d images from %s",
            len(texts),
            len(tables),
            len(images),
            path.name,
        )

        return ModalChunks(texts=texts, tables=tables, images=images)
