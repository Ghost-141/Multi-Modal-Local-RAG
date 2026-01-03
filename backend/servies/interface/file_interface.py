from typing import Protocol

from backend.servies.types import ModalChunks


class FileInterface(Protocol):
    def load(self, file_path: str) -> ModalChunks:
        """Parse a file into modal chunks (text, tables, images)."""
