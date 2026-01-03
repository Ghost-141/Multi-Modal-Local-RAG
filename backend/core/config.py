import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _load_env() -> None:
    """Load environment variables from a .env at repo root if present."""
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()  # fallback to default search


_load_env()


class Settings:
    """Runtime configuration for the backend service."""

    def __init__(self) -> None:
        self.env: str = os.environ.get("APP_ENV", "local")
        self.data_dir: Path = Path(
            os.environ.get("DATA_DIR", Path.cwd() / "storage")
        ).resolve()
        self.vector_store_path: Path = self.data_dir / "vector_store"
        self.upload_dir: Path = self.data_dir / "uploads"
        self.docstore_path: Path = self.data_dir / "docstore.json"
        self.log_dir: Path = self.data_dir / "logs"
        self.log_level: str = os.environ.get("LOG_LEVEL", "INFO").upper()
        self.log_to_file: bool = (
            os.environ.get("LOG_TO_FILE", "false").lower() == "true"
        )
        self.log_file: Path = Path(
            os.environ.get("LOG_FILE", self.log_dir / "app.log")
        ).resolve()
        self.embedding_model: str = os.environ.get(
            "EMBEDDING_MODEL", "embeddinggemma:300m"
        )
        self.chat_model: str = os.environ.get("CHAT_MODEL", "gemma3")
        self.ollama_base_url: str = os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.search_k: int = int(os.environ.get("SEARCH_K", "4"))

    def ensure_dirs(self) -> None:
        """Create required directories if they do not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        if self.log_to_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def model_dump(self) -> dict[str, Any]:
        return {
            "env": self.env,
            "data_dir": str(self.data_dir),
            "vector_store_path": str(self.vector_store_path),
            "upload_dir": str(self.upload_dir),
            "docstore_path": str(self.docstore_path),
            "log_dir": str(self.log_dir),
            "log_level": self.log_level,
            "log_to_file": self.log_to_file,
            "log_file": str(self.log_file),
            "embedding_model": self.embedding_model,
            "chat_model": self.chat_model,
            "ollama_base_url": self.ollama_base_url,
            "search_k": self.search_k,
        }


settings = Settings()
