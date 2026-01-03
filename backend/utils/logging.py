import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from backend.core.config import Settings
from backend.core.dependency import get_settings


def configure_logging(cfg: Optional[Settings] = None) -> None:
    """Configure application logging based on settings.

    - Respects LOG_LEVEL
    - If LOG_TO_FILE is true, writes to LOG_FILE (rotating handler).
    - Always logs to console.
    """
    cfg = cfg or get_settings()
    handlers = []

    console = logging.StreamHandler()
    console.setLevel(cfg.log_level)
    handlers.append(console)

    if cfg.log_to_file:
        cfg.log_dir.mkdir(parents=True, exist_ok=True)
        log_path: Path = cfg.log_file
        file_handler = RotatingFileHandler(
            log_path, maxBytes=5 * 1024 * 1024, backupCount=3
        )
        file_handler.setLevel(cfg.log_level)
        handlers.append(file_handler)

    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=handlers,
        force=True,
    )
