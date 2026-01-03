import json
from pathlib import Path
from typing import Dict, List, Optional


class ParentStore:
    """Lightweight JSON-backed parent store for mapping doc_id -> parent payload."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._store: Dict[str, dict] = {}
        if self.path.exists():
            try:
                self._store = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._store = {}

    def set_many(self, pairs: List[tuple[str, dict]]) -> None:
        for key, value in pairs:
            self._store[key] = value
        self._persist()

    def get_many(self, keys: List[str]) -> List[Optional[dict]]:
        return [self._store.get(k) for k in keys]

    def _persist(self) -> None:
        self.path.write_text(
            json.dumps(self._store, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
