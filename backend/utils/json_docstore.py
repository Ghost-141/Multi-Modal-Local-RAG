import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, Iterator

from langchain.schema import Document
from langchain_core.stores import BaseStore


class JsonDocStore(BaseStore[str, Document]):
    """File-backed docstore implementing the BaseStore interface."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.store: Dict[str, Document] = {}
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self.store = {
                    k: Document(page_content=v["page_content"], metadata=v.get("metadata", {}))
                    for k, v in raw.items()
                }
            except Exception:
                self.store = {}

    def mget(self, keys: Sequence[str]) -> List[Optional[Document]]:
        return [self.store.get(k) for k in keys]

    def mset(self, key_value_pairs: Sequence[Tuple[str, Document]]) -> None:
        for key, value in key_value_pairs:
            self.store[key] = value
        self._persist()

    def mdelete(self, keys: Sequence[str]) -> None:
        for k in keys:
            if k in self.store:
                del self.store[k]
        self._persist()

    def yield_keys(self, *, prefix: Optional[str] = None) -> Union[Iterator[str], Iterator[str]]:
        if prefix is None:
            return iter(self.store.keys())
        return (k for k in self.store.keys() if k.startswith(prefix))

    def _persist(self) -> None:
        payload = {
            k: {"page_content": v.page_content, "metadata": v.metadata}
            for k, v in self.store.items()
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
