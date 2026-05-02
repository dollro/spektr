from __future__ import annotations

import json
from pathlib import Path


class DeltaState:
    """JSON-backed Graph delta-token persistence with atomic writes."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> str | None:
        if not self._path.exists():
            return None
        data = json.loads(self._path.read_text(encoding="utf-8"))
        token = data.get("delta_token")
        return token if isinstance(token, str) else None

    def save(self, token: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps({"delta_token": token}), encoding="utf-8")
        tmp.replace(self._path)
