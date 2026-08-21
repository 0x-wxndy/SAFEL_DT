"""JSONL tracing for per-round simulator outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlWriter:
    """Append-only JSONL writer used by :func:`run_simulation`."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")

    def append(self, row: dict[str, Any]) -> None:
        self._fh.write(json.dumps(row, default=_json_default) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _json_default(obj: object) -> object:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()  # type: ignore[no-any-return]
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")
