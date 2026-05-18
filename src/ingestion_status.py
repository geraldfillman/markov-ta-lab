"""Persistence helpers for data ingestion status artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

RunStatus = Literal["running", "success", "partial", "failed"]
SymbolStatus = Literal["pending", "success", "error"]


@dataclass(frozen=True)
class IngestionSymbolStatus:
    symbol: str
    status: SymbolStatus
    rows: int = 0
    first_date: str | None = None
    last_date: str | None = None
    total_missing: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "status": self.status,
            "rows": self.rows,
            "first_date": self.first_date,
            "last_date": self.last_date,
            "total_missing": self.total_missing,
            "error": self.error,
        }


@dataclass(frozen=True)
class IngestionRun:
    run_id: str
    provider: str
    start: str
    end: str
    status: RunStatus
    started_at: str
    finished_at: str | None = None
    symbols: tuple[IngestionSymbolStatus, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.symbols, tuple):
            object.__setattr__(self, "symbols", tuple(self.symbols))

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "provider": self.provider,
            "start": self.start,
            "end": self.end,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "symbols": [symbol.to_dict() for symbol in self.symbols],
        }


def save_ingestion_status(run: IngestionRun, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        encoding="utf-8",
    )
    temp_path = Path(temp_file.name)

    try:
        with temp_file:
            json.dump(run.to_dict(), temp_file, indent=2, sort_keys=True)
            temp_file.write("\n")

        os.replace(temp_path, target)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        finally:
            raise

    return target


def load_ingestion_status(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None

    with target.open(encoding="utf-8") as file:
        return json.load(file)
