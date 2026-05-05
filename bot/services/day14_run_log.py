"""JSONL logging for Day 14 security loop MVP."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOG_DIR = _PROJECT_ROOT / "logs" / "day14"


def _json_default(value: Any) -> Any:
    """Convert non-JSON values to a safe representation."""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _log_path(run_id: str) -> Path:
    return _LOG_DIR / f"run_{run_id}.jsonl"


def log_event(run_id: str, event: str, **details: Any) -> Path:
    """Append one JSONL event for a run."""
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "event": event,
            "details": details,
        }
        path = _log_path(run_id)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False, default=_json_default) + "\n")
        return path
    except Exception as exc:
        logger.exception("day14_run_log log_event failed: %s", exc)
        return _log_path(run_id)


def read_run_events(run_id: str) -> list[dict[str, Any]]:
    """Read all JSONL events for a run."""
    path = _log_path(run_id)
    if not path.exists():
        return []

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    except Exception as exc:
        logger.exception("day14_run_log read_run_events failed: %s", exc)
        return []


def read_last_event(run_id: str) -> dict[str, Any] | None:
    """Return the last event for a run."""
    events = read_run_events(run_id)
    return events[-1] if events else None
