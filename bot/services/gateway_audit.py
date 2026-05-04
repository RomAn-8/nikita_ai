"""Audit logging для LLM Gateway — День 13.

Пишет JSONL-файлы по дням в logs/gateway_audit/.
Оригинальные значения секретов НИКОГДА не попадают в лог — только типы и количество.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .gateway_guard import Finding

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AUDIT_DIR = _PROJECT_ROOT / "logs" / "gateway_audit"


def _audit_path() -> Path:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return AUDIT_DIR / f"audit_{today}.jsonl"


def log_event(
    user_id: int,
    mode: str,
    input_findings: list[Finding],
    output_findings: list[Finding],
    redacted: bool,
    restored: bool,
    blocked: bool,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
) -> None:
    """Записывает одно событие в JSONL audit log.

    Значения секретов не логируются — только типы (Finding.type) и позиции заменяются
    на порядковые номера для анонимности.
    """
    try:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "mode": mode,
            "input_findings": [f.type for f in input_findings],
            "output_findings": [f.type for f in output_findings],
            "findings_count": len(input_findings),
            "output_triggered": len(output_findings) > 0,
            "redacted": redacted,
            "restored": restored,
            "blocked": blocked,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": round(cost_usd, 8),
        }
        with _audit_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.exception("gateway_audit log_event failed: %s", e)


def read_recent_events(n: int = 10) -> list[dict]:
    """Возвращает последние n событий из сегодняшнего audit log."""
    path = _audit_path()
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        recent = lines[-n:] if len(lines) > n else lines
        return [json.loads(line) for line in recent]
    except Exception as e:
        logger.exception("gateway_audit read_recent_events failed: %s", e)
        return []


def read_today_stats() -> dict:
    """Возвращает агрегированную статистику за сегодня."""
    events = read_recent_events(n=10000)
    if not events:
        return {
            "total": 0,
            "blocked": 0,
            "redacted": 0,
            "output_triggered": 0,
            "findings_total": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd": 0.0,
        }
    return {
        "total": len(events),
        "blocked": sum(1 for e in events if e.get("blocked")),
        "redacted": sum(1 for e in events if e.get("redacted")),
        "output_triggered": sum(1 for e in events if e.get("output_triggered")),
        "findings_total": sum(e.get("findings_count", 0) for e in events),
        "prompt_tokens": sum(e.get("prompt_tokens", 0) for e in events),
        "completion_tokens": sum(e.get("completion_tokens", 0) for e in events),
        "cost_usd": round(sum(e.get("cost_usd", 0.0) for e in events), 8),
    }
