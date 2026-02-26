"""Database service for bot memory and settings."""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone
import os

logger = logging.getLogger(__name__)

# Constants
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MEMORY_ENABLED = True
DB_PATH = Path(os.getenv("DB_PATH", str(Path(__file__).resolve().parent.parent / "bot_memory.sqlite3")))
MEMORY_LIMIT_MESSAGES = 30
MEMORY_CHAT_MODES = ("text", "thinking", "experts", "rag")


def build_messages_with_db_memory(system_prompt: str, chat_id: int) -> list[dict]:
    """Build messages list with system prompt and database history."""
    from .database import db_get_history
    from ..config import MEMORY_CHAT_MODES, MEMORY_LIMIT_MESSAGES
    history = db_get_history(chat_id=chat_id, modes=MEMORY_CHAT_MODES, limit=MEMORY_LIMIT_MESSAGES)
    return [{"role": "system", "content": system_prompt}] + history


def utc_now_iso() -> str:
    """Get current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def open_db() -> sqlite3.Connection:
    """Open database connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    """
    Ensure column exists in table.
    
    Args:
        conn: Database connection
        table: Table name
        column: Column name
        ddl: DDL statement to add column
    """
    cur = conn.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]
    if column not in cols:
        conn.execute(ddl)


def init_db() -> None:
    """Initialize database tables."""
    with open_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              chat_id INTEGER NOT NULL,
              mode TEXT NOT NULL,
              role TEXT NOT NULL,
              content TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_id_id ON messages(chat_id, id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_id_mode_id ON messages(chat_id, mode, id)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_settings (
              chat_id INTEGER PRIMARY KEY,
              temperature REAL NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )

        # Migrations
        _ensure_column(
            conn,
            table="chat_settings",
            column="memory_enabled",
            ddl="ALTER TABLE chat_settings ADD COLUMN memory_enabled INTEGER NOT NULL DEFAULT 1",
        )
        _ensure_column(
            conn,
            table="chat_settings",
            column="model",
            ddl="ALTER TABLE chat_settings ADD COLUMN model TEXT",
        )

        conn.commit()


def db_get_chat_settings(chat_id: int) -> tuple[float | None, bool | None, str | None]:
    """Get chat settings from database."""
    try:
        with open_db() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT temperature, memory_enabled, model FROM chat_settings WHERE chat_id = ?",
                (int(chat_id),),
            )
            row = cur.fetchone()
            if not row:
                return None, None, None

            temp = None
            mem = None
            model = None

            try:
                temp = float(row["temperature"])
            except Exception:
                temp = None

            try:
                mem = bool(int(row["memory_enabled"]))
            except Exception:
                mem = None

            try:
                m = row["model"]
                model = str(m).strip() if m else None
            except Exception:
                model = None

            return temp, mem, model
    except Exception as e:
        logger.exception("DB get settings failed: %s", e)
        return None, None, None


def db_set_temperature(chat_id: int, temperature: float) -> None:
    """Set temperature for chat."""
    try:
        old_temp, old_mem, old_model = db_get_chat_settings(chat_id)
        mem_val = int(old_mem) if isinstance(old_mem, bool) else int(DEFAULT_MEMORY_ENABLED)
        model_val = (old_model or "").strip() or None

        with open_db() as conn:
            conn.execute(
                """
                INSERT INTO chat_settings(chat_id, temperature, memory_enabled, model, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                  temperature=excluded.temperature,
                  updated_at=excluded.updated_at
                """,
                (int(chat_id), float(temperature), int(mem_val), model_val, utc_now_iso()),
            )
            conn.commit()
    except Exception as e:
        logger.exception("DB set temperature failed: %s", e)


def db_set_memory_enabled(chat_id: int, enabled: bool) -> None:
    """Set memory enabled for chat."""
    try:
        old_temp, old_mem, old_model = db_get_chat_settings(chat_id)
        temp_val = float(old_temp) if isinstance(old_temp, (int, float)) else float(DEFAULT_TEMPERATURE)
        model_val = (old_model or "").strip() or None

        with open_db() as conn:
            conn.execute(
                """
                INSERT INTO chat_settings(chat_id, temperature, memory_enabled, model, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                  memory_enabled=excluded.memory_enabled,
                  updated_at=excluded.updated_at
                """,
                (int(chat_id), float(temp_val), int(bool(enabled)), model_val, utc_now_iso()),
            )
            conn.commit()
    except Exception as e:
        logger.exception("DB set memory_enabled failed: %s", e)


def db_set_model(chat_id: int, model: str) -> None:
    """Set model for chat."""
    try:
        old_temp, old_mem, old_model = db_get_chat_settings(chat_id)
        temp_val = float(old_temp) if isinstance(old_temp, (int, float)) else float(DEFAULT_TEMPERATURE)
        mem_val = int(old_mem) if isinstance(old_mem, bool) else int(DEFAULT_MEMORY_ENABLED)
        model_val = (model or "").strip() or None

        with open_db() as conn:
            conn.execute(
                """
                INSERT INTO chat_settings(chat_id, temperature, memory_enabled, model, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                  model=excluded.model,
                  updated_at=excluded.updated_at
                """,
                (int(chat_id), float(temp_val), int(mem_val), model_val, utc_now_iso()),
            )
            conn.commit()
    except Exception as e:
        logger.exception("DB set model failed: %s", e)


def db_get_temperature(chat_id: int) -> float | None:
    """Get temperature from database."""
    t, _, _ = db_get_chat_settings(chat_id)
    return t


def db_get_memory_enabled(chat_id: int) -> bool | None:
    """Get memory enabled from database."""
    _, m, _ = db_get_chat_settings(chat_id)
    return m


def db_get_model(chat_id: int) -> str | None:
    """Get model from database."""
    _, _, m = db_get_chat_settings(chat_id)
    return m


def db_add_message(chat_id: int, mode: str, role: str, content: str) -> None:
    """Add message to database."""
    content = (content or "").strip()
    if not content:
        return
    
    try:
        with open_db() as conn:
            conn.execute(
                "INSERT INTO messages(chat_id, mode, role, content, created_at) VALUES(?, ?, ?, ?, ?)",
                (int(chat_id), str(mode), str(role), content, utc_now_iso()),
            )
            conn.commit()
    except Exception as e:
        logger.exception("DB add message failed: %s", e)


def db_get_messages(chat_id: int, mode: str, limit: int = MEMORY_LIMIT_MESSAGES) -> list[dict]:
    """Get messages from database."""
    try:
        with open_db() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT role, content FROM messages WHERE chat_id = ? AND mode = ? ORDER BY id DESC LIMIT ?",
                (int(chat_id), str(mode), int(limit)),
            )
            rows = cur.fetchall()
            # Reverse to get chronological order
            return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
    except Exception as e:
        logger.exception("DB get messages failed: %s", e)
        return []


def db_clear_messages(chat_id: int, mode: str | None = None) -> None:
    """Clear messages from database."""
    try:
        with open_db() as conn:
            if mode:
                conn.execute("DELETE FROM messages WHERE chat_id = ? AND mode = ?", (int(chat_id), str(mode)))
            else:
                conn.execute("DELETE FROM messages WHERE chat_id = ?", (int(chat_id),))
            conn.commit()
    except Exception as e:
        logger.exception("DB clear messages failed: %s", e)
