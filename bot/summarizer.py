import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from telegram import Update
from telegram.ext import ContextTypes

from .openrouter import chat_completion

DB_PATH = Path(__file__).resolve().parent / "bot_memory.sqlite3"

# режим чата в messages.mode
MODE_SUMMARY = "summary"

# каждые N сообщений (user/assistant) обновляем summary и удаляем старое
COMPRESS_EVERY_MESSAGES = 10

# сколько последних сообщений оставлять "живыми" (не сжимать)
KEEP_TAIL_MESSAGES = 8

# сколько последних сообщений добавлять в контекст вместе с summary
TAIL_IN_CONTEXT = 12

# опционально отдельная модель для summary (иначе возьмётся OPENROUTER_MODEL из config)
# .env: OPENROUTER_MODEL_SUMMARIZER=...
SUMMARY_MODEL = (os.getenv("OPENROUTER_MODEL_SUMMARIZER") or "").strip() or None

SYSTEM_PROMPT_SUMMARY = """
Ты сжимаешь историю диалога для дальнейшего продолжения.

Сохраняй только полезное:
- факты, числа, договорённости
- требования/ограничения/настройки
- что уже пробовали и что не сработало
- текущая цель и незакрытые вопросы

Формат: короткие пункты.
Без воды. Без художественности.
Не добавляй того, чего не было в диалоге.
""".strip()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _open_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def ensure_summary_table() -> None:
    with _open_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_summaries (
              chat_id INTEGER NOT NULL,
              mode TEXT NOT NULL,
              summary TEXT NOT NULL,
              last_message_id INTEGER NOT NULL DEFAULT 0,
              updated_at TEXT NOT NULL,
              PRIMARY KEY (chat_id, mode)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_summaries_chat_mode ON chat_summaries(chat_id, mode)")
        conn.commit()


def get_summary(chat_id: int, mode: str = MODE_SUMMARY) -> tuple[str, int]:
    ensure_summary_table()
    with _open_db() as conn:
        cur = conn.execute(
            "SELECT summary, last_message_id FROM chat_summaries WHERE chat_id=? AND mode=?",
            (int(chat_id), str(mode)),
        )
        row = cur.fetchone()
        if not row:
            return "", 0
        return (row[0] or "").strip(), int(row[1] or 0)


def _get_summary_meta(chat_id: int, mode: str = MODE_SUMMARY) -> tuple[str, int, str]:
    """
    summary, last_message_id, updated_at
    """
    ensure_summary_table()
    with _open_db() as conn:
        cur = conn.execute(
            "SELECT summary, last_message_id, updated_at FROM chat_summaries WHERE chat_id=? AND mode=?",
            (int(chat_id), str(mode)),
        )
        row = cur.fetchone()
        if not row:
            return "", 0, ""
        return (row[0] or "").strip(), int(row[1] or 0), str(row[2] or "").strip()


def set_summary(chat_id: int, summary: str, last_message_id: int, mode: str = MODE_SUMMARY) -> None:
    ensure_summary_table()
    with _open_db() as conn:
        conn.execute(
            """
            INSERT INTO chat_summaries(chat_id, mode, summary, last_message_id, updated_at)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(chat_id, mode) DO UPDATE SET
              summary=excluded.summary,
              last_message_id=excluded.last_message_id,
              updated_at=excluded.updated_at
            """,
            (int(chat_id), str(mode), (summary or "").strip(), int(last_message_id), _utc_now_iso()),
        )
        conn.commit()


def clear_summary(chat_id: int, mode: str = MODE_SUMMARY) -> None:
    ensure_summary_table()
    with _open_db() as conn:
        conn.execute("DELETE FROM chat_summaries WHERE chat_id=? AND mode=?", (int(chat_id), str(mode)))
        conn.commit()


def _fetch_messages_after(chat_id: int, after_id: int, mode: str = MODE_SUMMARY) -> list[tuple[int, str, str]]:
    with _open_db() as conn:
        cur = conn.execute(
            """
            SELECT id, role, content
            FROM messages
            WHERE chat_id=? AND mode=? AND id>?
            ORDER BY id ASC
            """,
            (int(chat_id), str(mode), int(after_id)),
        )
        rows = cur.fetchall() or []
    out: list[tuple[int, str, str]] = []
    for mid, role, content in rows:
        role = (role or "").strip()
        content = (content or "").strip()
        if role in ("user", "assistant") and content:
            out.append((int(mid), role, content))
    return out


def _fetch_tail(chat_id: int, limit: int, mode: str = MODE_SUMMARY) -> list[dict]:
    with _open_db() as conn:
        cur = conn.execute(
            """
            SELECT role, content
            FROM messages
            WHERE chat_id=? AND mode=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(chat_id), str(mode), int(limit)),
        )
        rows = cur.fetchall() or []

    rows = list(reversed(rows))
    msgs: list[dict] = []
    for role, content in rows:
        role = (role or "").strip()
        content = (content or "").strip()
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": content})
    return msgs


def _delete_messages_upto(chat_id: int, upto_id: int, mode: str = MODE_SUMMARY) -> None:
    with _open_db() as conn:
        conn.execute(
            "DELETE FROM messages WHERE chat_id=? AND mode=? AND id<=?",
            (int(chat_id), str(mode), int(upto_id)),
        )
        conn.commit()


def build_messages_with_summary(system_prompt: str, chat_id: int, mode: str = MODE_SUMMARY) -> list[dict]:
    summary, _ = get_summary(chat_id, mode=mode)
    tail = _fetch_tail(chat_id, limit=TAIL_IN_CONTEXT, mode=mode)

    msgs = [{"role": "system", "content": system_prompt}]
    if summary:
        msgs.append({"role": "system", "content": "SUMMARY ИСТОРИИ (контекст):\n" + summary})
    msgs.extend(tail)
    return msgs


def maybe_compress_history(chat_id: int, temperature: float = 0.0, mode: str = MODE_SUMMARY) -> bool:
    """
    Если накопилось >= COMPRESS_EVERY_MESSAGES новых сообщений — обновляем summary и удаляем старое.
    Возвращает True если сжимали.
    """
    summary, last_id = get_summary(chat_id, mode=mode)
    new_msgs = _fetch_messages_after(chat_id, after_id=last_id, mode=mode)

    if len(new_msgs) < COMPRESS_EVERY_MESSAGES:
        return False
    if len(new_msgs) <= KEEP_TAIL_MESSAGES:
        return False

    # сжимаем всё, кроме хвоста KEEP_TAIL_MESSAGES
    cutoff = len(new_msgs) - KEEP_TAIL_MESSAGES
    chunk = new_msgs[:cutoff]
    last_summarized_id = chunk[-1][0]

    lines = []
    for _, role, content in chunk:
        if role == "user":
            lines.append(f"USER: {content}")
        else:
            lines.append(f"ASSISTANT: {content}")
    dialog_text = "\n".join(lines).strip()

    user_payload = (
        "Текущее summary (может быть пустым):\n"
        f"{summary or '(пусто)'}\n\n"
        "Новые сообщения для добавления в summary:\n"
        f"{dialog_text}\n\n"
        "Сформируй обновлённое summary."
    )

    new_summary = (chat_completion(
        [
            {"role": "system", "content": SYSTEM_PROMPT_SUMMARY},
            {"role": "user", "content": user_payload},
        ],
        temperature=float(temperature),
        model=SUMMARY_MODEL,
    ) or "").strip()

    if not new_summary:
        return False

    set_summary(chat_id, new_summary, last_summarized_id, mode=mode)
    _delete_messages_upto(chat_id, last_summarized_id, mode=mode)
    return True


# -------------------- /summary_debug --------------------

async def summary_debug_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    chat_id = int(update.effective_chat.id) if update.effective_chat else 0

    summary, last_id, updated_at = _get_summary_meta(chat_id, mode=MODE_SUMMARY)

    # сколько "хвоста" сейчас хранится в messages для summary-режима
    try:
        with _open_db() as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE chat_id=? AND mode=?",
                (int(chat_id), MODE_SUMMARY),
            )
            tail_count = int((cur.fetchone() or [0])[0])
    except Exception:
        tail_count = -1

    lines = []
    lines.append("SUMMARY DEBUG")
    lines.append(f"chat_id: {chat_id}")
    lines.append(f"mode: {MODE_SUMMARY}")
    lines.append(f"updated_at: {updated_at or 'n/a'}")
    lines.append(f"last_message_id (сжато до): {last_id}")
    lines.append(f"messages в хвосте (в БД): {tail_count if tail_count >= 0 else 'n/a'}")
    lines.append("")
    lines.append("SUMMARY:")
    lines.append(summary if summary else "(пусто)")

    await update.message.reply_text("\n".join(lines))
