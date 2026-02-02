import os
import json
import re
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path

from telegram import Update, BotCommand
from telegram.error import TimedOut, BadRequest
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from telegram.request import HTTPXRequest

from .config import TELEGRAM_BOT_TOKEN, OPENROUTER_MODEL
from .openrouter import chat_completion, chat_completion_raw
from .tokens_test import tokens_test_cmd, tokens_next_cmd, tokens_stop_cmd, tokens_test_intercept

# NEW: summary-mode
from .summarizer import MODE_SUMMARY, build_messages_with_summary, maybe_compress_history, clear_summary, summary_debug_cmd
from .mcp_weather import get_weather_via_mcp  # MCP-клиент для получения погоды
from .mcp_news import get_news_via_mcp  # MCP-клиент для получения новостей
from .mcp_docker import site_up_via_mcp, site_screenshot_via_mcp, site_down_via_mcp  # MCP-клиент для управления Docker
from .weather_subscription import start_weather_subscription, stop_weather_subscription  # Подписка на погоду


logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _short_model_name(m: str) -> str:
    m = (m or "").strip()
    if not m:
        return "default"
    return m.split("/")[-1]


def _get_usage_tokens(data: dict) -> tuple[int | None, int | None, int | None]:
    usage = data.get("usage") or {}
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    tt = usage.get("total_tokens")

    try:
        pt = int(pt) if pt is not None else None
    except Exception:
        pt = None
    try:
        ct = int(ct) if ct is not None else None
    except Exception:
        ct = None
    try:
        tt = int(tt) if tt is not None else None
    except Exception:
        tt = None

    return pt, ct, tt


def _get_content_from_raw(data: dict) -> str:
    try:
        return (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    except Exception:
        return ""


def _city_prepositional_case(city: str) -> str:
    """
    Склоняет название города в предложный падеж (где? в чём?).
    Примеры: Москва -> Москве, Самара -> Самаре, Саратов -> Саратове, Томск -> Томске.
    """
    city = (city or "").strip()
    if not city:
        return city
    
    # Простая эвристика для склонения русских названий городов
    city_lower = city.lower()
    
    # Если заканчивается на "а" (Москва, Самара, Тула) -> "е" (в Москве, в Самаре, в Туле)
    if city_lower.endswith("а"):
        return city[:-1] + "е"
    
    # Если заканчивается на "о" (Тула уже обработана, но на всякий случай)
    if city_lower.endswith("о"):
        return city[:-1] + "е"
    
    # Если заканчивается на "ь" (Тверь, Рязань) -> "и" (в Твери, в Рязани)
    if city_lower.endswith("ь"):
        return city[:-1] + "и"
    
    # Если заканчивается на согласную (Саратов, Томск, Новосибирск) -> "е" (в Саратове, в Томске, в Новосибирске)
    # Проверяем последнюю букву
    last_char = city_lower[-1]
    if last_char not in "аеёиоуыэюяь":
        return city + "е"
    
    # Если не подошло ни одно правило, возвращаем как есть
    return city


# -------------------- TEMPERATURE --------------------

DEFAULT_TEMPERATURE = 0.7
TEMPERATURE_MIN = 0.0
TEMPERATURE_MAX = 2.0

# -------------------- MEMORY SWITCH --------------------

DEFAULT_MEMORY_ENABLED = True  # по умолчанию память включена

# -------------------- MODELS FROM ENV --------------------
# Добавь в .env:
# OPENROUTER_MODEL_GLM=z-ai/glm-4.7-flash
# OPENROUTER_MODEL_GEMMA=google/gemma-3-12b-it

MODEL_GLM = (os.getenv("OPENROUTER_MODEL_GLM") or "").strip()
MODEL_GEMMA = (os.getenv("OPENROUTER_MODEL_GEMMA") or "").strip()


# -------------------- SQLITE MEMORY + SETTINGS --------------------

DB_PATH = Path(__file__).resolve().parent / "bot_memory.sqlite3"
MEMORY_LIMIT_MESSAGES = 30  # сколько последних сообщений хранить в контексте для LLM
MEMORY_CHAT_MODES = ("text", "thinking", "experts")  # общая память между этими режимами


def open_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    """
    ddl пример: 'ALTER TABLE chat_settings ADD COLUMN memory_enabled INTEGER NOT NULL DEFAULT 1'
    """
    cur = conn.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]  # (cid, name, type, notnull, dflt_value, pk)
    if column not in cols:
        conn.execute(ddl)


def init_db() -> None:
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

        # миграции: добавляем колонки если таблица уже существовала раньше
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
    t, _, _ = db_get_chat_settings(chat_id)
    return t


def db_get_memory_enabled(chat_id: int) -> bool | None:
    _, m, _ = db_get_chat_settings(chat_id)
    return m


def db_get_model(chat_id: int) -> str | None:
    _, _, m = db_get_chat_settings(chat_id)
    return m


def get_temperature(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> float:
    t = context.user_data.get("temperature", None)
    if isinstance(t, (int, float)):
        return float(t)

    db_t = db_get_temperature(chat_id)
    if isinstance(db_t, (int, float)):
        context.user_data["temperature"] = float(db_t)
        return float(db_t)

    context.user_data["temperature"] = float(DEFAULT_TEMPERATURE)
    return float(DEFAULT_TEMPERATURE)


def get_memory_enabled(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> bool:
    v = context.user_data.get("memory_enabled", None)
    if isinstance(v, bool):
        return v

    db_v = db_get_memory_enabled(chat_id)
    if isinstance(db_v, bool):
        context.user_data["memory_enabled"] = bool(db_v)
        return bool(db_v)

    context.user_data["memory_enabled"] = bool(DEFAULT_MEMORY_ENABLED)
    return bool(DEFAULT_MEMORY_ENABLED)


def get_model(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> str:
    v = context.user_data.get("model", None)
    if isinstance(v, str) and v.strip():
        return v.strip()

    db_v = db_get_model(chat_id)
    if isinstance(db_v, str) and db_v.strip():
        context.user_data["model"] = db_v.strip()
        return db_v.strip()

    # пустая строка => openrouter.py возьмёт OPENROUTER_MODEL из config
    return ""


def get_effective_model(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> str:
    selected = get_model(context, chat_id)
    return selected if selected else OPENROUTER_MODEL


def clamp_temperature(value: float) -> float:
    if value < TEMPERATURE_MIN:
        return TEMPERATURE_MIN
    if value > TEMPERATURE_MAX:
        return TEMPERATURE_MAX
    return value


def db_add_message(chat_id: int, mode: str, role: str, content: str) -> None:
    content = (content or "").strip()
    if not content:
        return
    try:
        with open_db() as conn:
            conn.execute(
                "INSERT INTO messages(chat_id, mode, role, content, created_at) VALUES(?,?,?,?,?)",
                (int(chat_id), str(mode), str(role), content, utc_now_iso()),
            )
            conn.commit()
    except Exception as e:
        logger.exception("DB add failed: %s", e)


def db_clear_history(chat_id: int) -> None:
    try:
        with open_db() as conn:
            conn.execute("DELETE FROM messages WHERE chat_id = ?", (int(chat_id),))
            conn.commit()
    except Exception as e:
        logger.exception("DB clear history failed: %s", e)


def db_get_history(chat_id: int, modes: tuple[str, ...], limit: int) -> list[dict]:
    placeholders = ",".join(["?"] * len(modes))
    sql = f"""
        SELECT role, content
        FROM messages
        WHERE chat_id = ? AND mode IN ({placeholders})
        ORDER BY id DESC
        LIMIT ?
    """
    try:
        with open_db() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(sql, (int(chat_id), *modes, int(limit)))
            rows = cur.fetchall()
    except Exception as e:
        logger.exception("DB read failed: %s", e)
        return []

    rows = list(reversed(rows))
    out: list[dict] = []
    for r in rows:
        role = (r["role"] or "").strip()
        content = (r["content"] or "").strip()
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content})
    return out


def build_messages_with_db_memory(system_prompt: str, chat_id: int) -> list[dict]:
    history = db_get_history(chat_id=chat_id, modes=MEMORY_CHAT_MODES, limit=MEMORY_LIMIT_MESSAGES)
    return [{"role": "system", "content": system_prompt}] + history


# -------------------- PROMPTS --------------------

SYSTEM_PROMPT_JSON = """
Всегда отвечай строго одним валидным JSON-объектом. Никакого текста вне JSON. Никакого markdown.

Схема (всегда все поля, без дополнительных):
{
  "title": "",
  "time": "",
  "tag": "",
  "answer": "",
  "steps": [],
  "warnings": [],
  "need_clarification": false,
  "clarifying_question": ""
}

Правила:
- time всегда оставляй пустым "" (его заполнит бот).
- steps и warnings всегда массивы строк.
- need_clarification=true -> clarifying_question содержит ровно один вопрос, иначе "".
- Никаких новых полей. Никаких комментариев. Только валидный JSON.
"""

SYSTEM_PROMPT_TEXT = """
Ты ассистент в Telegram. Отвечай обычным текстом, кратко и по делу.
Если данных не хватает — задай один уточняющий вопрос.
"""

SYSTEM_PROMPT_TZ = """
Ты — AI-интервьюер, который собирает требования для ТЗ на создание сайта.

РЕЖИМ РАБОТЫ:
1) Пока данных недостаточно — отвечай ТОЛЬКО обычным текстом и задай РОВНО ОДИН следующий вопрос.
2) Когда данных достаточно — верни ТОЛЬКО один валидный JSON по схеме ниже (без любого текста до/после).
3) Вопросов должно быть мало: старайся уложиться в 3–4 вопроса. Как только понятно — сразу финализируй JSON.

СХЕМА JSON (всегда все поля, без дополнительных):
{
  "title": "ТЗ на создание сайта",
  "time": "",
  "tag": "tz_site",
  "answer": "",
  "steps": [],
  "warnings": [],
  "need_clarification": false,
  "clarifying_question": ""
}

ПРАВИЛА:
- Пока ты задаёшь вопросы — НЕ ПИШИ JSON.
- Когда финализируешь — пиши ТОЛЬКО JSON.
- time в JSON оставляй пустым "" (его заполнит бот).
- steps/warnings всегда массивы строк.
- Не добавляй новых полей.
"""

SYSTEM_PROMPT_FOREST = """
Ты — AI-ассистент, который рассчитывает, кто кому сколько должен перевести за общие расходы (поход/лес/кафе).

ВАЖНО: весь диалог (вопросы и ответы) — обычным текстом.
Когда данных достаточно — ты должен САМ остановиться и выдать финальный результат.

РЕЖИМ РАБОТЫ:
1) Пока данных недостаточно — задай РОВНО ОДИН вопрос за сообщение.
2) Старайся уложиться в 3–4 вопроса. Не растягивай.
3) Как только данных достаточно — выдай финальный расчет и больше вопросов не задавай.

ЧТО НУЖНО СОБРАТЬ:
- Список участников (имена).
- Сколько заплатил каждый (в рублях).
- Как делим расходы: "поровну" (по умолчанию) или "по долям" (если пользователь явно скажет, тогда спроси доли).

ПРЕДПОЧТИТЕЛЬНЫЙ ФОРМАТ СБОРА (чтобы вопросов было мало):
- 1-й вопрос: "Кто участники? (перечисли через запятую)"
- 2-й вопрос: "Напиши, кто сколько заплатил одной строкой: Имя сумма, Имя сумма, ..."
- 3-й вопрос (если не сказано): "Делим поровну? (да/нет). Если нет — как делим?"

АЛГОРИТМ (делай сам, без Python):
- Total = сумма всех оплат.
- Если делим поровну: Share = Total / N.
- Баланс участника = paid - share.
  - balance > 0: должен получить
  - balance < 0: должен заплатить
- Составь переводы от должников к получателям так, чтобы закрыть балансы.
- Всегда сделай проверку: сумма балансов = 0 (или очень близко из-за округления).

ОКРУГЛЕНИЕ:
- Если суммы целые — работай в целых.
- Если появляются копейки — округляй до 2 знаков и в конце проверь, чтобы переводы сошлись.

ФОРМАТ ВЫВОДА ФИНАЛА (ОДИН РАЗ, в конце):
1) Коротко: Total, N, Share (или правило деления)
2) Таблица строками:
   Имя: paid=..., share=..., balance=... (получить/заплатить ...)
3) "Финальные переводы:" списком "Имя -> Имя: сумма"
4) Строка "Проверка: сумма балансов = ..."

КРИТИЧЕСКОЕ ПРАВИЛО:
- Слово "FINAL" пиши ТОЛЬКО в самом начале финального сообщения и только один раз.
- До финала "FINAL" не писать.
"""

SYSTEM_PROMPT_THINKING = """
Ты решаешь задачи в режиме "пошаговое рассуждение".
Правила:
- Решай задачу пошагово.
- В конце дай короткий итоговый ответ отдельной строкой: "ИТОГ: ...".
- Пиши понятно и без воды.
"""

SYSTEM_PROMPT_EXPERTS = """
Ты решаешь задачу как "группа экспертов" внутри одного ответа.

Эксперты:
1) Логик — строгая проверка условий, поиск противоречий.
2) Математик — вычисления/формулы/аккуратная арифметика (если нужна).
3) Ревизор — проверяет решения Логика и Математика, ищет ошибки, даёт финальную сверку.

Формат ответа строго такой:
ЛОГИК:
...

МАТЕМАТИК:
...

РЕВИЗОР:
...

ИТОГ:
(одна финальная формулировка результата)

Правила:
- Все три части должны быть.
- Пиши кратко, но так, чтобы было ясно, почему итог верный.
"""


# -------------------- HELPERS --------------------

TELEGRAM_MESSAGE_LIMIT = 3900  # безопаснее 4096


def split_telegram_text(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    t = (text or "").strip()
    if not t:
        return [""]

    if len(t) <= limit:
        return [t]

    parts: list[str] = []
    cur = t
    while len(cur) > limit:
        cut = cur.rfind("\n", 0, limit)
        if cut < 200:
            cut = limit
        parts.append(cur[:cut].rstrip())
        cur = cur[cut:].lstrip()
    if cur:
        parts.append(cur)
    return parts


async def safe_reply_text(update: Update, text: str) -> None:
    if not update.message:
        return

    chunks = split_telegram_text(text)
    for ch in chunks:
        try:
            await update.message.reply_text(ch)
        except TimedOut:
            return
        except BadRequest as e:
            msg = str(e).lower()
            if "message is too long" in msg and len(ch) > 500:
                for sub in split_telegram_text(ch, limit=2000):
                    try:
                        await update.message.reply_text(sub)
                    except Exception:
                        return
                continue
            return
        except Exception:
            return


def extract_json_object(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("JSON object not found in model output")
    return m.group(0)


def normalize_payload(data: dict) -> dict:
    normalized = {
        "title": str(data.get("title", "")).strip() or "Ответ",
        "time": utc_now_iso(),
        "tag": str(data.get("tag", "")).strip() or "general",
        "answer": str(data.get("answer", "")).strip(),
        "steps": data.get("steps", []),
        "warnings": data.get("warnings", []),
        "need_clarification": bool(data.get("need_clarification", False)),
        "clarifying_question": str(data.get("clarifying_question", "")).strip(),
    }

    if not isinstance(normalized["steps"], list):
        normalized["steps"] = []
    if not isinstance(normalized["warnings"], list):
        normalized["warnings"] = []

    normalized["steps"] = [str(x).strip() for x in normalized["steps"] if str(x).strip()]
    normalized["warnings"] = [str(x).strip() for x in normalized["warnings"] if str(x).strip()]

    if normalized["need_clarification"]:
        if not normalized["clarifying_question"]:
            normalized["clarifying_question"] = "Уточни, пожалуйста: что именно ты имеешь в виду?"
        if not normalized["answer"]:
            normalized["answer"] = normalized["clarifying_question"]
    else:
        normalized["clarifying_question"] = ""

    if not normalized["answer"]:
        normalized["answer"] = "Пустой ответ от модели."

    return normalized


def repair_json_with_model(system_prompt: str, raw: str, temperature: float, model: str | None) -> str:
    repair_prompt = (
        system_prompt
        + "\n\nИсправь следующий ответ так, чтобы он стал валидным JSON строго по схеме. Верни только JSON."
    )
    fixed = chat_completion(
        [
            {"role": "system", "content": repair_prompt},
            {"role": "user", "content": raw or ""},
        ],
        temperature=temperature,
        model=model,
    )
    return fixed


def get_mode(context: ContextTypes.DEFAULT_TYPE) -> str:
    return context.user_data.get("mode", "text")  # text | json | tz | forest | thinking | experts | summary


def looks_like_json(text: str) -> bool:
    t = (text or "").lstrip()
    return (t.startswith("{") and t.endswith("}")) or t.startswith("{")


def is_forest_final(text: str) -> bool:
    t = (text or "").lstrip()
    return t.upper().startswith("FINAL")


def strip_forest_final_marker(text: str) -> str:
    lines = (text or "").splitlines()
    if not lines:
        return ""
    if lines[0].strip().upper() == "FINAL":
        return "\n".join(lines[1:]).strip()
    return (text or "").strip()


def user_asked_to_show_result(user_text: str) -> bool:
    t = (user_text or "").strip().lower()
    keywords = ["покажи", "выведи", "результат", "расч", "итог", "финал", "переводы", "кто кому"]
    return any(k in t for k in keywords)


def reset_tz(context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.pop("tz_history", None)
    context.user_data.pop("tz_questions", None)
    context.user_data.pop("tz_done", None)


def reset_forest(context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.pop("forest_history", None)
    context.user_data.pop("forest_questions", None)
    context.user_data.pop("forest_done", None)
    context.user_data.pop("forest_result", None)


# -------------------- COMMANDS --------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    mode = get_mode(context)
    chat_id = int(update.effective_chat.id) if update.effective_chat else 0
    t = get_temperature(context, chat_id)
    mem = get_memory_enabled(context, chat_id)
    current_model = get_effective_model(context, chat_id)

    lines = [
        "Привет!",
        "",
        "Команды:",
        f"/mode_text — режим text + {_short_model_name(OPENROUTER_MODEL)}",
        "/mode_json — JSON на каждое сообщение",
        f"/mode_summary — режим summary + {_short_model_name(OPENROUTER_MODEL)} (сжатие истории)",
        "/tz_creation_site — требования для ТЗ (вопросы текстом, итог JSON)",
        "/forest_split — кто кому должен (вопросы текстом, итог текстом)",
        "/thinking_model — решай пошагово",
        "/expert_group_model — группа экспертов",
        "/tokens_test — тест токенов (режим: короткий/длинный/перелимит)",
        "/tokens_next — следующий этап теста токенов",
        "/tokens_stop — сводка и выход из теста токенов",
        "/ch_temperature — показать/изменить температуру (пример: /ch_temperature 0.7)",
        "/ch_memory — память ВКЛ/ВЫКЛ (пример: /ch_memory off)",
        "/clear_memory — очистить память чата",
    ]

    if MODEL_GLM:
        lines.append(f"/model_glm — модель {_short_model_name(MODEL_GLM)}")
    if MODEL_GEMMA:
        lines.append(f"/model_gemma — модель {_short_model_name(MODEL_GEMMA)}")

    lines.extend([
        "",
        f"Текущий режим: {mode}",
        f"Температура: {t}",
        f"Память: {'ВКЛ' if mem else 'ВЫКЛ'}",
        f"Модель: {current_model}",
    ])

    await safe_reply_text(update, "\n".join(lines))


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lines = [
        "Команды:",
        f"/mode_text — режим text + {_short_model_name(OPENROUTER_MODEL)}",
        "/mode_json — JSON на каждое сообщение",
        f"/mode_summary — режим summary + {_short_model_name(OPENROUTER_MODEL)} (сжатие истории)",
        "/tz_creation_site — собрать ТЗ на сайт (в конце JSON)",
        "/forest_split — посчитать кто кому должен (в конце текст)",
        "/thinking_model — решать пошагово",
        "/expert_group_model — решить как группа экспертов",
        "/tokens_test — тест токенов (режим)",
        "/tokens_next — следующий этап теста токенов",
        "/tokens_stop — сводка и выход из теста токенов",
        "/ch_temperature — показать/изменить температуру (пример: /ch_temperature 1.2)",
        "/ch_memory — память ВКЛ/ВЫКЛ (пример: /ch_memory on)",
        "/clear_memory — очистить историю памяти",
    ]

    if MODEL_GLM:
        lines.append(f"/model_glm — переключить на {MODEL_GLM}")
    if MODEL_GEMMA:
        lines.append(f"/model_gemma — переключить на {MODEL_GEMMA}")

    await safe_reply_text(update, "\n".join(lines))


async def ch_temperature_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = int(update.effective_chat.id) if update.effective_chat else 0

    if not context.args:
        t = get_temperature(context, chat_id)
        await safe_reply_text(
            update,
            f"Текущая температура: {t}\n"
            f"Изменить: /ch_temperature <число от {TEMPERATURE_MIN} до {TEMPERATURE_MAX}>\n"
            "Примеры: /ch_temperature 0, /ch_temperature 0.7, /ch_temperature 1.2"
        )
        return

    raw = (context.args[0] or "").replace(",", ".").strip()
    try:
        val = float(raw)
    except Exception:
        await safe_reply_text(update, "Не понял число. Пример: /ch_temperature 0.7")
        return

    val = clamp_temperature(val)

    context.user_data["temperature"] = val
    db_set_temperature(chat_id, val)

    await safe_reply_text(update, f"Ок. Температура установлена: {val}")


async def ch_memory_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /ch_memory
    /ch_memory on|off
    """
    chat_id = int(update.effective_chat.id) if update.effective_chat else 0

    if not context.args:
        mem = get_memory_enabled(context, chat_id)
        await safe_reply_text(
            update,
            f"Память сейчас: {'ВКЛ' if mem else 'ВЫКЛ'}\n"
            "Изменить: /ch_memory on или /ch_memory off\n"
            "Пример: /ch_memory off (для честных тестов температуры)"
        )
        return

    v = (context.args[0] or "").strip().lower()
    truthy = {"on", "1", "true", "yes", "y", "да", "вкл"}
    falsy = {"off", "0", "false", "no", "n", "нет", "выкл"}

    if v in truthy:
        enabled = True
    elif v in falsy:
        enabled = False
    else:
        await safe_reply_text(update, "Не понял. Используй: /ch_memory on или /ch_memory off")
        return

    context.user_data["memory_enabled"] = enabled
    db_set_memory_enabled(chat_id, enabled)

    await safe_reply_text(update, f"Ок. Память: {'ВКЛ' if enabled else 'ВЫКЛ'}")


async def clear_memory_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = int(update.effective_chat.id) if update.effective_chat else 0
    db_clear_history(chat_id)

    # NEW: чистим summary-таблицу тоже
    try:
        clear_summary(chat_id, mode=MODE_SUMMARY)
    except Exception:
        pass

    await safe_reply_text(update, "Ок. Память чата очищена.")


async def model_glm_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not MODEL_GLM:
        await safe_reply_text(update, "Модель OPENROUTER_MODEL_GLM не задана в .env")
        return
    chat_id = int(update.effective_chat.id) if update.effective_chat else 0
    context.user_data["model"] = MODEL_GLM
    db_set_model(chat_id, MODEL_GLM)
    await safe_reply_text(update, f"Ок. Модель установлена: {MODEL_GLM}")


async def model_gemma_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not MODEL_GEMMA:
        await safe_reply_text(update, "Модель OPENROUTER_MODEL_GEMMA не задана в .env")
        return
    chat_id = int(update.effective_chat.id) if update.effective_chat else 0
    context.user_data["model"] = MODEL_GEMMA
    db_set_model(chat_id, MODEL_GEMMA)
    await safe_reply_text(update, f"Ок. Модель установлена: {MODEL_GEMMA}")


async def mode_text_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = int(update.effective_chat.id) if update.effective_chat else 0

    context.user_data["mode"] = "text"
    reset_tz(context)
    reset_forest(context)

    # Сброс на дефолтную модель из .env (OPENROUTER_MODEL)
    context.user_data.pop("model", None)
    db_set_model(chat_id, "")

    await safe_reply_text(update, f"Ок. Режим: text. Модель: {OPENROUTER_MODEL}")


async def mode_json_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["mode"] = "json"
    reset_tz(context)
    reset_forest(context)

    payload = {
        "title": "Режим установлен",
        "time": utc_now_iso(),
        "tag": "system",
        "answer": "Ок. Режим установлен: json",
        "steps": [],
        "warnings": [],
        "need_clarification": False,
        "clarifying_question": "",
    }
    context.user_data["last_payload"] = payload
    await safe_reply_text(update, json.dumps(payload, ensure_ascii=False, indent=2))


# NEW: summary mode command
async def mode_summary_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = int(update.effective_chat.id) if update.effective_chat else 0

    context.user_data["mode"] = MODE_SUMMARY
    reset_tz(context)
    reset_forest(context)

    # В summary-режиме память нужна всегда
    context.user_data["memory_enabled"] = True
    db_set_memory_enabled(chat_id, True)

    await safe_reply_text(update, "Ок. Режим: summary (сжатие истории: summary вместо полной истории).")


async def thinking_model_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["mode"] = "thinking"
    reset_tz(context)
    reset_forest(context)
    await safe_reply_text(update, "Ок. Режим установлен: thinking_model (пошаговое решение).")


async def expert_group_model_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["mode"] = "experts"
    reset_tz(context)
    reset_forest(context)
    await safe_reply_text(update, "Ок. Режим установлен: expert_group_model (Логик/Математик/Ревизор).")


async def tz_creation_site_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["mode"] = "tz"
    context.user_data["tz_history"] = []
    context.user_data["tz_questions"] = 0
    context.user_data["tz_done"] = False
    reset_forest(context)

    chat_id = int(update.effective_chat.id) if update.effective_chat else 0
    temperature = get_temperature(context, chat_id)
    model = get_model(context, chat_id) or None

    first = (chat_completion(
        [
            {"role": "system", "content": SYSTEM_PROMPT_TZ},
            {"role": "user", "content": "Начни. Задай первый вопрос, чтобы собрать требования для ТЗ на создание сайта."},
        ],
        temperature=temperature,
        model=model,
    ) or "").strip()

    if looks_like_json(first):
        await send_final_tz_json(update, context, first, temperature=temperature, model=model)
        return

    context.user_data["tz_questions"] = 1
    context.user_data["tz_history"].append({"role": "assistant", "content": first})
    await safe_reply_text(update, first)


async def forest_split_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["mode"] = "forest"
    context.user_data["forest_history"] = []
    context.user_data["forest_questions"] = 0
    context.user_data["forest_done"] = False
    context.user_data.pop("forest_result", None)
    reset_tz(context)

    chat_id = int(update.effective_chat.id) if update.effective_chat else 0
    temperature = get_temperature(context, chat_id)
    model = get_model(context, chat_id) or None

    first = (chat_completion(
        [
            {"role": "system", "content": SYSTEM_PROMPT_FOREST},
            {"role": "user", "content": "Начни. Задай первый вопрос для расчёта кто кому сколько должен."},
        ],
        temperature=temperature,
        model=model,
    ) or "").strip()

    context.user_data["forest_questions"] = 1
    context.user_data["forest_history"].append({"role": "assistant", "content": first})
    await safe_reply_text(update, first)


# -------------------- WEATHER SUBSCRIPTION --------------------
async def weather_sub_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Команда для подписки на погоду с периодическим сбором данных.
    Формат: /weather_sub <Город> <время_в_секундах>
    Пример: /weather_sub Москва 30
    """
    chat_id = int(update.effective_chat.id) if update.effective_chat else 0

    if not context.args or len(context.args) < 2:
        await safe_reply_text(
            update,
            "Использование: /weather_sub <Город> <время_в_секундах>\n"
            "Пример: /weather_sub Москва 30\n"
            "Подписка будет собирать погоду каждые 10 секунд и отправлять summary каждые указанные секунды.",
        )
        return

    city = context.args[0].strip()
    try:
        summary_interval = int(context.args[1])
        if summary_interval < 10:
            await safe_reply_text(update, "Интервал summary должен быть не менее 10 секунд.")
            return
    except ValueError:
        await safe_reply_text(update, "Время должно быть числом (в секундах).")
        return

    # Запускаем подписку
    try:
        start_weather_subscription(
            chat_id=chat_id,
            city=city,
            summary_interval=summary_interval,
            bot=context.bot,
            context=context,
            db_add_message=db_add_message,
        )
    except Exception as e:
        logger.exception(f"Failed to start weather subscription: {e}")
        await safe_reply_text(update, f"Ошибка при запуске подписки: {e}")


async def weather_sub_stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Команда для остановки подписки на погоду.
    Формат: /weather_sub_stop <Город>
    """
    chat_id = int(update.effective_chat.id) if update.effective_chat else 0

    if not context.args or len(context.args) < 1:
        await safe_reply_text(update, "Использование: /weather_sub_stop <Город>\nПример: /weather_sub_stop Москва")
        return

    city = context.args[0].strip()
    stopped = stop_weather_subscription(chat_id=chat_id, city=city, context=context)

    if stopped:
        await safe_reply_text(update, f"✅ Подписка на погоду для {city} остановлена.")
    else:
        await safe_reply_text(update, f"❌ Подписка на погоду для {city} не найдена.")


# -------------------- DIGEST COMMAND --------------------

async def digest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Команда для создания утренней сводки: погода + новости.
    Формат: /digest <город погоды>, <тема новостей>
    Пример: /digest Москва, технологии
    """
    if not update.message:
        return
    
    chat_id = int(update.effective_chat.id) if update.effective_chat else 0
    
    # Парсим аргументы: город и тема новостей (через запятую)
    if not context.args:
        await safe_reply_text(
            update,
            "Использование: /digest <город погоды>, <тема новостей>\n"
            "Пример: /digest Москва, технологии\n"
            "Пример: /digest Самара, спорт"
        )
        return
    
    # Объединяем все аргументы и разбиваем по запятой
    full_text = " ".join(context.args)
    parts = [p.strip() for p in full_text.split(",", 1)]
    
    if len(parts) < 2:
        await safe_reply_text(
            update,
            "Неверный формат. Используйте: /digest <город>, <тема>\n"
            "Пример: /digest Москва, технологии"
        )
        return
    
    city = parts[0]
    news_topic = parts[1]
    
    if not city or not news_topic:
        await safe_reply_text(update, "Город и тема новостей должны быть указаны.")
        return
    
    await update.message.chat.send_action("typing")
    
    # Склоняем город в предложный падеж для использования в тексте
    city_prep = _city_prepositional_case(city)
    
    # Получаем погоду через MCP
    weather_text = await get_weather_via_mcp(city)
    
    # Получаем новости через MCP (5 новостей)
    news_text = await get_news_via_mcp(news_topic, count=5)
    
    # Формируем Markdown файл
    from datetime import datetime, timedelta, timezone
    
    # Самарское время (UTC+4)
    SAMARA_OFFSET = timedelta(hours=4)
    SAMARA_TIMEZONE = timezone(SAMARA_OFFSET)
    now = datetime.now(SAMARA_TIMEZONE)
    date_str = now.strftime("%d.%m.%Y %H:%M")
    
    markdown_content = f"""# Сводка погоды в {city_prep} и новости по теме {news_topic}
**Дата:** {date_str}

## Погода: {city}

{weather_text}

## Новости: {news_topic}

{news_text}

---
*Сгенерировано автоматически*
"""
    
    # Сохраняем Markdown файл
    digest_dir = Path(__file__).resolve().parent / "digests"
    digest_dir.mkdir(exist_ok=True)
    filename = f"digest_{chat_id}_{now.strftime('%Y%m%d_%H%M%S')}.md"
    filepath = digest_dir / filename
    
    try:
        filepath.write_text(markdown_content, encoding="utf-8")
    except Exception as e:
        logger.exception(f"Failed to save digest file: {e}")
        await safe_reply_text(update, f"Ошибка при сохранении файла: {e}")
        return
    
    # Формируем текст для ИИ
    mode = MODE_SUMMARY
    temperature = get_temperature(context, chat_id)
    model = get_model(context, chat_id) or None
    
    # Создаём промпт для ИИ
    system_prompt = """Ты помощник, который формирует сводку на основе данных о погоде и новостях.
Сделай сводку краткой, информативной и приятной для чтения.
Используй данные о погоде и новостях, которые тебе предоставлены."""
    
    user_prompt = f"""Создай сводку на основе следующих данных:

ПОГОДА:
{weather_text}

НОВОСТИ:
{news_text}

ВАЖНО: Начни сводку с фразы "Сводка погоды в {city_prep} и новости по теме {news_topic}!" (без кавычек).
Затем сформируй краткую и информативную сводку, которая объединяет погоду и новости."""
    
    # Получаем ответ от ИИ через mode_summary
    try:
        messages = build_messages_with_summary(system_prompt, chat_id=chat_id, mode=mode)
        messages.append({"role": "user", "content": user_prompt})
        
        data = chat_completion_raw(messages, temperature=temperature, model=model)
        ai_response = _get_content_from_raw(data)
        
        if not ai_response:
            ai_response = f"Погода: {weather_text}\n\nНовости: {news_text}"
        
        # Сохраняем в БД
        db_add_message(chat_id, mode, "user", f"/digest {city}, {news_topic}")
        db_add_message(chat_id, mode, "assistant", ai_response)
        
        # Сжимаем историю
        try:
            maybe_compress_history(chat_id, temperature=0.0, mode=mode)
        except Exception:
            pass
        
        # Отправляем ответ от ИИ
        await safe_reply_text(update, ai_response)
        
        # Отправляем Markdown файл
        try:
            with open(filepath, "rb") as f:
                await update.message.reply_document(
                    document=f,
                    filename=filename,
                    caption=f"📄 Markdown файл со сводкой: {city}, {news_topic}"
                )
        except Exception as e:
            logger.exception(f"Failed to send digest file: {e}")
            await safe_reply_text(update, f"⚠️ Сводка создана, но не удалось отправить файл: {e}")
    
    except Exception as e:
        logger.exception(f"Failed to generate digest: {e}")
        await safe_reply_text(update, f"Ошибка при создании сводки: {e}")


# -------------------- TZ FLOW --------------------

async def send_final_tz_json(update: Update, context: ContextTypes.DEFAULT_TYPE, raw: str, temperature: float, model: str | None) -> None:
    try:
        json_str = extract_json_object(raw)
        data = json.loads(json_str)
        payload = normalize_payload(data)
    except Exception:
        try:
            fixed_raw = repair_json_with_model(SYSTEM_PROMPT_TZ, raw, temperature=temperature, model=model)
            json_str = extract_json_object(fixed_raw)
            data = json.loads(json_str)
            payload = normalize_payload(data)
        except Exception as e2:
            err_payload = {
                "title": "Ошибка",
                "time": utc_now_iso(),
                "tag": "error",
                "answer": "Модель вернула непарсируемый формат для итогового ТЗ.",
                "steps": [],
                "warnings": [str(e2)],
                "need_clarification": False,
                "clarifying_question": "",
            }
            await safe_reply_text(update, json.dumps(err_payload, ensure_ascii=False, indent=2))
            return

    context.user_data["tz_done"] = True
    context.user_data["last_payload"] = payload
    await safe_reply_text(update, json.dumps(payload, ensure_ascii=False, indent=2))


async def handle_tz_message(update: Update, context: ContextTypes.DEFAULT_TYPE, user_text: str, temperature: float, model: str | None) -> None:
    if context.user_data.get("tz_done"):
        await safe_reply_text(update, "ТЗ уже сформировано. Если хочешь заново — вызови /tz_creation_site.")
        return

    history = context.user_data.get("tz_history", [])
    questions_asked = int(context.user_data.get("tz_questions", 0))

    history.append({"role": "user", "content": user_text})

    force_finalize = questions_asked >= 4

    messages = [{"role": "system", "content": SYSTEM_PROMPT_TZ}]
    messages.extend(history)
    if force_finalize:
        messages.append({"role": "user", "content": "Сформируй финальное ТЗ прямо сейчас. Верни только JSON по схеме."})

    try:
        raw = (chat_completion(messages, temperature=temperature, model=model) or "").strip()
    except Exception as e:
        await safe_reply_text(update, f"Ошибка запроса к LLM: {e}")
        return

    if looks_like_json(raw):
        await send_final_tz_json(update, context, raw, temperature=temperature, model=model)
        return

    history.append({"role": "assistant", "content": raw})
    context.user_data["tz_history"] = history
    context.user_data["tz_questions"] = questions_asked + 1
    await safe_reply_text(update, raw)


# -------------------- FOREST FLOW --------------------

async def handle_forest_message(update: Update, context: ContextTypes.DEFAULT_TYPE, user_text: str, temperature: float, model: str | None) -> None:
    if context.user_data.get("forest_done"):
        if user_asked_to_show_result(user_text):
            res = (context.user_data.get("forest_result") or "").strip()
            if res:
                await safe_reply_text(update, res)
            else:
                await safe_reply_text(update, "Расчёт готов, но результат не сохранён. Запусти /forest_split заново.")
            return
        await safe_reply_text(update, "Расчёт уже готов. Если хочешь заново — вызови /forest_split.")
        return

    history = context.user_data.get("forest_history", [])
    questions_asked = int(context.user_data.get("forest_questions", 0))

    history.append({"role": "user", "content": user_text})

    force_finalize = questions_asked >= 6

    messages = [{"role": "system", "content": SYSTEM_PROMPT_FOREST}]
    messages.extend(history)
    if force_finalize:
        messages.append({
            "role": "user",
            "content": "Хватит вопросов. Сформируй финальный отчёт прямо сейчас. Первая строка FINAL, далее отчёт текстом."
        })

    try:
        raw = (chat_completion(messages, temperature=temperature, model=model) or "").strip()
    except Exception as e:
        await safe_reply_text(update, f"Ошибка запроса к LLM: {e}")
        return

    if not raw:
        await safe_reply_text(update, "Пустой ответ от модели.")
        return

    if is_forest_final(raw):
        report = strip_forest_final_marker(raw)
        if not report:
            await safe_reply_text(update, "Ошибка: финал без отчёта. Запусти /forest_split заново.")
            return

        context.user_data["forest_done"] = True
        context.user_data["forest_result"] = report
        history.append({"role": "assistant", "content": raw})
        context.user_data["forest_history"] = history
        await safe_reply_text(update, report)
        return

    history.append({"role": "assistant", "content": raw})
    context.user_data["forest_history"] = history
    context.user_data["forest_questions"] = questions_asked + 1
    await safe_reply_text(update, raw)


# -------------------- MAIN TEXT HANDLER --------------------

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    text = (update.message.text or "").strip()
    if not text:
        return

    # перехват режима теста токенов (если включен)
    if await tokens_test_intercept(update, context, text):
        return

    await update.message.chat.send_action("typing")

    mode = get_mode(context)
    chat_id = int(update.effective_chat.id) if update.effective_chat else 0
    temperature = get_temperature(context, chat_id)
    memory_enabled = get_memory_enabled(context, chat_id)
    model = get_model(context, chat_id) or None

    if mode == "tz":
        await handle_tz_message(update, context, text, temperature=temperature, model=model)
        return

    if mode == "forest":
        await handle_forest_message(update, context, text, temperature=temperature, model=model)
        return

    # ---- CHAT MODES (text/thinking/experts/summary) ----
    if mode in ("text", "thinking", "experts", MODE_SUMMARY):
        # Проверка на команды управления сайтом в режиме summary
        if mode == MODE_SUMMARY:
            # Команда "Подними сайт"
            if re.match(r"^(?:подними|поднять|запусти|запустить)\s+сайт$", text, re.IGNORECASE):
                await update.message.chat.send_action("typing")
                result = await site_up_via_mcp()
                # Сохраняем запрос и ответ в БД
                db_add_message(chat_id, mode, "user", text)
                db_add_message(chat_id, mode, "assistant", result)
                # Сжимаем историю
                try:
                    maybe_compress_history(chat_id, temperature=0.0, mode=MODE_SUMMARY)
                except Exception:
                    pass
                await safe_reply_text(update, result)
                return
            
            # Команда "Сделай скрин" или "Сделай скриншот"
            if re.match(r"^(?:сделай|создай|снять)\s+скрин(?:шот)?$", text, re.IGNORECASE):
                await update.message.chat.send_action("typing")
                screenshot_path = await site_screenshot_via_mcp()
                
                # Сохраняем запрос в БД
                db_add_message(chat_id, mode, "user", text)
                
                # Проверяем, что путь к файлу получен
                if screenshot_path and Path(screenshot_path).exists():
                    try:
                        # Отправляем PNG файл в Telegram
                        with open(screenshot_path, "rb") as f:
                            await update.message.reply_document(
                                document=f,
                                filename="site.png",
                                caption="📸 Скриншот сайта"
                            )
                        # Сохраняем ответ в БД
                        db_add_message(chat_id, mode, "assistant", f"Скриншот создан: {screenshot_path}")
                    except Exception as e:
                        logger.exception(f"Failed to send screenshot: {e}")
                        await safe_reply_text(update, f"Скриншот создан, но не удалось отправить: {e}")
                else:
                    # Если файл не найден, отправляем текстовый ответ
                    db_add_message(chat_id, mode, "assistant", screenshot_path)
                    await safe_reply_text(update, screenshot_path)
                
                # Сжимаем историю
                try:
                    maybe_compress_history(chat_id, temperature=0.0, mode=MODE_SUMMARY)
                except Exception:
                    pass
                return
            
            # Команда "Останови сайт"
            if re.match(r"^(?:останови|остановить|выключи|выключить)\s+сайт$", text, re.IGNORECASE):
                await update.message.chat.send_action("typing")
                result = await site_down_via_mcp()
                # Сохраняем запрос и ответ в БД
                db_add_message(chat_id, mode, "user", text)
                db_add_message(chat_id, mode, "assistant", result)
                # Сжимаем историю
                try:
                    maybe_compress_history(chat_id, temperature=0.0, mode=MODE_SUMMARY)
                except Exception:
                    pass
                await safe_reply_text(update, result)
                return
        
        # Проверка на запрос погоды в режиме summary (например: "Погода Москва" или "Погода Самара")
        weather_request_handled = False
        if mode == MODE_SUMMARY:
            # Паттерн: "Погода" + название города (может быть на русском или английском)
            weather_match = re.match(r"^(?:погода|weather)\s+(.+)$", text, re.IGNORECASE)
            if weather_match:
                city = weather_match.group(1).strip()
                if city:
                    # Получаем погоду через MCP и возвращаем результат
                    weather_text = await get_weather_via_mcp(city)
                    # Сохраняем запрос и ответ в БД для истории
                    db_add_message(chat_id, mode, "user", text)
                    db_add_message(chat_id, mode, "assistant", weather_text)
                    
                    # Вызываем сжатие истории (как для обычных сообщений)
                    try:
                        maybe_compress_history(chat_id, temperature=0.0, mode=MODE_SUMMARY)
                    except Exception:
                        pass
                    
                    # Отправляем ответ с погодой
                    await safe_reply_text(update, weather_text)
                    weather_request_handled = True
                    return

        if mode == "thinking":
            system_prompt = SYSTEM_PROMPT_THINKING
        elif mode == "experts":
            system_prompt = SYSTEM_PROMPT_EXPERTS
        else:
            system_prompt = SYSTEM_PROMPT_TEXT

        if memory_enabled:
            # NEW: summary-context builder
            if mode == MODE_SUMMARY:
                messages = build_messages_with_summary(system_prompt, chat_id=chat_id, mode=MODE_SUMMARY)
            else:
                messages = build_messages_with_db_memory(system_prompt, chat_id=chat_id)
        else:
            messages = [{"role": "system", "content": system_prompt}]  # без истории

        messages.append({"role": "user", "content": text})

        # SUMMARY: нужен raw, чтобы взять usage
        if mode == MODE_SUMMARY:
            try:
                data = chat_completion_raw(messages, temperature=temperature, model=model)
                answer = _get_content_from_raw(data)
                pt, ct, tt = _get_usage_tokens(data)
                req_id = str(data.get("id") or "").strip()
            except Exception as e:
                await safe_reply_text(update, f"Ошибка запроса к LLM: {e}")
                return

            answer = (answer or "").strip() or "Пустой ответ от модели."

            # пишем в БД (summary всегда с памятью)
            db_add_message(chat_id, mode, "user", text)
            db_add_message(chat_id, mode, "assistant", answer)

            try:
                maybe_compress_history(chat_id, temperature=0.0, mode=MODE_SUMMARY)
            except Exception:
                pass

            # 1) ответ
            def fmt(x: int | None) -> str:
                return str(x) if isinstance(x, int) else "n/a"

            rid = f", id={req_id}" if req_id else ""
            combined = f"{answer}\n\nТокены: запрос={fmt(pt)}, ответ={fmt(ct)}, всего={fmt(tt)}{rid}"
            await safe_reply_text(update, combined)
            return


        # НЕ summary — как было
        try:
            answer = (chat_completion(messages, temperature=temperature, model=model) or "").strip()
        except Exception as e:
            await safe_reply_text(update, f"Ошибка запроса к LLM: {e}")
            return

        answer = answer or "Пустой ответ от модели."

        # пишем в БД только если память включена
        if memory_enabled:
            db_add_message(chat_id, mode, "user", text)
            db_add_message(chat_id, mode, "assistant", answer)

        await safe_reply_text(update, answer)
        return

    # ---- JSON MODE (без памяти) ----
    raw = ""
    try:
        raw = chat_completion(
            [
                {"role": "system", "content": SYSTEM_PROMPT_JSON},
                {"role": "user", "content": text},
            ],
            temperature=temperature,
            model=model,
        ) or ""

        json_str = extract_json_object(raw)
        data = json.loads(json_str)
        payload = normalize_payload(data)

    except Exception:
        try:
            fixed_raw = repair_json_with_model(SYSTEM_PROMPT_JSON, raw or text, temperature=temperature, model=model)
            json_str = extract_json_object(fixed_raw)
            data = json.loads(json_str)
            payload = normalize_payload(data)
        except Exception as e2:
            err_payload = {
                "title": "Ошибка",
                "time": utc_now_iso(),
                "tag": "error",
                "answer": "Модель вернула непарсируемый формат.",
                "steps": [],
                "warnings": [str(e2)],
                "need_clarification": False,
                "clarifying_question": "",
            }
            await safe_reply_text(update, json.dumps(err_payload, ensure_ascii=False, indent=2))
            return

    context.user_data["last_payload"] = payload
    await safe_reply_text(update, json.dumps(payload, ensure_ascii=False, indent=2))


# -------------------- ERROR HANDLER --------------------

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled error: %s", context.error)
    if isinstance(update, Update) and update.message:
        await safe_reply_text(update, f"Внутренняя ошибка: {type(context.error).__name__}: {context.error}")


# -------------------- BOT COMMANDS MENU --------------------

async def post_init(app: Application) -> None:
    cmds = [
        BotCommand("start", "Старт"),
        BotCommand("help", "Справка"),
        BotCommand("mode_text", f"Режим text + {_short_model_name(OPENROUTER_MODEL)}"),
        BotCommand("mode_json", "JSON на каждое сообщение"),
        BotCommand("mode_summary", f"Режим summary + {_short_model_name(OPENROUTER_MODEL)}"),
        BotCommand("summary_debug", "Показать текущее summary (режим summary)"),
        BotCommand("tz_creation_site", "Собрать ТЗ на сайт (итог JSON)"),
        BotCommand("forest_split", "Кто кому должен (итог текст)"),
        BotCommand("thinking_model", "Решать пошагово"),
        BotCommand("expert_group_model", "Группа экспертов"),
        BotCommand("tokens_test", "Тест токенов (включить режим)"),
        BotCommand("tokens_next", "Тест токенов: следующий этап"),
        BotCommand("tokens_stop", "Тест токенов: сводка и выход"),
        BotCommand("ch_temperature", "Показать/изменить температуру (пример: /ch_temperature 0.7)"),
        BotCommand("ch_memory", "Память ВКЛ/ВЫКЛ (пример: /ch_memory off)"),
        BotCommand("clear_memory", "Очистить память чата"),
        BotCommand("weather_sub", "Подписка на погоду (пример: /weather_sub Москва 30)"),
        BotCommand("weather_sub_stop", "Остановить подписку на погоду (пример: /weather_sub_stop Москва)"),
        BotCommand("digest", "Утренняя сводка: погода + новости (пример: /digest Москва, технологии)"),
    ]

    if MODEL_GLM:
        cmds.append(BotCommand("model_glm", f"Модель: {_short_model_name(MODEL_GLM)}"))
    if MODEL_GEMMA:
        cmds.append(BotCommand("model_gemma", f"Модель: {_short_model_name(MODEL_GEMMA)}"))

    await app.bot.set_my_commands(cmds)


def run() -> None:
    init_db()

    request = HTTPXRequest(
        connect_timeout=20.0,
        read_timeout=60.0,
        write_timeout=60.0,
        pool_timeout=20.0,
    )

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .request(request)
        .post_init(post_init)
        .build()
    )

    # deps для tokens_test.py (чтобы не дублировать логику)
    app.bot_data["tokens_deps"] = {
        "get_temperature": get_temperature,
        "get_model": get_model,
        "get_effective_model": get_effective_model,
        "SYSTEM_PROMPT_TEXT": SYSTEM_PROMPT_TEXT,
        "safe_reply_text": safe_reply_text,
    }

    app.add_error_handler(error_handler)

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))

    app.add_handler(CommandHandler("tokens_test", tokens_test_cmd))
    app.add_handler(CommandHandler("tokens_next", tokens_next_cmd))
    app.add_handler(CommandHandler("tokens_stop", tokens_stop_cmd))

    app.add_handler(CommandHandler("ch_temperature", ch_temperature_cmd))
    app.add_handler(CommandHandler("ch_memory", ch_memory_cmd))
    app.add_handler(CommandHandler("clear_memory", clear_memory_cmd))

    if MODEL_GLM:
        app.add_handler(CommandHandler("model_glm", model_glm_cmd))
    if MODEL_GEMMA:
        app.add_handler(CommandHandler("model_gemma", model_gemma_cmd))

    app.add_handler(CommandHandler("mode_text", mode_text_cmd))
    app.add_handler(CommandHandler("mode_json", mode_json_cmd))
    app.add_handler(CommandHandler("mode_summary", mode_summary_cmd))
    app.add_handler(CommandHandler("summary_debug", summary_debug_cmd))
    app.add_handler(CommandHandler("tz_creation_site", tz_creation_site_cmd))
    app.add_handler(CommandHandler("forest_split", forest_split_cmd))
    app.add_handler(CommandHandler("thinking_model", thinking_model_cmd))
    app.add_handler(CommandHandler("expert_group_model", expert_group_model_cmd))
    app.add_handler(CommandHandler("weather_sub", weather_sub_cmd))
    app.add_handler(CommandHandler("weather_sub_stop", weather_sub_stop_cmd))
    app.add_handler(CommandHandler("digest", digest_cmd))


    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run()
