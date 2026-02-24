# Тестовый комментарий для PR

import os
import json
import re
import sqlite3
import logging
import requests
from datetime import datetime, timezone
from pathlib import Path

from telegram import Update, BotCommand
from telegram.error import TimedOut, BadRequest
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from telegram.request import HTTPXRequest

from .config import TELEGRAM_BOT_TOKEN, OPENROUTER_API_KEY, OPENROUTER_MODEL, RAG_SIM_THRESHOLD, RAG_TOP_K, EMBEDDING_MODEL, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_TEMPERATURE, OLLAMA_NUM_CTX, OLLAMA_NUM_PREDICT, OLLAMA_SYSTEM_PROMPT, ANALYZE_MODEL, ME_MODEL, USER_PROFILE_PATH
from .openrouter import chat_completion, chat_completion_raw
from .tokens_test import tokens_test_cmd, tokens_next_cmd, tokens_stop_cmd, tokens_test_intercept

# NEW: summary-mode
from .summarizer import MODE_SUMMARY, build_messages_with_summary, maybe_compress_history, clear_summary, summary_debug_cmd
from .mcp_weather import get_weather_via_mcp  # MCP-клиент для получения погоды
from .mcp_news import get_news_via_mcp  # MCP-клиент для получения новостей
from .mcp_docker import site_up_via_mcp, site_screenshot_via_mcp, site_down_via_mcp  # MCP-клиент для управления Docker
from .mcp_client import (
    get_git_branch, get_pr_diff, get_pr_files, get_pr_info,  # MCP-клиент для получения git ветки и PR данных
    user_get, user_register, user_block, user_unblock, user_delete,  # MCP-клиент для работы с пользователями
    reg_create, reg_find_by_user, reg_reschedule, reg_cancel,  # MCP-клиент для работы с записями
    task_create, task_list, task_delete,  # MCP-клиент для работы с задачами
    deploy_check_docker, deploy_upload_image, deploy_load_image, deploy_create_compose, deploy_create_env, deploy_start_bot, deploy_check_container, deploy_stop_bot,  # MCP-клиент для деплоя
)

# Импортируем функции для анализа PR из скрипта
import sys
from pathlib import Path
REVIEW_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "review_pr.py"
if REVIEW_SCRIPT_PATH.exists():
    # Добавляем корень проекта в путь для импорта
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from scripts.review_pr import (
            extract_keywords_from_text,
            get_rag_context as get_rag_context_for_pr,
            format_pr_files,
            create_review_prompt,
        )
        PR_REVIEW_AVAILABLE = True
    except ImportError as e:
        PR_REVIEW_AVAILABLE = False
        logger.warning(f"PR review functions not available: {e}")
else:
    PR_REVIEW_AVAILABLE = False
from .weather_subscription import start_weather_subscription, stop_weather_subscription  # Подписка на погоду
from .embeddings import process_readme_file, process_docs_folder, search_relevant_chunks, has_embeddings, list_indexed_documents, EMBEDDING_MODEL  # Модуль для работы с эмбеддингами


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


# -------------------- USER PROFILE FUNCTIONS --------------------

def load_user_profile() -> dict:
    """Загружает профиль пользователя из JSON файла. Создает базовый профиль, если файл не существует."""
    try:
        if not USER_PROFILE_PATH.exists():
            # Создаем базовый профиль
            default_profile = {
                "name": "",
                "interests": [],
                "communication_style": "",
                "habits": [],
                "preferences": {}
            }
            save_user_profile(default_profile)
            return default_profile
        
        with open(USER_PROFILE_PATH, "r", encoding="utf-8") as f:
            profile = json.load(f)
            # Убеждаемся, что все необходимые поля присутствуют
            default_profile = {
                "name": "",
                "interests": [],
                "communication_style": "",
                "habits": [],
                "preferences": {}
            }
            for key in default_profile:
                if key not in profile:
                    profile[key] = default_profile[key]
            return profile
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing user profile JSON: {e}")
        raise ValueError("Профиль пользователя содержит невалидный JSON. Попробуйте восстановить файл.")
    except Exception as e:
        logger.error(f"Error loading user profile: {e}")
        raise ValueError(f"Ошибка при загрузке профиля: {e}")


def save_user_profile(profile: dict) -> None:
    """Сохраняет профиль пользователя в JSON файл."""
    try:
        # Создаем директорию, если её нет
        USER_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        with open(USER_PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving user profile: {e}")
        raise ValueError(f"Ошибка при сохранении профиля: {e}")


def build_me_system_prompt(profile: dict) -> str:
    """Создает системный промпт для персонального ассистента на основе профиля пользователя."""
    profile_text = json.dumps(profile, ensure_ascii=False, indent=2)
    return f"""Ты — персональный агент пользователя. Вот что ты о нем знаешь:

{profile_text}

Твоя задача — помогать ему, исходя из его привычек и интересов. Отвечай в его любимом стиле общения."""


def update_profile_from_text(text: str) -> dict:
    """Обновляет профиль пользователя, извлекая новые факты из текста через Gemini."""
    try:
        # Загружаем текущий профиль
        current_profile = load_user_profile()
        
        # Создаем промпт для Gemini
        profile_structure = json.dumps({
            "name": "",
            "interests": [],
            "communication_style": "",
            "habits": [],
            "preferences": {}
        }, ensure_ascii=False, indent=2)
        
        update_prompt = f"""Извлеки из этого сообщения новые факты о пользователе и верни обновленный JSON-профиль.

Текущий профиль:
{json.dumps(current_profile, ensure_ascii=False, indent=2)}

Сообщение пользователя:
{text}

ВАЖНО:
1. Сохрани все существующие данные из текущего профиля
2. Добавь новые факты из сообщения
3. Обнови существующие поля, если в сообщении есть более актуальная информация
4. Верни ТОЛЬКО валидный JSON без дополнительных объяснений
5. Структура должна соответствовать этой схеме:
{profile_structure}

Верни только JSON объект."""
        
        messages = [
            {"role": "user", "content": update_prompt}
        ]
        
        # Отправляем запрос в Gemini через OpenRouter
        response = chat_completion(messages, temperature=0.3, model=ME_MODEL)
        
        if not response:
            raise ValueError("Модель не вернула ответ при обновлении профиля")
        
        # Пытаемся извлечь JSON из ответа (может быть обернут в markdown код блоки)
        response_clean = response.strip()
        
        # Удаляем markdown код блоки, если есть
        if response_clean.startswith("```json"):
            response_clean = response_clean[7:]
        elif response_clean.startswith("```"):
            response_clean = response_clean[3:]
        
        if response_clean.endswith("```"):
            response_clean = response_clean[:-3]
        
        response_clean = response_clean.strip()
        
        # Парсим JSON
        try:
            updated_profile = json.loads(response_clean)
            
            # Валидируем структуру
            required_keys = {"name", "interests", "communication_style", "habits", "preferences"}
            if not all(key in updated_profile for key in required_keys):
                raise ValueError("Профиль не содержит все необходимые поля")
            
            # Убеждаемся, что interests и habits - это списки
            if not isinstance(updated_profile.get("interests"), list):
                updated_profile["interests"] = []
            if not isinstance(updated_profile.get("habits"), list):
                updated_profile["habits"] = []
            if not isinstance(updated_profile.get("preferences"), dict):
                updated_profile["preferences"] = {}
            
            return updated_profile
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from Gemini response: {e}")
            logger.error(f"Response was: {response_clean[:500]}")
            raise ValueError("Модель вернула невалидный JSON. Попробуйте еще раз или обновите профиль вручную.")
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error updating profile from text: {e}")
        raise ValueError(f"Ошибка при обновлении профиля: {e}")


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

# Путь к базе данных можно переопределить через переменную окружения
DB_PATH = Path(os.getenv("DB_PATH", str(Path(__file__).resolve().parent / "bot_memory.sqlite3")))
MEMORY_LIMIT_MESSAGES = 30  # сколько последних сообщений хранить в контексте для LLM
MEMORY_CHAT_MODES = ("text", "thinking", "experts", "rag")  # общая память между этими режимами


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


async def safe_reply_text(update: Update, text: str, parse_mode: str | None = None) -> None:
    if not update.message:
        return

    chunks = split_telegram_text(text)
    for ch in chunks:
        try:
            await update.message.reply_text(ch, parse_mode=parse_mode)
        except TimedOut:
            return
        except BadRequest as e:
            msg = str(e).lower()
            if "message is too long" in msg and len(ch) > 500:
                for sub in split_telegram_text(ch, limit=2000):
                    try:
                        await update.message.reply_text(sub, parse_mode=parse_mode)
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
        "Привет! 👋",
        "",
        "📋 Доступные команды:",
        "",
        "🔧 Основные режимы:",
        f"/mode_text — режим text + {_short_model_name(OPENROUTER_MODEL)}",
        "/mode_json — JSON на каждое сообщение",
        f"/mode_summary — режим summary + {_short_model_name(OPENROUTER_MODEL)} (сжатие истории)",
        "/summary_debug — показать текущее summary (режим summary)",
    ]

    if MODEL_GLM:
        lines.append(f"/model_glm — модель {_short_model_name(MODEL_GLM)}")
    if MODEL_GEMMA:
        lines.append(f"/model_gemma — модель {_short_model_name(MODEL_GEMMA)}")
    
    lines.extend([
        "",
        "🤖 Специальные режимы:",
        "/tz_creation_site — собрать ТЗ на сайт (итог JSON)",
        "/forest_split — кто кому должен (итог текст)",
        "/thinking_model — решать пошагово",
        "/expert_group_model — группа экспертов",
        "",
        "⚙️ Настройки:",
        "/ch_temperature — показать/изменить температуру (пример: /ch_temperature 0.7)",
        "/ch_memory — память ВКЛ/ВЫКЛ (пример: /ch_memory off)",
        "/clear_memory — очистить память чата",
        "/clear_embeddings — удалить все эмбеддинги",
        "",
        "🧪 Тестирование:",
        "/tokens_test — тест токенов (включить режим)",
        "/tokens_next — тест токенов: следующий этап",
        "/tokens_stop — тест токенов: сводка и выход",
        "",
        "📚 RAG и эмбеддинги:",
        "/embed_create — создать эмбеддинги из .md файла (сначала отправьте файл)",
        "/embed_docs — создать эмбеддинги из всех файлов в папке docs/",
        "/rag_model — режим RAG",
        "",
        "💬 Словесные команды (в режиме RAG):",
        "• \"RAG+фильтр\" или \"RAG+фильтр <вопрос>\" — поиск с порогом похожести",
        "• \"RAG без фильтра\" или \"RAG без фильтра <вопрос>\" — поиск без порога",
        "• \"Без RAG\" или \"Без RAG <вопрос>\" — обычный ответ без поиска",
        "",
        "🌤️ Погода:",
        "/weather_sub — подписка на погоду (пример: /weather_sub Москва 30)",
        "/weather_sub_stop — остановить подписку (пример: /weather_sub_stop Москва)",
        "/digest — утренняя сводка: погода + новости (пример: /digest Москва, технологии)",
        "",
        "👤 Регистрация и записи:",
        "/register — регистрация (пример: /register Иванов Иван Иванович +79991234567)",
        "/unregister — удалить свою регистрацию",
        "/train_signup — запись на тренировку (пример: /train_signup 15-02-2026 18:00 [примечание])",
        "/train_move — перенос записи (пример: /train_move 1 16-02-2026 19:00)",
        "/train_cancel — отмена записи (пример: /train_cancel 1)",
        "/support — поддержка с RAG (пример: /support можно перенести запись?)",
        "/task_list — режим работы с задачами (словесные команды для создания, просмотра, удаления задач)",
        "",
        "🤖 Локальные модели:",
        "/local_model — режим локальной модели Ollama (переключение режима, затем просто пишите сообщения)",
        "/analyze — анализ JSON файлов с логами через Ollama (отправьте JSON файл, затем задайте вопрос)",
        "/me — персональный ассистент (использует профиль пользователя, команды: 'Обновить профиль', 'Кто я?')",
        "",
        "🚀 Деплой:",
        "/deploy_bot — деплой бота на сервер (требует настройки переменных окружения)",
    ])
    
    if PR_REVIEW_AVAILABLE:
        lines.append("/review_pr — анализ Pull Request (пример: /review_pr 123)")
    
    lines.extend([
        "",
        "📖 Справка:",
        "/help — показать список команд или ответить на вопрос о проекте",
    ])

    lines.extend([
        "",
        f"Текущий режим: {mode}",
        f"Температура: {t}",
        f"Память: {'ВКЛ' if mem else 'ВЫКЛ'}",
        f"Модель: {current_model}",
    ])

    await safe_reply_text(update, "\n".join(lines))


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Команда /help: показывает список команд или отвечает на вопросы о проекте используя RAG.
    
    Использование:
    - /help - показать список команд
    - /help <вопрос> - ответить на вопрос о проекте используя RAG
    """
    if not update.message:
        return
    
    # Если аргументов нет - показываем список команд
    if not context.args:
        lines = [
            "📋 Доступные команды:",
            "",
            "🔧 Основные режимы:",
        f"/mode_text — режим text + {_short_model_name(OPENROUTER_MODEL)}",
        "/mode_json — JSON на каждое сообщение",
        f"/mode_summary — режим summary + {_short_model_name(OPENROUTER_MODEL)} (сжатие истории)",
            "/summary_debug — показать текущее summary (режим summary)",
            "",
            "🤖 Специальные режимы:",
            "/tz_creation_site — собрать ТЗ на сайт (итог JSON)",
            "/forest_split — кто кому должен (итог текст)",
        "/thinking_model — решать пошагово",
            "/expert_group_model — группа экспертов",
            "",
            "⚙️ Настройки:",
            "/ch_temperature — показать/изменить температуру (пример: /ch_temperature 0.7)",
            "/ch_memory — память ВКЛ/ВЫКЛ (пример: /ch_memory off)",
            "/clear_memory — очистить память чата",
            "/clear_embeddings — удалить все эмбеддинги",
            "",
            "🧪 Тестирование:",
            "/tokens_test — тест токенов (включить режим)",
            "/tokens_next — тест токенов: следующий этап",
            "/tokens_stop — тест токенов: сводка и выход",
            "",
            "📚 RAG и эмбеддинги:",
            "/embed_create — создать эмбеддинги из .md файла (сначала отправьте файл)",
            "/embed_docs — создать эмбеддинги из всех файлов в папке docs/",
            "/rag_model — режим RAG",
            "",
            "💬 Словесные команды (в режиме RAG):",
            "• \"RAG+фильтр\" или \"RAG+фильтр <вопрос>\" — поиск с порогом похожести",
            "• \"RAG без фильтра\" или \"RAG без фильтра <вопрос>\" — поиск без порога",
            "• \"Без RAG\" или \"Без RAG <вопрос>\" — обычный ответ без поиска",
            "",
            "🌤️ Погода:",
            "/weather_sub — подписка на погоду (пример: /weather_sub Москва 30)",
            "/weather_sub_stop — остановить подписку (пример: /weather_sub_stop Москва)",
            "/digest — утренняя сводка: погода + новости (пример: /digest Москва, технологии)",
            "",
            "👤 Регистрация и записи:",
            "/register — регистрация (пример: /register Иванов Иван Иванович +79991234567)",
            "/unregister — удалить свою регистрацию",
            "/train_signup — запись на тренировку (пример: /train_signup 15-02-2026 18:00 [примечание])",
            "/train_move — перенос записи (пример: /train_move 1 16-02-2026 19:00)",
            "/train_cancel — отмена записи (пример: /train_cancel 1)",
            "/support — поддержка с RAG (пример: /support можно перенести запись?)",
            "/task_list — режим работы с задачами (словесные команды для создания, просмотра, удаления задач)",
            "/local_model — режим локальной модели Ollama (переключение режима, затем просто пишите сообщения)",
            "/analyze — анализ JSON файлов с логами через Ollama (отправьте JSON файл, затем задайте вопрос)",
            "/me — персональный ассистент (использует профиль пользователя, команды: 'Обновить профиль', 'Кто я?')",
            "",
            "🚀 Деплой:",
            "/deploy_bot — деплой бота на сервер (требует настройки переменных окружения)",
            "/stop_bot — остановить бота на сервере (опции: -v удалить данные, -i удалить образы)",
            "",
            "📖 Справка:",
            "/help <вопрос> — ответить на вопрос о проекте используя RAG",
    ]

        if PR_REVIEW_AVAILABLE:
            lines.insert(-2, "/review_pr — анализ Pull Request (пример: /review_pr 123)")

        if MODEL_GLM:
            lines.insert(4, f"/model_glm — модель {_short_model_name(MODEL_GLM)}")
        if MODEL_GEMMA:
            lines.insert(5 if MODEL_GLM else 4, f"/model_gemma — модель {_short_model_name(MODEL_GEMMA)}")

        await safe_reply_text(update, "\n".join(lines))
        return
    
    # Если есть аргументы - используем RAG для ответа на вопрос
    question_text = " ".join(context.args).strip()
    if not question_text:
        await safe_reply_text(update, "Пожалуйста, задайте вопрос о проекте. Пример: /help Как работает RAG система?")
        return
    
    await update.message.chat.send_action("typing")
    
    chat_id = int(update.effective_chat.id) if update.effective_chat else 0
    temperature = get_temperature(context, chat_id)
    memory_enabled = get_memory_enabled(context, chat_id)
    model = get_model(context, chat_id) or None
    
    # Проверяем наличие эмбеддингов
    if not has_embeddings(EMBEDDING_MODEL):
        await safe_reply_text(
            update,
            "⚠️ Эмбеддинги не найдены в базе данных.\n"
            "Сначала создайте эмбеддинги с помощью команды /embed_create.\n"
            "Отправьте README.md и файлы из папки docs/ для индексации документации."
        )
        return
    
    # Проверяем, является ли вопрос про git ветку
    question_lower = question_text.lower()
    is_git_branch_question = any(keyword in question_lower for keyword in [
        "ветка", "ветку", "ветки", "branch", "git branch", "текущая ветка",
        "какая ветка", "какую ветку", "какие ветки"
    ])
    
    # Получаем текущую ветку git через MCP (опционально)
    git_branch_info = None
    git_branch_name = None
    try:
        git_branch_name = await get_git_branch()
        if git_branch_name:
            git_branch_info = f"Текущая ветка git: {git_branch_name}"
    except Exception as e:
        logger.debug(f"Не удалось получить git ветку через MCP: {e}")
        # Продолжаем без информации о git
    
    # Если вопрос про git ветку и мы получили информацию - отвечаем напрямую
    if is_git_branch_question and git_branch_name:
        await safe_reply_text(update, f"🌿 Текущая ветка git: `{git_branch_name}`")
        return
    elif is_git_branch_question and not git_branch_name:
        await safe_reply_text(
            update,
            "⚠️ Не удалось получить текущую ветку git.\n"
            "Убедитесь, что:\n"
            "- MCP сервер запущен (http://127.0.0.1:8000/mcp)\n"
            "- Текущая директория MCP сервера является git репозиторием"
        )
        return
    
    # Ищем релевантные чанки (сначала с порогом)
    filtered_chunks = []
    try:
        relevant_chunks = search_relevant_chunks(
            question_text,
            model=EMBEDDING_MODEL,
            top_k=RAG_TOP_K,
            min_similarity=RAG_SIM_THRESHOLD,
            apply_threshold=True
        )
        # Фильтруем чанки по порогу
        filtered_chunks = [chunk for chunk in relevant_chunks if chunk["similarity"] >= RAG_SIM_THRESHOLD]
        
        # Если с порогом ничего не найдено, пробуем без порога (для общих вопросов)
        if not filtered_chunks:
            logger.debug(f"No chunks found with threshold {RAG_SIM_THRESHOLD}, trying without threshold")
            relevant_chunks_no_threshold = search_relevant_chunks(
                question_text,
                model=EMBEDDING_MODEL,
                top_k=RAG_TOP_K * 2,  # Берем больше чанков
                min_similarity=0.0,
                apply_threshold=False
            )
            # Берем топ чанки даже с низкой похожестью (но не нулевой)
            filtered_chunks = [chunk for chunk in relevant_chunks_no_threshold if chunk["similarity"] > 0.3]
            
    except Exception as e:
        logger.exception(f"Error searching relevant chunks: {e}")
        await safe_reply_text(update, f"Ошибка при поиске релевантных фрагментов: {e}")
        return
    
    if not filtered_chunks:
        # Получаем список проиндексированных документов для информативного сообщения
        indexed_docs = list_indexed_documents(EMBEDDING_MODEL)
        
        error_msg = "⚠️ Не нашла релевантных фрагментов в документации для ответа на ваш вопрос."
        
        if indexed_docs:
            error_msg += f"\n\nПроиндексированные документы: {', '.join(indexed_docs[:5])}"
            if len(indexed_docs) > 5:
                error_msg += f" и еще {len(indexed_docs) - 5}"
        else:
            error_msg += "\n\nДокументация не проиндексирована. Используйте:\n"
            error_msg += "- `/embed_create` для индексации README.md\n"
            error_msg += "- `/embed_docs` для индексации папки docs/"
        
        error_msg += "\n\nПопробуйте переформулировать вопрос или проиндексировать документацию."
        
        await safe_reply_text(update, error_msg)
        return
    
    # Формируем контекст для LLM
    context_parts = ["Релевантная информация из документации проекта:\n"]
    for i, chunk in enumerate(filtered_chunks, 1):
        context_parts.append(f"[Фрагмент {i} (doc_name={chunk['doc_name']}, chunk_index={chunk['chunk_index']}, score={chunk['similarity']:.4f})]:")
        context_parts.append(chunk["text"])
        context_parts.append("")
    
    context_parts.append(f"Вопрос пользователя о проекте: {question_text}")
    
    # Добавляем информацию о git ветке, если доступна
    if git_branch_info:
        context_parts.append(f"\n{git_branch_info}")
    
    context_parts.append("\nОтветь на вопрос пользователя, используя информацию из документации выше.")
    context_parts.append("Если информация недостаточна, укажи это в ответе.")
    
    user_content = "\n".join(context_parts)
    
    # Формируем сообщения для LLM
    system_prompt = SYSTEM_PROMPT_TEXT
    if memory_enabled:
        messages = build_messages_with_db_memory(system_prompt, chat_id=chat_id)
    else:
        messages = [{"role": "system", "content": system_prompt}]
    
    messages.append({"role": "user", "content": user_content})
    
    # Отправляем запрос к LLM
    try:
        answer = chat_completion(messages, temperature=temperature, model=model)
        answer = (answer or "").strip() or "Пустой ответ от модели."
    except Exception as e:
        await safe_reply_text(update, f"Ошибка запроса к LLM: {e}")
        return
    
    # Сохраняем в БД
    mode = "text"  # Используем режим text для сохранения истории
    db_add_message(chat_id, mode, "user", f"/help {question_text}")
    db_add_message(chat_id, mode, "assistant", answer)
    
    await safe_reply_text(update, answer)


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


async def clear_embeddings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда для удаления всех эмбеддингов из базы данных."""
    try:
        from .embeddings import clear_all_embeddings
        deleted_count = clear_all_embeddings()
        if deleted_count > 0:
            logger.info(f"Cleared {deleted_count} embedding chunks from database")
            await safe_reply_text(update, f"✅ Удалено {deleted_count} эмбеддингов из базы данных.")
        else:
            await safe_reply_text(update, "ℹ️ Эмбеддинги не найдены в базе данных.")
    except Exception as e:
        logger.exception(f"Error clearing embeddings: {e}")
        await safe_reply_text(update, f"❌ Ошибка при удалении эмбеддингов: {e}")


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


# -------------------- EMBEDDINGS COMMAND --------------------

async def embed_create_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Команда для создания эмбеддингов из .md файла.
    Формат: /embed_create
    После вызова команды нужно отправить любой .md файл в чат (как документ).
    """
    if not update.message:
        return
    
    # Устанавливаем флаг ожидания файла
    context.user_data["waiting_for_readme"] = True
    
    await safe_reply_text(
        update,
        "✅ Ожидаю .md файл.\n"
        "Пожалуйста, отправьте любой .md файл в чат (как документ)."
    )


async def embed_docs_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Команда для создания эмбеддингов из всех .md файлов в папке docs/.
    Формат: /embed_docs
    Автоматически находит и индексирует все .md файлы из папки docs/ рекурсивно.
    """
    if not update.message:
        return
    
    await update.message.chat.send_action("typing")
    
    try:
        # Обрабатываем папку docs/
        result = process_docs_folder(replace_existing=True)
        
        if not result["success"]:
            error_msg = result.get("error", "Неизвестная ошибка")
            await safe_reply_text(
                update,
                f"❌ Ошибка при индексации папки docs/: {error_msg}\n"
                f"Обработано файлов: {result.get('files_processed', 0)}/{result.get('total_files', 0)}"
            )
            return
        
        # Формируем ответ со статистикой
        stats = []
        stats.append(f"✅ Эмбеддинги успешно созданы для папки docs/!")
        stats.append(f"📁 Обработано файлов: {result['files_processed']}/{result['total_files']}")
        stats.append(f"📦 Всего чанков: {result['total_chunks']}")
        stats.append("")
        
        # Добавляем информацию о каждом файле
        if result.get("results"):
            stats.append("📄 Обработанные файлы:")
            for file_result in result["results"]:
                if file_result.get("status") == "success":
                    stats.append(f"  ✅ {file_result['file']} ({file_result['chunks']} чанков)")
                else:
                    stats.append(f"  ❌ {file_result['file']}: {file_result.get('error', 'Ошибка')}")
        
        # Добавляем ошибки, если есть
        if result.get("errors"):
            stats.append("")
            stats.append("⚠️ Ошибки:")
            for error in result["errors"][:5]:  # Показываем первые 5 ошибок
                stats.append(f"  - {error}")
            if len(result["errors"]) > 5:
                stats.append(f"  ... и еще {len(result['errors']) - 5} ошибок")
        
        response_text = "\n".join(stats)
        await safe_reply_text(update, response_text)
        
    except Exception as e:
        logger.exception(f"Error in embed_docs_cmd: {e}")
        await safe_reply_text(update, f"❌ Ошибка при индексации папки docs/: {e}")


async def rag_model_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Команда для активации режима RAG.
    В этом режиме доступны 3 подрежима: RAG+фильтр, RAG без фильтра, Без RAG.
    """
    if not update.message:
        return
    
    context.user_data["mode"] = "rag"
    context.user_data["rag_submode"] = "rag_filter"  # Режим по умолчанию
    reset_tz(context)
    reset_forest(context)
    
    await safe_reply_text(
        update,
        "✅ Режим RAG активирован. Доступны 3 режима:\n"
        "- \"RAG+фильтр\" или \"RAG+фильтр <вопрос>\" - поиск с порогом похожести (по умолчанию)\n"
        "- \"RAG без фильтра\" или \"RAG без фильтра <вопрос>\" - поиск без порога\n"
        "- \"Без RAG\" или \"Без RAG <вопрос>\" - обычный ответ без поиска\n\n"
        "После выбора режима можно просто задавать вопросы - режим сохраняется."
    )


async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обработчик документов: обрабатывает .md файлы для создания эмбеддингов и JSON файлы для анализа.
    """
    if not update.message or not update.message.document:
        return
    
    document = update.message.document
    file_name = document.file_name or ""
    
    # Проверяем режим analyze и обработку JSON файлов
    mode = context.user_data.get("mode")
    if mode == "analyze" and file_name.lower().endswith(".json"):
        try:
            # Скачиваем файл
            file = await context.bot.get_file(document.file_id)
            
            # Читаем содержимое
            file_content_bytes = await file.download_as_bytearray()
            file_content = file_content_bytes.decode("utf-8", errors="replace")
            
            # Парсим JSON для валидации
            try:
                json.loads(file_content)
            except json.JSONDecodeError as e:
                await safe_reply_text(update, f"❌ Ошибка: файл не является валидным JSON. {str(e)}")
                return
            
            # Сохраняем содержимое JSON
            context.user_data["analyze_json_content"] = file_content
            
            await safe_reply_text(
                update,
                "Файл получен! Что хочешь узнать? Например: какая ошибка встречается чаще всего?"
            )
        except Exception as e:
            logger.exception(f"Error processing JSON file {file_name}: {e}")
            await safe_reply_text(update, f"❌ Ошибка при обработке файла {file_name}: {e}")
        return
    
    # Проверяем, что это .md файл
    if not file_name.lower().endswith(".md"):
        return  # Игнорируем файлы не .md формата
    
    # Проверяем, ожидается ли файл для embed_create
    waiting_for_readme = context.user_data.get("waiting_for_readme", False)
    
    try:
        # Скачиваем файл
        file = await context.bot.get_file(document.file_id)
        
        # Читаем содержимое
        file_content_bytes = await file.download_as_bytearray()
        file_content = file_content_bytes.decode("utf-8", errors="replace")
        
        # Если ожидается файл для embed_create, обрабатываем его сразу
        if waiting_for_readme:
            # Убираем флаг ожидания
            context.user_data.pop("waiting_for_readme", None)
            
            # Показываем, что обрабатываем
            await update.message.chat.send_action("typing")
            
            # Обрабатываем файл
            result = process_readme_file(
                file_content=file_content,
                doc_name=file_name,  # Используем реальное имя файла
                replace_existing=True,  # Удаляем старые записи и создаем новые
            )
            
            if not result["success"]:
                error_msg = result.get("error", "Неизвестная ошибка")
                await safe_reply_text(update, f"❌ Ошибка при создании эмбеддингов: {error_msg}")
                return
            
            # Формируем ответ со статистикой
            stats = []
            stats.append(f"✅ Эмбеддинги успешно созданы!")
            stats.append(f"📄 Документ: {result['doc_name']}")
            stats.append(f"📊 Символов: {result['text_length']}")
            stats.append(f"📦 Чанков: {result['chunks_count']}")
            stats.append(f"🔢 Размерность эмбеддинга: {result['embedding_dim']}")
            stats.append(f"🤖 Модель: {result['model']}")
            stats.append("")
            stats.append("📝 Превью первого чанка:")
            stats.append(result['first_chunk_preview'])
            stats.append("")
            stats.append("🔢 Первые 10 чисел первого вектора:")
            first_vec_preview = ", ".join([f"{x:.6f}" for x in result['first_embedding_preview']])
            stats.append(first_vec_preview)
            
            response_text = "\n".join(stats)
            await safe_reply_text(update, response_text)
        else:
            # Сохраняем в user_data для возможного использования в будущем
            context.user_data["last_readme_file"] = {
                "file_name": file_name,
                "content": file_content,
                "file_id": document.file_id,
            }
            
            # Уведомляем пользователя
            await safe_reply_text(
                update,
                f"✅ Файл {file_name} получен ({len(file_content)} символов).\n"
                f"Вызовите /embed_create, затем отправьте этот файл для создания эмбеддингов."
            )
    except Exception as e:
        logger.exception(f"Error processing document {file_name}: {e}")
        await safe_reply_text(update, f"❌ Ошибка при обработке файла {file_name}: {e}")


# -------------------- PR REVIEW COMMAND --------------------

async def review_pr_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Команда для анализа Pull Request с использованием RAG и MCP.
    Формат: /review_pr <номер_pr>
    Пример: /review_pr 123
    """
    if not PR_REVIEW_AVAILABLE:
        await safe_reply_text(
            update,
            "❌ Функция анализа PR недоступна. Убедитесь, что скрипт review_pr.py существует."
        )
        return
    
    if not update.message:
        return
    
    # Парсим аргументы
    if not context.args or len(context.args) != 1:
        await safe_reply_text(
            update,
            "Использование: /review_pr <номер_pr>\n"
            "Пример: /review_pr 123\n\n"
            "Убедитесь, что:\n"
            "1. MCP сервер python-sdk запущен (http://127.0.0.1:8000/mcp)\n"
            "2. В переменных окружения установлен GB_TOKEN (или добавьте в .env)"
        )
        return
    
    try:
        pr_number = int(context.args[0])
    except ValueError:
        await safe_reply_text(update, f"❌ Номер PR должен быть числом, получено: {context.args[0]}")
        return
    
    await update.message.chat.send_action("typing")
    
    # Получаем GitHub token из переменных окружения (пробуем GB_TOKEN, затем GITHUB_TOKEN)
    github_token = os.getenv("GB_TOKEN", "").strip() or os.getenv("GITHUB_TOKEN", "").strip()
    if not github_token:
        await safe_reply_text(
            update,
            "❌ GitHub token не найден в переменных окружения.\n"
            "Добавьте GB_TOKEN или GITHUB_TOKEN в .env файл или установите как переменную окружения."
        )
        return
    
    # Параметры репозитория (nikita_ai)
    owner = "RomAn-8"
    repo = "nikita_ai"
    
    try:
        # 1. Получаем данные PR через MCP
        await safe_reply_text(update, f"📥 Получаю данные PR #{pr_number}...")
        try:
            pr_info = await get_pr_info(owner, repo, pr_number, github_token)
        except ValueError as e:
            error_msg = str(e)
            if "404" in error_msg or "не найден" in error_msg.lower():
                await safe_reply_text(update, f"❌ PR #{pr_number} не найден в репозитории {owner}/{repo}.\nПроверьте номер PR.")
            elif "401" in error_msg or "Unauthorized" in error_msg:
                await safe_reply_text(update, f"❌ Ошибка авторизации GitHub.\nПроверьте правильность GB_TOKEN в .env файле.")
            else:
                await safe_reply_text(update, f"❌ Ошибка при получении информации о PR:\n{error_msg}\n\nПроверьте:\n1. MCP сервер запущен (http://127.0.0.1:8000/mcp)\n2. Правильность GB_TOKEN")
            return
        
        try:
            pr_files = await get_pr_files(owner, repo, pr_number, github_token)
        except ValueError as e:
            error_msg = str(e)
            await safe_reply_text(update, f"❌ Ошибка при получении файлов PR:\n{error_msg}\n\nПроверьте:\n1. MCP сервер запущен\n2. Правильность GB_TOKEN\n3. Доступ к репозиторию")
            return
        
        try:
            pr_diff = await get_pr_diff(owner, repo, pr_number, github_token)
        except ValueError as e:
            error_msg = str(e)
            await safe_reply_text(update, f"❌ Ошибка при получении diff PR:\n{error_msg}\n\nПроверьте:\n1. MCP сервер запущен\n2. Правильность GB_TOKEN")
            return
        
        pr_title = pr_info.get("title", "N/A")
        await safe_reply_text(update, f"✅ Получены данные PR: {pr_title}\n📁 Файлов изменено: {len(pr_files)}\n🔍 Ищу релевантную документацию...")
        
        # 2. Получаем RAG контекст
        rag_context = await get_rag_context_for_pr(pr_info, pr_files, pr_diff)
        if rag_context:
            await safe_reply_text(update, "✅ Найдена релевантная документация\n🤖 Генерирую ревью...")
        else:
            await safe_reply_text(update, "⚠️ Релевантная документация не найдена\n🤖 Генерирую ревью...")
        
        # 3. Генерируем ревью через LLM
        messages = create_review_prompt(pr_info, pr_files, pr_diff, rag_context)
        review_text = chat_completion(messages, temperature=0.3, model=OPENROUTER_MODEL)
        
        if not review_text or not review_text.strip():
            await safe_reply_text(update, "❌ LLM вернул пустое ревью.")
            return
        
        # 4. Отправляем результат (разбиваем на части, если слишком длинный)
        max_length = 4000  # Telegram limit
        if len(review_text) <= max_length:
            await safe_reply_text(update, f"📝 **Ревью PR #{pr_number}:**\n\n{review_text}", parse_mode="Markdown")
        else:
            # Отправляем первую часть
            await safe_reply_text(update, f"📝 **Ревью PR #{pr_number}:**\n\n{review_text[:max_length]}...", parse_mode="Markdown")
            # Отправляем остаток
            remaining = review_text[max_length:]
            while remaining:
                chunk = remaining[:max_length]
                remaining = remaining[max_length:]
                await safe_reply_text(update, chunk, parse_mode="Markdown")
        
        await safe_reply_text(update, "✅ Анализ завершен!")
        
    except Exception as e:
        logger.exception(f"Error reviewing PR #{pr_number}: {e}")
        await safe_reply_text(update, f"❌ Ошибка при анализе PR: {e}")


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

    # ---- TASK LIST MODE ----
    if mode == "task_list":
        await handle_task_list_message(update, context, text, temperature=temperature, model=model)
        return

    # ---- LOCAL MODEL MODE (OLLAMA) ----
    if mode == "local_model":
        text_lower = text.lower().strip()
        
        # Обработка словесных команд
        # Изменить температуру
        temp_match = re.search(r'изменить\s+температуру\s+([\d.]+)', text_lower)
        if temp_match:
            try:
                new_temp = float(temp_match.group(1))
                if 0.0 <= new_temp <= 2.0:
                    context.user_data["ollama_temperature"] = new_temp
                    await safe_reply_text(update, f"✅ Температура изменена на {new_temp}")
                else:
                    await safe_reply_text(update, "❌ Температура должна быть в диапазоне от 0.0 до 2.0")
                return
            except ValueError:
                await safe_reply_text(update, "❌ Неверный формат температуры")
                return
        
        # Изменить контекстное окно
        ctx_match = re.search(r'изменить\s+контекстное\s+окно\s+(\d+)', text_lower)
        if ctx_match:
            try:
                new_ctx = int(ctx_match.group(1))
                if new_ctx > 0:
                    context.user_data["ollama_num_ctx"] = new_ctx
                    await safe_reply_text(update, f"✅ Контекстное окно изменено на {new_ctx}")
                else:
                    await safe_reply_text(update, "❌ Контекстное окно должно быть больше 0")
                return
            except ValueError:
                await safe_reply_text(update, "❌ Неверный формат контекстного окна")
                return
        
        # Изменить максимальную длину ответа
        predict_match = re.search(r'изменить\s+максимальную\s+длину\s+ответа\s+(\d+)', text_lower)
        if predict_match:
            try:
                new_predict = int(predict_match.group(1))
                if new_predict > 0:
                    context.user_data["ollama_num_predict"] = new_predict
                    await safe_reply_text(update, f"✅ Максимальная длина ответа изменена на {new_predict}")
                else:
                    await safe_reply_text(update, "❌ Максимальная длина ответа должна быть больше 0")
                return
            except ValueError:
                await safe_reply_text(update, "❌ Неверный формат максимальной длины ответа")
                return
        
        # Показать текущие настройки
        if "показать текущие настройки модели" in text_lower or "показать настройки" in text_lower:
            settings_text = _get_ollama_settings_display(context.user_data)
            await safe_reply_text(update, settings_text)
            return
        
        # Сбросить настройки к значениям по умолчанию
        if "сбросить настройки модели" in text_lower or "сбросить настройки" in text_lower:
            # Удаляем пользовательские настройки
            context.user_data.pop("ollama_temperature", None)
            context.user_data.pop("ollama_num_ctx", None)
            context.user_data.pop("ollama_num_predict", None)
            context.user_data.pop("ollama_system_prompt", None)
            settings_text = _get_ollama_settings_display(context.user_data)
            await safe_reply_text(update, f"✅ Настройки сброшены к значениям по умолчанию:\n\n{settings_text}")
            return
        
        # Если это не команда - отправляем запрос в модель
        try:
            answer = await send_to_ollama(text, context.user_data)
            await safe_reply_text(update, answer)
        except ValueError as e:
            # Ошибки валидации или от модели
            await safe_reply_text(update, f"❌ {str(e)}\n\n💡 Попробуйте сбросить настройки командой: сбросить настройки модели")
        except ConnectionError as e:
            await safe_reply_text(update, f"❌ {str(e)}")
        except Exception as e:
            logger.exception("Error in local_model mode")
            await safe_reply_text(update, f"❌ Ошибка при обработке запроса: {str(e)}")
        return

    # ---- ANALYZE MODE ----
    if mode == "analyze":
        # Проверяем наличие JSON данных
        json_content = context.user_data.get("analyze_json_content")
        if not json_content:
            await safe_reply_text(update, "❌ JSON файл не загружен. Отправь JSON файл с логами для анализа.")
            return
        
        # Отправляем запрос в Ollama для анализа
        try:
            answer = await send_to_ollama_analyze(json_content, text)
            await safe_reply_text(update, answer)
        except ConnectionError as e:
            await safe_reply_text(update, f"❌ {str(e)}")
        except ValueError as e:
            await safe_reply_text(update, f"❌ {str(e)}")
        except Exception as e:
            logger.exception("Error in analyze mode")
            await safe_reply_text(update, f"❌ Ошибка при обработке запроса: {str(e)}")
        return

    # ---- ME MODE (PERSONAL ASSISTANT) ----
    if mode == "me":
        text_lower = text.lower().strip()
        
        # Команда "Обновить профиль [текст]"
        update_profile_match = re.match(r'^обновить\s+профиль\s+(.+)$', text, re.IGNORECASE)
        if update_profile_match:
            update_text = update_profile_match.group(1).strip()
            if not update_text:
                await safe_reply_text(update, "❌ Укажите текст с информацией о себе после команды 'Обновить профиль'")
                return
            
            try:
                await safe_reply_text(update, "⏳ Обновляю профиль...")
                updated_profile = update_profile_from_text(update_text)
                save_user_profile(updated_profile)
                await safe_reply_text(update, "✅ Профиль успешно обновлен!")
            except ValueError as e:
                await safe_reply_text(update, f"❌ {str(e)}")
            except Exception as e:
                logger.exception("Error updating profile")
                await safe_reply_text(update, f"❌ Ошибка при обновлении профиля: {str(e)}")
            return
        
        # Команда "Кто я?"
        if text_lower == "кто я?" or text_lower == "кто я":
            try:
                profile = load_user_profile()
                
                profile_text = "👤 **Ваш профиль:**\n\n"
                
                if profile.get("name"):
                    profile_text += f"**Имя:** {profile['name']}\n"
                
                if profile.get("interests"):
                    interests_str = ", ".join(profile["interests"]) if isinstance(profile["interests"], list) else str(profile["interests"])
                    profile_text += f"**Интересы:** {interests_str}\n"
                
                if profile.get("communication_style"):
                    profile_text += f"**Стиль общения:** {profile['communication_style']}\n"
                
                if profile.get("habits"):
                    habits_str = ", ".join(profile["habits"]) if isinstance(profile["habits"], list) else str(profile["habits"])
                    profile_text += f"**Привычки:** {habits_str}\n"
                
                if profile.get("preferences") and isinstance(profile["preferences"], dict) and profile["preferences"]:
                    prefs_str = ", ".join([f"{k}: {v}" for k, v in profile["preferences"].items()])
                    profile_text += f"**Предпочтения:** {prefs_str}\n"
                
                # Если профиль пустой
                if not any([profile.get("name"), profile.get("interests"), profile.get("communication_style"), 
                           profile.get("habits"), (profile.get("preferences") and profile["preferences"])]):
                    profile_text += "Профиль пока пуст. Используйте команду 'Обновить профиль [текст]' для добавления информации о себе."
                
                await safe_reply_text(update, profile_text)
            except Exception as e:
                logger.exception("Error loading profile for display")
                await safe_reply_text(update, f"❌ Ошибка при загрузке профиля: {str(e)}")
            return
        
        # Обычные сообщения - отправляем в OpenRouter с системным промптом из профиля
        try:
            # Загружаем профиль
            profile = load_user_profile()
            
            # Создаем системный промпт
            system_prompt = build_me_system_prompt(profile)
            
            # Формируем сообщения
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
            
            logger.debug(f"ME mode: sending request to model {ME_MODEL}, messages count: {len(messages)}")
            
            # Отправляем запрос в OpenRouter
            answer = chat_completion(messages, temperature=temperature, model=ME_MODEL)
            
            if not answer:
                await safe_reply_text(update, "❌ Модель не вернула ответ. Попробуйте еще раз.")
                return
            
            await safe_reply_text(update, answer)
        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            if "400" in error_msg or "Bad Request" in error_msg:
                await safe_reply_text(
                    update,
                    f"❌ Ошибка запроса к модели {ME_MODEL}.\n"
                    f"Возможно, модель недоступна или указана неверно.\n"
                    f"Проверьте настройку ME_MODEL в .env файле.\n\n"
                    f"Детали: {error_msg}"
                )
            else:
                await safe_reply_text(update, f"❌ Ошибка API: {error_msg}")
            logger.exception("HTTPError in me mode")
        except ValueError as e:
            await safe_reply_text(update, f"❌ {str(e)}")
        except Exception as e:
            logger.exception("Error in me mode")
            await safe_reply_text(update, f"❌ Ошибка при обработке запроса: {str(e)}")
        return

    # ---- RAG MODE ----
    if mode == "rag":
        # Получаем текущий подрежим или устанавливаем по умолчанию
        rag_submode = context.user_data.get("rag_submode", "rag_filter")
        
        # Проверяем команды переключения режима
        question_text = None
        new_submode = None
        
        # Проверяем "RAG+фильтр" или "RAG фильтр"
        rag_filter_match = re.match(r"^rag\+?фильтр(?:\s+(.+))?$", text, re.IGNORECASE)
        if rag_filter_match:
            new_submode = "rag_filter"
            question_text = rag_filter_match.group(1).strip() if rag_filter_match.group(1) else None
        
        # Проверяем "RAG без фильтра"
        if not new_submode:
            rag_no_filter_match = re.match(r"^rag\s+без\s+фильтра(?:\s+(.+))?$", text, re.IGNORECASE)
            if rag_no_filter_match:
                new_submode = "rag_no_filter"
                question_text = rag_no_filter_match.group(1).strip() if rag_no_filter_match.group(1) else None
        
        # Проверяем "Без RAG"
        if not new_submode:
            no_rag_match = re.match(r"^без\s+rag(?:\s+(.+))?$", text, re.IGNORECASE)
            if no_rag_match:
                new_submode = "no_rag"
                question_text = no_rag_match.group(1).strip() if no_rag_match.group(1) else None
        
        # Если режим переключен, обновляем и подтверждаем
        if new_submode:
            rag_submode = new_submode
            context.user_data["rag_submode"] = rag_submode
            mode_names = {
                "rag_filter": "RAG+фильтр",
                "rag_no_filter": "RAG без фильтра",
                "no_rag": "Без RAG"
            }
            if question_text:
                # Если вопрос указан сразу, продолжаем обработку
                pass
            else:
                # Если только переключение режима, подтверждаем
                await safe_reply_text(update, f"✅ Режим установлен: {mode_names[rag_submode]}. Задайте вопрос.")
                return
        
        # Если вопрос не был извлечен из команды, используем весь текст как вопрос
        if question_text is None:
            question_text = text.strip()
        
        if not question_text:
            await safe_reply_text(
                update,
                "Пожалуйста, задайте вопрос или используйте команды:\n"
                "- \"RAG+фильтр\" или \"RAG+фильтр <вопрос>\"\n"
                "- \"RAG без фильтра\" или \"RAG без фильтра <вопрос>\"\n"
                "- \"Без RAG\" или \"Без RAG <вопрос>\""
            )
            return
        
        # Обработка в зависимости от подрежима
        if rag_submode == "rag_filter":
            # Режим RAG+фильтр
            if not has_embeddings(EMBEDDING_MODEL):
                await safe_reply_text(
                    update,
                    "⚠️ Эмбеддинги не найдены в базе данных.\n"
                    "Сначала создайте эмбеддинги с помощью команды /embed_create."
                )
                return
            
            try:
                relevant_chunks = search_relevant_chunks(
                    question_text,
                    model=EMBEDDING_MODEL,
                    top_k=RAG_TOP_K,
                    min_similarity=RAG_SIM_THRESHOLD,
                    apply_threshold=True
                )
            except Exception as e:
                logger.exception(f"Error searching relevant chunks: {e}")
                await safe_reply_text(update, f"Ошибка при поиске релевантных фрагментов: {e}")
                return
            
            # Фильтруем чанки по порогу (дополнительная проверка)
            filtered_chunks = [chunk for chunk in relevant_chunks if chunk["similarity"] >= RAG_SIM_THRESHOLD]
            
            if not filtered_chunks:
                await safe_reply_text(update, "⚠️ Не нашла релевантных фрагментов.")
                return
            
            # Формируем контекст для LLM
            context_parts = ["Релевантная информация из документов:\n"]
            for i, chunk in enumerate(filtered_chunks, 1):
                context_parts.append(f"[Фрагмент {i} (doc_name={chunk['doc_name']}, chunk_index={chunk['chunk_index']}, score={chunk['similarity']:.4f})]:")
                context_parts.append(chunk["text"])
                context_parts.append("")
            context_parts.append(f"Вопрос пользователя: {question_text}")
            context_parts.append("\nВ конце ответа обязательно укажи список использованных фрагментов документа в формате:")
            context_parts.append("[Фрагмент N: doc_name=..., chunk_index=..., score=...]")
            context_parts.append('Цитата: "точная дословная выдержка из текста фрагмента (1-2 предложения)"')
            context_parts.append("\nВажно:")
            context_parts.append("- Цитата должна быть точной дословной выдержкой из текста фрагмента (не перефразирование)")
            context_parts.append("- Цитата должна быть короткой (1-2 предложения)")
            context_parts.append("- Цитата должна быть наиболее релевантной частью фрагмента для ответа на вопрос")
            context_parts.append("- Каждый использованный фрагмент должен иметь свою цитату")
            user_content = "\n".join(context_parts)
            
            # Формируем сообщения для LLM
            system_prompt = SYSTEM_PROMPT_TEXT
            if memory_enabled:
                messages = build_messages_with_db_memory(system_prompt, chat_id=chat_id)
            else:
                messages = [{"role": "system", "content": system_prompt}]
            
            messages.append({"role": "user", "content": user_content})
            
            # Отправляем запрос к LLM
            try:
                answer = chat_completion(messages, temperature=temperature, model=model)
                answer = (answer or "").strip() or "Пустой ответ от модели."
            except Exception as e:
                await safe_reply_text(update, f"Ошибка запроса к LLM: {e}")
                return
            
            # Сохраняем в БД
            db_add_message(chat_id, mode, "user", text)
            db_add_message(chat_id, mode, "assistant", answer)
            
            await safe_reply_text(update, answer)
            return
        
        elif rag_submode == "rag_no_filter":
            # Режим RAG без фильтра
            if not has_embeddings(EMBEDDING_MODEL):
                await safe_reply_text(
                    update,
                    "⚠️ Эмбеддинги не найдены в базе данных.\n"
                    "Сначала создайте эмбеддинги с помощью команды /embed_create."
                )
                return
            
            try:
                relevant_chunks = search_relevant_chunks(
                    question_text,
                    model=EMBEDDING_MODEL,
                    top_k=RAG_TOP_K,
                    min_similarity=0.0,
                    apply_threshold=False
                )
            except Exception as e:
                logger.exception(f"Error searching relevant chunks: {e}")
                await safe_reply_text(update, f"Ошибка при поиске релевантных фрагментов: {e}")
                return
            
            if not relevant_chunks:
                await safe_reply_text(update, "⚠️ Не нашла релевантных фрагментов.")
                return
            
            # Формируем контекст для LLM
            context_parts = ["Релевантная информация из документов:\n"]
            for i, chunk in enumerate(relevant_chunks, 1):
                context_parts.append(f"[Фрагмент {i} (doc_name={chunk['doc_name']}, chunk_index={chunk['chunk_index']}, score={chunk['similarity']:.4f})]:")
                context_parts.append(chunk["text"])
                context_parts.append("")
            context_parts.append(f"Вопрос пользователя: {question_text}")
            context_parts.append("\nВ конце ответа обязательно укажи список использованных фрагментов документа в формате:")
            context_parts.append("[Фрагмент N: doc_name=..., chunk_index=..., score=...]")
            context_parts.append('Цитата: "точная дословная выдержка из текста фрагмента (1-2 предложения)"')
            context_parts.append("\nВажно:")
            context_parts.append("- Цитата должна быть точной дословной выдержкой из текста фрагмента (не перефразирование)")
            context_parts.append("- Цитата должна быть короткой (1-2 предложения)")
            context_parts.append("- Цитата должна быть наиболее релевантной частью фрагмента для ответа на вопрос")
            context_parts.append("- Каждый использованный фрагмент должен иметь свою цитату")
            user_content = "\n".join(context_parts)
            
            # Формируем сообщения для LLM
            system_prompt = SYSTEM_PROMPT_TEXT
            if memory_enabled:
                messages = build_messages_with_db_memory(system_prompt, chat_id=chat_id)
            else:
                messages = [{"role": "system", "content": system_prompt}]
            
            messages.append({"role": "user", "content": user_content})
            
            # Отправляем запрос к LLM
            try:
                answer = chat_completion(messages, temperature=temperature, model=model)
                answer = (answer or "").strip() or "Пустой ответ от модели."
            except Exception as e:
                await safe_reply_text(update, f"Ошибка запроса к LLM: {e}")
                return
            
            # Сохраняем в БД
            db_add_message(chat_id, mode, "user", text)
            db_add_message(chat_id, mode, "assistant", answer)
            
            await safe_reply_text(update, answer)
            return
        
        elif rag_submode == "no_rag":
            # Режим Без RAG - обычный ответ без поиска
            system_prompt = SYSTEM_PROMPT_TEXT
            if memory_enabled:
                messages = build_messages_with_db_memory(system_prompt, chat_id=chat_id)
            else:
                messages = [{"role": "system", "content": system_prompt}]
            
            messages.append({"role": "user", "content": question_text})
            
            # Отправляем запрос к LLM
            try:
                answer = chat_completion(messages, temperature=temperature, model=model)
                answer = (answer or "").strip() or "Пустой ответ от модели."
            except Exception as e:
                await safe_reply_text(update, f"Ошибка запроса к LLM: {e}")
                return
            
            # Сохраняем в БД
            db_add_message(chat_id, mode, "user", text)
            db_add_message(chat_id, mode, "assistant", answer)
            
            await safe_reply_text(update, answer)
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


# -------------------- GOOGLE SHEETS COMMANDS --------------------

async def register_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /register <ФИО> <телефон>"""
    if not update.message:
        return
    
    if not context.args or len(context.args) < 2:
        await safe_reply_text(update, "Использование: /register <ФИО> <телефон>\nПример: /register Иванов Иван Иванович +79991234567")
        return
    
    username = update.effective_user.username
    if not username:
        await safe_reply_text(update, "❌ Ошибка: у вас не установлен username в Telegram. Пожалуйста, установите username в настройках Telegram и попробуйте снова.")
        return
    
    fio = context.args[0]
    phone = context.args[1]
    
    # Если ФИО состоит из нескольких слов, объединяем их
    if len(context.args) > 2:
        fio = " ".join(context.args[:-1])
        phone = context.args[-1]
    
    try:
        result = await user_register(username, fio, phone)
        if result and result.get("status") == "registered":
            await safe_reply_text(update, "✅ Вы зарегистрированы")
        elif result and result.get("status") == "updated":
            await safe_reply_text(update, "✅ Данные обновлены")
        else:
            await safe_reply_text(update, "❌ Ошибка при регистрации")
    except ValueError as e:
        await safe_reply_text(update, f"❌ {e}")
    except Exception as e:
        logger.exception(f"Error in register_cmd: {e}")
        await safe_reply_text(update, f"❌ Неизвестная ошибка: {e}")


async def unregister_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /unregister - удалить свою регистрацию"""
    if not update.message:
        return
    
    username = update.effective_user.username
    if not username:
        await safe_reply_text(update, "❌ Ошибка: у вас не установлен username в Telegram. Пожалуйста, установите username в настройках Telegram.")
        return
    
    try:
        # Проверяем, есть ли активные записи
        active_regs = []
        try:
            active_regs = await reg_find_by_user(username) or []
        except ValueError:
            pass
        
        if active_regs:
            await safe_reply_text(
                update,
                f"⚠️ У вас есть {len(active_regs)} активных записей. Сначала отмените их командой /train_cancel <reg_id>"
            )
            return
        
        # Удаляем регистрацию
        result = await user_delete(username)
        if result:
            await safe_reply_text(update, "✅ Ваша регистрация удалена")
        else:
            await safe_reply_text(update, "❌ Ошибка при удалении регистрации")
    except ValueError as e:
        await safe_reply_text(update, f"❌ {e}")
    except Exception as e:
        logger.exception(f"Error in unregister_cmd: {e}")
        await safe_reply_text(update, f"❌ Неизвестная ошибка: {e}")


async def train_signup_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /train_signup <дата DD-MM-YYYY> <время HH:MM> [примечание]"""
    if not update.message:
        return
    
    if not context.args or len(context.args) < 2:
        await safe_reply_text(update, "Использование: /train_signup <дата DD-MM-YYYY> <время HH:MM> [примечание]\nПример: /train_signup 15-02-2026 18:00\nПример с примечанием: /train_signup 15-02-2026 10:00 Уличная тренировка кроссфит гиря 16 кг")
        return
    
    username = update.effective_user.username
    if not username:
        await safe_reply_text(update, "❌ Ошибка: у вас не установлен username в Telegram. Пожалуйста, установите username в настройках Telegram.")
        return
    
    date = context.args[0]
    time = context.args[1]
    # Все остальные аргументы после времени - это примечание
    note = " ".join(context.args[2:]) if len(context.args) > 2 else ""
    
    try:
        result = await reg_create(username, date, time, note)
        if result:
            reg_id = result.get("reg_id")
            row_url = result.get("row_url", "")
            response_text = f"✅ Вы записаны на {date} в {time}\nID записи: {reg_id}"
            if note:
                response_text += f"\nПримечание: {note}"
            response_text += f"\nСсылка: {row_url}"
            await safe_reply_text(update, response_text)
        else:
            await safe_reply_text(update, "❌ Ошибка при создании записи")
    except ValueError as e:
        await safe_reply_text(update, f"❌ {e}")
    except Exception as e:
        logger.exception(f"Error in train_signup_cmd: {e}")
        await safe_reply_text(update, f"❌ Неизвестная ошибка: {e}")


async def train_move_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /train_move <reg_id> <дата DD-MM-YYYY> <время HH:MM>"""
    if not update.message:
        return
    
    if not context.args or len(context.args) < 3:
        await safe_reply_text(update, "Использование: /train_move <reg_id> <дата DD-MM-YYYY> <время HH:MM>\nПример: /train_move 1 16-02-2026 19:00")
        return
    
    try:
        reg_id = int(context.args[0])
        new_date = context.args[1]
        new_time = context.args[2]
        
        result = await reg_reschedule(reg_id, new_date, new_time)
        if result:
            row_url = result.get("row_url", "")
            await safe_reply_text(
                update,
                f"✅ Запись {reg_id} перенесена на {new_date} {new_time}\nСсылка: {row_url}"
            )
        else:
            await safe_reply_text(update, "❌ Ошибка при переносе записи")
    except ValueError as e:
        await safe_reply_text(update, f"❌ {e}")
    except Exception as e:
        logger.exception(f"Error in train_move_cmd: {e}")
        await safe_reply_text(update, f"❌ Неизвестная ошибка: {e}")


async def train_cancel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /train_cancel <reg_id>"""
    if not update.message:
        return
    
    if not context.args or len(context.args) < 1:
        await safe_reply_text(update, "Использование: /train_cancel <reg_id>\nПример: /train_cancel 1")
        return
    
    try:
        reg_id = int(context.args[0])
        result = await reg_cancel(reg_id)
        if result:
            await safe_reply_text(update, f"✅ Запись {reg_id} отменена и удалена из системы")
        else:
            await safe_reply_text(update, "❌ Ошибка при отмене записи")
    except ValueError as e:
        await safe_reply_text(update, f"❌ {e}")
    except Exception as e:
        logger.exception(f"Error in train_cancel_cmd: {e}")
        await safe_reply_text(update, f"❌ Неизвестная ошибка: {e}")


async def support_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /support <вопрос> - поддержка с RAG + MCP"""
    if not update.message:
        return
    
    if not context.args:
        await safe_reply_text(update, "Использование: /support <вопрос>\nПример: /support можно перенести запись?")
        return
    
    question = " ".join(context.args)
    username = update.effective_user.username
    if not username:
        await safe_reply_text(update, "❌ Ошибка: у вас не установлен username в Telegram. Пожалуйста, установите username в настройках Telegram.")
        return
    
    try:
        # Получаем данные пользователя через MCP
        user_data = None
        try:
            user_data = await user_get(username)
        except ValueError as e:
            logger.warning(f"Could not get user data: {e}")
        
        # Получаем активные записи через MCP
        active_regs = []
        try:
            active_regs = await reg_find_by_user(username) or []
            if active_regs:
                logger.info(f"Found {len(active_regs)} active registrations for user {username}: {active_regs}")
            else:
                logger.info(f"No active registrations found for user {username}")
        except ValueError as e:
            logger.warning(f"Could not get user registrations: {e}")
        
        # RAG поиск
        rag_chunks = []
        if has_embeddings(EMBEDDING_MODEL):
            try:
                rag_chunks = search_relevant_chunks(
                    question,
                    model=EMBEDDING_MODEL,
                    top_k=RAG_TOP_K,
                    min_similarity=RAG_SIM_THRESHOLD,
                    apply_threshold=True
                )
            except Exception as e:
                logger.exception(f"Error in RAG search: {e}")
        
        # Формируем контекст для LLM
        context_parts = []
        
        # Данные пользователя
        if user_data:
            context_parts.append("Контекст пользователя:")
            context_parts.append(f"- ФИО: {user_data.get('fio', 'не указано')}")
            context_parts.append(f"- Статус: {user_data.get('status', 'неизвестно')}")
            context_parts.append(f"- Дата регистрации: {user_data.get('date_reg', 'не указано')}")
            context_parts.append("")
        
        # Активные записи
        if active_regs:
            context_parts.append("Активные записи:")
            for reg in active_regs:
                context_parts.append(f"- Запись #{reg.get('reg_id')}: {reg.get('date')} {reg.get('time')}, статус: {reg.get('status')}")
            context_parts.append("")
        
        # RAG контекст
        if rag_chunks:
            context_parts.append("Релевантная документация:")
            for i, chunk in enumerate(rag_chunks, 1):
                context_parts.append(f"[Фрагмент {i} (doc_name={chunk['doc_name']}, chunk_index={chunk['chunk_index']}, score={chunk['similarity']:.4f})]:")
                context_parts.append(chunk["text"])
                context_parts.append("")
        
        context_parts.append(f"Вопрос пользователя: {question}")
        context_parts.append("")
        context_parts.append("ВАЖНО: Ответь на вопрос пользователя, используя:")
        context_parts.append("1. Команды из документации (если вопрос о действиях - укажи конкретную команду)")
        context_parts.append("2. Данные пользователя из контекста выше (его активные записи, если есть)")
        context_parts.append("3. Информацию из релевантной документации")
        context_parts.append("")
        context_parts.append("ОСОБОЕ ВНИМАНИЕ:")
        context_parts.append("- Если вопрос о времени тренировки или когда нужно прийти, ВСЕГДА указывай, что нужно приходить за 15 минут до начала тренировки.")
        context_parts.append("  Например: если тренировка в 10:00, нужно прийти к 09:45.")
        context_parts.append("")
        context_parts.append("В конце ответа НЕ указывай:")
        context_parts.append("- Данные регистрации (они будут добавлены автоматически)")
        context_parts.append("- Источники документации (они будут добавлены автоматически)")
        context_parts.append("Просто ответь на вопрос, используя информацию из контекста выше.")
        
        user_content = "\n".join(context_parts)
        
        # Формируем сообщения для LLM
        system_prompt = """Ты помощник поддержки для системы записи на тренировки. 

ВАЖНЫЕ ПРАВИЛА:
1. ВСЕГДА используй команды из документации для ответа на вопросы пользователей
2. Если в документации есть команда (например, /train_move, /train_cancel, /train_signup), ОБЯЗАТЕЛЬНО укажи её в ответе
3. НЕ говори "обратитесь к администратору", если в документации есть способ решить вопрос через команды бота
4. Используй конкретные данные из контекста пользователя (его записи, reg_id, даты, время)
5. Будь конкретным и давай практические инструкции
6. ВАЖНО: Если пользователь спрашивает о времени тренировки или когда нужно прийти, ВСЕГДА указывай, что нужно приходить за 15 минут до начала тренировки. Например, если тренировка в 10:00, нужно прийти к 09:45.

Отвечай на вопросы пользователей, используя предоставленный контекст и команды из документации."""
        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": user_content})
        
        # Отправляем запрос к LLM
        try:
            answer = chat_completion(messages, temperature=0.7, model=OPENROUTER_MODEL)
            answer = (answer or "").strip() or "Пустой ответ от модели."
        except Exception as e:
            await safe_reply_text(update, f"Ошибка запроса к LLM: {e}")
            return
        
        # Формируем финальный ответ с источниками и данными регистрации
        response_parts = [answer]
        
        # Добавляем источники (компактный формат)
        if rag_chunks:
            response_parts.append("")
            response_parts.append("📚 Источники:")
            for chunk in rag_chunks:
                # Берем компактную цитату (до 120 символов, первое предложение)
                chunk_text = chunk["text"]
                # Убираем переносы строк и лишние пробелы
                chunk_text = " ".join(chunk_text.split())
                # Берем первое предложение или первые 120 символов
                sentences = chunk_text.split(". ")
                if sentences:
                    quote = sentences[0]
                    if len(quote) > 120:
                        quote = quote[:120] + "..."
                    elif len(sentences) > 1 and len(quote) < 80:
                        # Если первое предложение короткое, добавляем второе
                        quote = ". ".join(sentences[:2])
                        if len(quote) > 120:
                            quote = quote[:120] + "..."
                    if not quote.endswith(".") and not quote.endswith("..."):
                        quote += "."
                else:
                    quote = chunk_text[:120] + "..." if len(chunk_text) > 120 else chunk_text
                
                # Компактный формат: (doc_name, chunk_index, score, цитата)
                response_parts.append(f"({chunk['doc_name']}, chunk_index={chunk['chunk_index']}, score={chunk['similarity']:.4f}, цитата=\"{quote}\")")
        
        # Добавляем данные регистрации
        if active_regs:
            response_parts.append("")
            response_parts.append("📅 Данные регистрации:")
            for reg in active_regs:
                reg_id = reg.get('reg_id') or 'не указан'
                date = reg.get('date') or 'не указана'
                time = reg.get('time') or 'не указано'
                status = reg.get('status') or 'не указан'
                response_parts.append(f"- Запись #{reg_id}: {date} {time}, статус: {status}")
        elif user_data:
            # Если есть пользователь, но нет записей
            response_parts.append("")
            response_parts.append("📅 Данные регистрации:")
            response_parts.append("- У вас пока нет активных записей. Используйте /train_signup для записи на тренировку.")
        
        final_response = "\n".join(response_parts)
        
        # Отправляем ответ (разбиваем на части, если слишком длинный)
        await safe_reply_text(update, final_response)
        
    except Exception as e:
        logger.exception(f"Error in support_cmd: {e}")
        await safe_reply_text(update, f"❌ Ошибка при обработке запроса поддержки: {e}")


async def task_list_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /task_list - переключение в режим работы с задачами"""
    if not update.message:
        return
    
    context.user_data["mode"] = "task_list"
    reset_tz(context)
    reset_forest(context)
    
    welcome_text = """✅ Режим работы с задачами активирован!

Теперь вы можете отправлять словесные команды для работы с задачами:

📝 Примеры команд:
• "Создай задачу на 15-02-2026 в 10:00 с приоритетом high: Подготовить презентацию"
• "Покажи задачи с приоритетом high"
• "Покажи невыполненные задачи"
• "Удали задачу в строке 5"
• "Покажи задачи с приоритетом high и предложи, что делать первым"

Для выхода из режима используйте команду /cancel или переключитесь на другой режим."""
    
    await safe_reply_text(update, welcome_text)


async def deploy_bot_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /deploy_bot - деплой бота на сервер"""
    if not update.message:
        return
    
    try:
        # Читаем переменные окружения для деплоя
        deploy_ssh_host = os.getenv("DEPLOY_SSH_HOST", "").strip()
        deploy_ssh_port = int(os.getenv("DEPLOY_SSH_PORT", "22"))
        deploy_ssh_username = os.getenv("DEPLOY_SSH_USERNAME", "").strip()
        deploy_ssh_password = os.getenv("DEPLOY_SSH_PASSWORD", "").strip()
        deploy_image_tar_path = os.getenv("DEPLOY_IMAGE_TAR_PATH", "").strip()
        deploy_remote_path = os.getenv("DEPLOY_REMOTE_PATH", "/opt/nikita_ai").strip()
        
        # Переменная для тестового бота (единственная, которая отличается от основного бота)
        deploy_bot_token = os.getenv("DEPLOY_BOT_TOKEN", "").strip()
        
        # Остальные настройки используем из config.py (те же, что и для основного бота)
        deploy_openrouter_api_key = OPENROUTER_API_KEY
        deploy_openrouter_model = OPENROUTER_MODEL
        deploy_embedding_model = EMBEDDING_MODEL
        deploy_rag_sim_threshold = str(RAG_SIM_THRESHOLD)
        deploy_rag_top_k = str(RAG_TOP_K)
        
        # Настройки Ollama для сервера
        deploy_ollama_base_url = "http://127.0.0.1:11434"  # Локальный адрес на сервере
        deploy_ollama_model = OLLAMA_MODEL  # Из config.py (можно задать в локальном .env)
        deploy_ollama_timeout = str(OLLAMA_TIMEOUT)  # Из config.py
        deploy_ollama_temperature = str(OLLAMA_TEMPERATURE)  # Из config.py
        deploy_ollama_num_ctx = str(OLLAMA_NUM_CTX)  # Из config.py
        deploy_ollama_num_predict = str(OLLAMA_NUM_PREDICT)  # Из config.py
        deploy_ollama_system_prompt = OLLAMA_SYSTEM_PROMPT  # Из config.py
        
        # Проверяем наличие обязательных переменных
        missing_vars = []
        if not deploy_ssh_host:
            missing_vars.append("DEPLOY_SSH_HOST")
        if not deploy_ssh_username:
            missing_vars.append("DEPLOY_SSH_USERNAME")
        if not deploy_ssh_password:
            missing_vars.append("DEPLOY_SSH_PASSWORD")
        if not deploy_image_tar_path:
            missing_vars.append("DEPLOY_IMAGE_TAR_PATH")
        if not deploy_bot_token:
            missing_vars.append("DEPLOY_BOT_TOKEN")
        
        if missing_vars:
            await safe_reply_text(
                update,
                f"❌ Отсутствуют обязательные переменные окружения:\n" + "\n".join(f"• {var}" for var in missing_vars)
            )
            return
        
        # Проверяем существование файла образа
        image_path = Path(deploy_image_tar_path)
        if not image_path.exists():
            await safe_reply_text(update, f"❌ Файл образа не найден: {deploy_image_tar_path}")
            return
        
        # Используем фиксированное имя образа (должно совпадать с именем при сохранении в .tar)
        image_name = "nikita_ai"  # Имя образа с подчеркиванием (как в docker save)
        image_tag = "latest"
        
        await safe_reply_text(update, "🚀 Начинаю деплой бота на сервер...")
        
        # 1. Проверка/установка Docker
        await safe_reply_text(update, "📦 Проверяю наличие Docker на сервере...")
        docker_result = await deploy_check_docker(deploy_ssh_host, deploy_ssh_port, deploy_ssh_username, deploy_ssh_password)
        if not docker_result or docker_result.get("status") != "installed":
            error_msg = docker_result.get("message", "Неизвестная ошибка") if docker_result else "Ошибка при проверке Docker"
            await safe_reply_text(update, f"❌ Ошибка при проверке Docker: {error_msg}")
            return
        await safe_reply_text(update, f"✅ {docker_result.get('message', 'Docker готов')}")
        
        # 2. Загрузка образа на сервер
        remote_image_path = f"{deploy_remote_path}/{image_path.name}"
        await safe_reply_text(update, f"📤 Загружаю образ на сервер: {deploy_image_tar_path}...")
        upload_result = await deploy_upload_image(
            deploy_ssh_host, deploy_ssh_port, deploy_ssh_username, deploy_ssh_password,
            deploy_image_tar_path, remote_image_path
        )
        if not upload_result or upload_result.get("status") != "success":
            error_msg = upload_result.get("message", "Неизвестная ошибка") if upload_result else "Ошибка при загрузке образа"
            await safe_reply_text(update, f"❌ Ошибка при загрузке образа: {error_msg}")
            return
        await safe_reply_text(update, f"✅ {upload_result.get('message', 'Образ загружен')}")
        
        # 3. Загрузка образа в Docker
        await safe_reply_text(update, "🐳 Загружаю образ в Docker...")
        load_result = await deploy_load_image(
            deploy_ssh_host, deploy_ssh_port, deploy_ssh_username, deploy_ssh_password,
            remote_image_path
        )
        if not load_result or load_result.get("status") != "success":
            error_msg = load_result.get("message", "Неизвестная ошибка") if load_result else "Ошибка при загрузке образа в Docker"
            await safe_reply_text(update, f"❌ Ошибка при загрузке образа в Docker: {error_msg}")
            return
        await safe_reply_text(update, f"✅ {load_result.get('message', 'Образ загружен в Docker')}")
        
        # 4. Создание docker-compose.yml
        compose_path = f"{deploy_remote_path}/docker-compose.yml"
        compose_content = f"""services:
  bot:
    image: {image_name}:{image_tag}
    container_name: nikita_ai_bot
    restart: unless-stopped
    network_mode: host
    env_file:
      - .env
    environment:
      - DB_PATH=/app/data/bot_memory.sqlite3
    volumes:
      - ./data:/app/data
      - ./digests:/app/bot/digests
    user: "0:0"
"""
        await safe_reply_text(update, "📝 Создаю docker-compose.yml...")
        compose_result = await deploy_create_compose(
            deploy_ssh_host, deploy_ssh_port, deploy_ssh_username, deploy_ssh_password,
            compose_content, compose_path
        )
        if not compose_result or compose_result.get("status") != "success":
            error_msg = compose_result.get("message", "Неизвестная ошибка") if compose_result else "Ошибка при создании docker-compose.yml"
            await safe_reply_text(update, f"❌ Ошибка при создании docker-compose.yml: {error_msg}")
            return
        compose_msg = compose_result.get('message', 'docker-compose.yml создан')
        if compose_result.get('skipped'):
            await safe_reply_text(update, f"⏭️ {compose_msg}")
        else:
            await safe_reply_text(update, f"✅ {compose_msg}")
        
        # 5. Создание .env файла с данными тестового бота
        env_path = f"{deploy_remote_path}/.env"
        env_content = f"""TELEGRAM_BOT_TOKEN={deploy_bot_token}
OPENROUTER_API_KEY={deploy_openrouter_api_key}
OPENROUTER_MODEL={deploy_openrouter_model}
EMBEDDING_MODEL={deploy_embedding_model}
RAG_SIM_THRESHOLD={deploy_rag_sim_threshold}
RAG_TOP_K={deploy_rag_top_k}
OLLAMA_BASE_URL={deploy_ollama_base_url}
OLLAMA_MODEL={deploy_ollama_model}
OLLAMA_TIMEOUT={deploy_ollama_timeout}
OLLAMA_TEMPERATURE={deploy_ollama_temperature}
OLLAMA_NUM_CTX={deploy_ollama_num_ctx}
OLLAMA_NUM_PREDICT={deploy_ollama_num_predict}
OLLAMA_SYSTEM_PROMPT={deploy_ollama_system_prompt}
"""
        await safe_reply_text(update, "📝 Проверяю .env файл...")
        env_result = await deploy_create_env(
            deploy_ssh_host, deploy_ssh_port, deploy_ssh_username, deploy_ssh_password,
            env_content, env_path
        )
        if not env_result or env_result.get("status") != "success":
            error_msg = env_result.get("message", "Неизвестная ошибка") if env_result else "Ошибка при создании .env файла"
            await safe_reply_text(update, f"❌ Ошибка при создании .env файла: {error_msg}")
            return
        env_msg = env_result.get('message', '.env файл создан')
        if env_result.get('skipped'):
            await safe_reply_text(update, f"⏭️ {env_msg}")
        else:
            await safe_reply_text(update, f"✅ {env_msg}")
        
        # 6. Запуск бота
        await safe_reply_text(update, "🚀 Запускаю бота...")
        start_result = await deploy_start_bot(
            deploy_ssh_host, deploy_ssh_port, deploy_ssh_username, deploy_ssh_password,
            compose_path
        )
        if not start_result or start_result.get("status") != "success":
            error_msg = start_result.get("message", "Неизвестная ошибка") if start_result else "Ошибка при запуске бота"
            await safe_reply_text(update, f"❌ Ошибка при запуске бота: {error_msg}")
            return
        
        # Ждем немного, чтобы контейнер успел запуститься
        import asyncio
        await asyncio.sleep(3)
        
        # Проверяем статус контейнера и логи
        await safe_reply_text(update, "🔍 Проверяю статус контейнера...")
        container_result = await deploy_check_container(
            deploy_ssh_host, deploy_ssh_port, deploy_ssh_username, deploy_ssh_password
        )
        
        if container_result:
            container_status = container_result.get("container_status", "неизвестно")
            container_list = container_result.get("container_list", "")
            container_id = container_result.get("container_id", "")
            logs = container_result.get("logs", "")
            # Берем последние 1000 символов логов, чтобы не перегружать сообщение
            logs_preview = logs[-1000:] if len(logs) > 1000 else logs
            
            status_msg = f"✅ Деплой завершен успешно!\n\n"
            status_msg += f"Бот запущен на сервере {deploy_ssh_host}\n"
            status_msg += f"Путь: {deploy_remote_path}\n"
            status_msg += f"Контейнер: nikita_ai_bot\n"
            status_msg += f"Статус: {container_status}\n"
            if container_id:
                status_msg += f"ID: {container_id}\n"
            if container_list:
                status_msg += f"\nВсе контейнеры:\n{container_list}\n"
            status_msg += f"\nПоследние логи:\n```\n{logs_preview}\n```"
            
            await safe_reply_text(update, status_msg)
        else:
            await safe_reply_text(
                update,
                f"✅ Деплой завершен успешно!\n\n"
                f"Бот запущен на сервере {deploy_ssh_host}\n"
                f"Путь: {deploy_remote_path}\n"
                f"Контейнер: nikita_ai_bot\n\n"
                f"⚠️ Не удалось получить логи контейнера. Проверьте вручную: docker logs nikita_ai_bot"
            )
        
    except Exception as e:
        logger.exception(f"Error in deploy_bot_cmd: {e}")
        await safe_reply_text(update, f"❌ Ошибка при деплое: {e}")


async def stop_bot_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /stop_bot - остановка и удаление бота с сервера."""
    if not update.message:
        return

    try:
        # Читаем переменные окружения для деплоя
        deploy_ssh_host = os.getenv("DEPLOY_SSH_HOST", "").strip()
        deploy_ssh_port = int(os.getenv("DEPLOY_SSH_PORT", "22"))
        deploy_ssh_username = os.getenv("DEPLOY_SSH_USERNAME", "").strip()
        deploy_ssh_password = os.getenv("DEPLOY_SSH_PASSWORD", "").strip()
        deploy_remote_path = os.getenv("DEPLOY_REMOTE_PATH", "/opt/nikita_ai").strip()
        
        # Проверяем наличие обязательных переменных
        if not deploy_ssh_host or not deploy_ssh_username or not deploy_ssh_password:
            await safe_reply_text(
                update,
                "❌ Ошибка: Не заданы переменные окружения для деплоя.\n\n"
                "Необходимо задать:\n"
                "- DEPLOY_SSH_HOST\n"
                "- DEPLOY_SSH_USERNAME\n"
                "- DEPLOY_SSH_PASSWORD"
            )
            return
        
        compose_path = f"{deploy_remote_path}/docker-compose.yml"
        
        # Парсим аргументы команды
        args = context.args or []
        remove_volumes = "--remove-volumes" in args or "-v" in args
        remove_images = "--remove-images" in args or "-i" in args
        
        await safe_reply_text(update, "🛑 Останавливаю бота на сервере...")
        
        stop_result = await deploy_stop_bot(
            deploy_ssh_host, deploy_ssh_port, deploy_ssh_username, deploy_ssh_password,
            compose_path, remove_volumes, remove_images
        )
        
        if not stop_result or stop_result.get("status") != "success":
            error_msg = stop_result.get("message", "Неизвестная ошибка") if stop_result else "Ошибка при остановке бота"
            await safe_reply_text(update, f"❌ Ошибка при остановке бота: {error_msg}")
            return
        
        details = stop_result.get("details", [])
        details_text = "\n".join(f"• {d}" for d in details) if details else ""
        
        await safe_reply_text(
            update,
            f"✅ {stop_result.get('message', 'Бот остановлен')}\n\n"
            f"{details_text}\n\n"
            f"Использование:\n"
            f"/stop_bot - остановить контейнер\n"
            f"/stop_bot -v - остановить и удалить данные\n"
            f"/stop_bot -i - остановить и удалить образы\n"
            f"/stop_bot -v -i - полное удаление"
        )
        
    except Exception as e:
        logger.exception(f"Error in stop_bot_cmd: {e}")
        await safe_reply_text(update, f"❌ Ошибка при остановке бота: {e}")


async def handle_task_list_message(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, temperature: float, model: str) -> None:
    """Обработчик сообщений в режиме task_list"""
    if not update.message:
        return
    
    # Проверка на выход из режима
    text_lower = text.lower().strip()
    if text_lower in ["выход", "отмена", "cancel", "/cancel"]:
        context.user_data["mode"] = "text"
        await safe_reply_text(update, "✅ Режим работы с задачами отключен. Возврат в обычный режим.")
        return
    
    # Fallback: попытка распознать простые команды без LLM
    # Удаление задачи: "удали задачу в строке X" или "удали строку X"
    delete_match = re.search(r'(?:удали|удалить|delete).*?(?:задачу|строку|task).*?(?:в|на|номер|#)?\s*(\d+)', text_lower)
    if delete_match:
        try:
            row_num = int(delete_match.group(1))
            result = await task_delete(row_num)
            if result:
                status = result.get("status", "deleted")
                if status == "cleared":
                    await safe_reply_text(update, f"✅ Задача в строке {row_num} очищена (последняя строка данных)")
                else:
                    await safe_reply_text(update, f"✅ Задача в строке {row_num} удалена")
            else:
                await safe_reply_text(update, f"❌ Не удалось удалить задачу в строке {row_num}")
            return
        except Exception as e:
            logger.exception(f"Error in fallback delete: {e}")
            # Продолжаем к обычной обработке через LLM
    
    # Просмотр всех задач: "покажи задачи", "список задач", "задачи"
    if text_lower in ["покажи задачи", "список задач", "задачи", "показать задачи", "list tasks", "show tasks"]:
        try:
            tasks = await task_list() or []
            if not tasks:
                await safe_reply_text(update, "📋 Задач не найдено")
                return
            
            response_parts = ["📋 Список задач:\n"]
            for task in tasks:
                status = "✅" if task.get("completed") else "⏳"
                priority_emoji = {"high": "🔴", "middle": "🟡", "low": "🟢"}.get(task.get("priority", "").lower(), "")
                response_parts.append(f"{status} Строка {task.get('row_number')}: {task.get('date')} {task.get('time')} | {priority_emoji} {task.get('priority', '').upper()} | {task.get('task', '')}")
            
            await safe_reply_text(update, "\n".join(response_parts))
            return
        except Exception as e:
            logger.exception(f"Error in fallback list: {e}")
            # Продолжаем к обычной обработке через LLM
    
    try:
        # RAG поиск
        rag_chunks = []
        if has_embeddings(EMBEDDING_MODEL):
            try:
                rag_chunks = search_relevant_chunks(
                    text,
                    model=EMBEDDING_MODEL,
                    top_k=RAG_TOP_K,
                    min_similarity=RAG_SIM_THRESHOLD,
                    apply_threshold=True
                )
            except Exception as e:
                logger.exception(f"Error in RAG search: {e}")
        
        # Получаем список всех задач для контекста
        all_tasks = []
        try:
            all_tasks = await task_list() or []
        except Exception as e:
            logger.warning(f"Could not get tasks: {e}")
        
        # Формируем контекст для LLM
        context_parts = []
        
        # RAG контекст
        if rag_chunks:
            context_parts.append("Релевантная документация:")
            for i, chunk in enumerate(rag_chunks, 1):
                context_parts.append(f"[Фрагмент {i} (doc_name={chunk['doc_name']}, chunk_index={chunk['chunk_index']}, score={chunk['similarity']:.4f})]:")
                context_parts.append(chunk["text"])
                context_parts.append("")
        
        # Текущие задачи
        if all_tasks:
            context_parts.append("Текущие задачи в системе:")
            for task in all_tasks:
                status = "✅ Выполнена" if task.get("completed") else "⏳ Не выполнена"
                priority_emoji = {"high": "🔴", "middle": "🟡", "low": "🟢"}.get(task.get("priority", "").lower(), "")
                context_parts.append(f"- Строка {task.get('row_number')}: {status} | {task.get('date')} {task.get('time')} | {priority_emoji} {task.get('priority', '').upper()} | {task.get('task', '')}")
            context_parts.append("")
        
        context_parts.append(f"Команда пользователя: {text}")
        context_parts.append("")
        context_parts.append("ВАЖНО: Распознай намерение пользователя и верни JSON с действием:")
        context_parts.append("- Если создание задачи: {\"action\": \"create\", \"date\": \"DD-MM-YYYY\", \"time\": \"HH:MM\", \"task\": \"описание\", \"priority\": \"high|middle|low\"}")
        context_parts.append("- Если просмотр задач: {\"action\": \"list\", \"priority\": \"high|middle|low\" (опционально), \"completed\": true/false (опционально)}")
        context_parts.append("- Если удаление задачи: {\"action\": \"delete\", \"row_number\": число}")
        context_parts.append("- Если запрос рекомендаций: {\"action\": \"recommend\", \"priority\": \"high|middle|low\" (опционально)}")
        context_parts.append("")
        context_parts.append("Если пользователь просит показать задачи и дать рекомендации, используй action: \"recommend\".")
        context_parts.append("Используй информацию из документации для рекомендаций (например, правила клуба о времени прихода).")
        
        user_content = "\n".join(context_parts)
        
        # System prompt для парсинга намерения
        system_prompt = """Ты помощник для работы с задачами. Твоя задача - распознать намерение пользователя из его словесной команды и вернуть JSON с действием и параметрами.

Доступные действия:
1. create - создание задачи (требует: date, time, task, priority)
2. list - просмотр задач (опционально: priority, completed)
3. delete - удаление задачи (требует: row_number)
4. recommend - рекомендации по задачам (опционально: priority)

Верни ТОЛЬКО валидный JSON, без дополнительного текста."""
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": user_content})
        
        # Парсинг намерения
        try:
            intent_response = chat_completion(messages, temperature=0.3, model=model)
            intent_response = (intent_response or "").strip()
            
            # Извлекаем JSON из ответа
            json_match = re.search(r'\{[^}]+\}', intent_response, re.DOTALL)
            if json_match:
                intent_json = json.loads(json_match.group(0))
            else:
                # Пробуем распарсить весь ответ как JSON
                intent_json = json.loads(intent_response)
        except requests.exceptions.HTTPError as e:
            # Ошибка от API (например, 500)
            logger.exception(f"Error from LLM API: {e}")
            error_msg = "❌ Временная ошибка сервиса. Попробуйте повторить запрос через несколько секунд."
            # Если это простая команда на удаление, попробуем fallback
            delete_match = re.search(r'(\d+)', text)
            if delete_match and any(word in text_lower for word in ["удали", "удалить", "delete"]):
                try:
                    row_num = int(delete_match.group(1))
                    result = await task_delete(row_num)
                    if result:
                        status = result.get("status", "deleted")
                        if status == "cleared":
                            await safe_reply_text(update, f"✅ Задача в строке {row_num} очищена (последняя строка данных)")
                        else:
                            await safe_reply_text(update, f"✅ Задача в строке {row_num} удалена")
                        return
                except Exception:
                    pass
            await safe_reply_text(update, error_msg)
            return
        except json.JSONDecodeError as e:
            logger.exception(f"Error parsing JSON from LLM: {e}")
            await safe_reply_text(update, f"❌ Не удалось распознать команду. Попробуйте сформулировать иначе.\nОтвет LLM: {intent_response[:100]}")
            return
        except Exception as e:
            logger.exception(f"Error parsing intent: {e}")
            await safe_reply_text(update, f"❌ Не удалось распознать команду. Попробуйте сформулировать иначе.\nОшибка: {e}")
            return
        
        action = intent_json.get("action", "").lower()
        
        # Выполнение действия
        if action == "create":
            date = intent_json.get("date", "")
            time = intent_json.get("time", "")
            task_desc = intent_json.get("task", "")
            priority = intent_json.get("priority", "middle").lower()
            
            if not date or not time or not task_desc:
                await safe_reply_text(update, "❌ Не указаны обязательные параметры для создания задачи (дата, время, описание)")
                return
            
            try:
                result = await task_create(date, time, task_desc, priority)
                if result:
                    row_url = result.get("row_url", "")
                    response_text = f"✅ Задача создана!\n📅 Дата: {date}\n⏰ Время: {time}\n📝 Задача: {task_desc}\n🎯 Приоритет: {priority.upper()}\nСтрока: {result.get('row_number')}"
                    if row_url:
                        response_text += f"\n🔗 Ссылка: {row_url}"
                    await safe_reply_text(update, response_text)
                else:
                    await safe_reply_text(update, "❌ Ошибка при создании задачи")
            except ValueError as e:
                await safe_reply_text(update, f"❌ {e}")
            except Exception as e:
                logger.exception(f"Error creating task: {e}")
                await safe_reply_text(update, f"❌ Неизвестная ошибка: {e}")
        
        elif action == "list":
            priority_filter = intent_json.get("priority")
            completed_filter = intent_json.get("completed")
            
            try:
                tasks = await task_list(
                    priority=priority_filter,
                    completed=completed_filter
                ) or []
                
                if not tasks:
                    await safe_reply_text(update, "📋 Задач не найдено")
                    return
                
                response_parts = ["📋 Список задач:\n"]
                for task in tasks:
                    status = "✅" if task.get("completed") else "⏳"
                    priority_emoji = {"high": "🔴", "middle": "🟡", "low": "🟢"}.get(task.get("priority", "").lower(), "")
                    response_parts.append(f"{status} Строка {task.get('row_number')}: {task.get('date')} {task.get('time')} | {priority_emoji} {task.get('priority', '').upper()} | {task.get('task', '')}")
                
                await safe_reply_text(update, "\n".join(response_parts))
            except Exception as e:
                logger.exception(f"Error listing tasks: {e}")
                await safe_reply_text(update, f"❌ Ошибка при получении списка задач: {e}")
        
        elif action == "delete":
            row_number = intent_json.get("row_number")
            if not row_number:
                await safe_reply_text(update, "❌ Не указан номер строки для удаления")
                return
            
            try:
                row_num = int(row_number)
                result = await task_delete(row_num)
                if result:
                    status = result.get("status", "deleted")
                    if status == "cleared":
                        await safe_reply_text(update, f"✅ Задача в строке {row_num} очищена (последняя строка данных)")
                    else:
                        await safe_reply_text(update, f"✅ Задача в строке {row_num} удалена")
                else:
                    await safe_reply_text(update, f"❌ Не удалось удалить задачу в строке {row_num}")
            except ValueError as e:
                await safe_reply_text(update, f"❌ {e}")
            except Exception as e:
                logger.exception(f"Error deleting task: {e}")
                await safe_reply_text(update, f"❌ Неизвестная ошибка: {e}")
        
        elif action == "recommend":
            priority_filter = intent_json.get("priority")
            
            try:
                # Получаем задачи для рекомендаций
                tasks = await task_list(priority=priority_filter, completed=False) or []
                
                if not tasks:
                    await safe_reply_text(update, "📋 Нет задач для рекомендаций")
                    return
                
                # Формируем контекст для AI рекомендаций
                tasks_context = []
                for task in tasks:
                    tasks_context.append(f"- Строка {task.get('row_number')}: {task.get('date')} {task.get('time')} | {task.get('priority', '').upper()} | {task.get('task', '')}")
                
                # RAG контекст для рекомендаций
                rag_context = ""
                if rag_chunks:
                    rag_context = "\n\nРелевантная информация из документации:\n"
                    for chunk in rag_chunks[:2]:  # Берем первые 2 чанка
                        rag_context += f"- {chunk['text'][:200]}...\n"
                
                # Дополнительный поиск правил выполнения упражнений, если есть задачи с упражнениями
                exercise_rules_context = ""
                exercise_rules_chunks = []
                exercise_keywords = ["присед", "отжаться", "подтянуться", "пресс", "упражнени", "ноги", "спина", "грудь"]
                has_exercises = any(
                    any(keyword in task.get("task", "").lower() for keyword in exercise_keywords)
                    for task in tasks
                )
                
                if has_exercises and has_embeddings(EMBEDDING_MODEL):
                    try:
                        # Специальный поиск правил выполнения упражнений
                        exercise_rules_chunks = search_relevant_chunks(
                            "правила выполнения упражнений последовательность ноги спина грудь пресс приоритет",
                            model=EMBEDDING_MODEL,
                            top_k=3,
                            min_similarity=0.5,
                            apply_threshold=True
                        )
                        if exercise_rules_chunks:
                            exercise_rules_context = "\n\nПравила выполнения упражнений:\n"
                            for chunk in exercise_rules_chunks:
                                exercise_rules_context += f"- {chunk['text'][:300]}...\n"
                    except Exception as e:
                        logger.warning(f"Error searching exercise rules: {e}")
                
                recommendation_prompt = f"""Проанализируй следующие задачи и дай рекомендации, что делать первым:

Задачи:
{chr(10).join(tasks_context)}
{rag_context}
{exercise_rules_context}

ВАЖНО: Если среди задач есть упражнения (приседания, отжимания, подтягивания, пресс и т.д.), обязательно используй правила выполнения упражнений из документации. Учитывай:
1. Приоритет задачи (HIGH > MIDDLE > LOW) - главный критерий
2. При одинаковом приоритете: упражнения на ноги → упражнения на верх тело (спина/грудь) → упражнения на пресс
3. Подтягивания и отжимания можно выполнять в суперсете

Дай конкретные рекомендации: какие задачи выполнить первыми и почему. Учитывай приоритеты, даты и информацию из документации."""
                
                rec_messages = [
                    {"role": "system", "content": "Ты помощник по планированию задач. Дай конкретные и практичные рекомендации."},
                    {"role": "user", "content": recommendation_prompt}
                ]
                
                recommendation = chat_completion(rec_messages, temperature=0.7, model=model)
                recommendation = (recommendation or "").strip()
                
                response_parts = [recommendation]
                
                # Добавляем источники, если есть RAG чанки
                all_rag_chunks = []
                if rag_chunks:
                    all_rag_chunks.extend(rag_chunks)
                if exercise_rules_chunks:
                    all_rag_chunks.extend(exercise_rules_chunks)
                
                # Убираем дубликаты по doc_name и chunk_index
                seen = set()
                unique_chunks = []
                for chunk in all_rag_chunks:
                    key = (chunk.get("doc_name", ""), chunk.get("chunk_index", -1))
                    if key not in seen:
                        seen.add(key)
                        unique_chunks.append(chunk)
                
                if unique_chunks:
                    response_parts.append("")
                    response_parts.append("📚 Источники:")
                    for chunk in unique_chunks[:3]:  # Показываем максимум 3 источника
                        # Берем компактную цитату (до 120 символов, первое предложение)
                        chunk_text = chunk["text"]
                        # Убираем переносы строк и лишние пробелы
                        chunk_text = " ".join(chunk_text.split())
                        # Берем первое предложение или первые 120 символов
                        sentences = chunk_text.split(". ")
                        if sentences:
                            quote = sentences[0]
                            if len(quote) > 120:
                                quote = quote[:120] + "..."
                        else:
                            quote = chunk_text[:120] + ("..." if len(chunk_text) > 120 else "")
                        
                        doc_name = chunk.get("doc_name", "unknown")
                        # Убираем префикс docs/ если есть
                        if doc_name.startswith("docs/"):
                            doc_name = doc_name[5:]
                        
                        response_parts.append(f"- {doc_name}, chunk_index={chunk.get('chunk_index', 0)}, score={chunk.get('similarity', 0):.4f}")
                        response_parts.append(f"  Цитата: {quote}")
                
                response_parts.append("")
                response_parts.append("📋 Задачи для рассмотрения:")
                for task in tasks[:10]:  # Показываем максимум 10 задач
                    priority_emoji = {"high": "🔴", "middle": "🟡", "low": "🟢"}.get(task.get("priority", "").lower(), "")
                    response_parts.append(f"• Строка {task.get('row_number')}: {task.get('date')} {task.get('time')} | {priority_emoji} {task.get('priority', '').upper()} | {task.get('task', '')}")
                
                await safe_reply_text(update, "\n".join(response_parts))
            except Exception as e:
                logger.exception(f"Error getting recommendations: {e}")
                await safe_reply_text(update, f"❌ Ошибка при получении рекомендаций: {e}")
        
        else:
            await safe_reply_text(update, f"❌ Неизвестное действие: {action}")
    
    except Exception as e:
        logger.exception(f"Error in handle_task_list_message: {e}")
        await safe_reply_text(update, f"❌ Ошибка при обработке команды: {e}")


# -------------------- LOCAL MODEL (OLLAMA) --------------------

async def send_to_ollama(question: str, user_data: dict = None) -> str:
    """Отправляет запрос в Ollama API и возвращает ответ модели."""
    try:
        # Получаем настройки из user_data или используем значения по умолчанию из config
        temperature = float(user_data.get("ollama_temperature", OLLAMA_TEMPERATURE)) if user_data else OLLAMA_TEMPERATURE
        num_ctx = int(user_data.get("ollama_num_ctx", OLLAMA_NUM_CTX)) if user_data else OLLAMA_NUM_CTX
        num_predict = int(user_data.get("ollama_num_predict", OLLAMA_NUM_PREDICT)) if user_data else OLLAMA_NUM_PREDICT
        system_prompt = user_data.get("ollama_system_prompt", OLLAMA_SYSTEM_PROMPT) if user_data else OLLAMA_SYSTEM_PROMPT
        
        # Валидация параметров
        if not (0.0 <= temperature <= 2.0):
            raise ValueError(f"Температура должна быть в диапазоне от 0.0 до 2.0, получено: {temperature}")
        if num_ctx <= 0 or num_ctx > 32768:
            raise ValueError(f"Контекстное окно должно быть от 1 до 32768, получено: {num_ctx}")
        if num_predict <= 0 or num_predict > 8192:
            raise ValueError(f"Максимальная длина ответа должна быть от 1 до 8192, получено: {num_predict}")
        
        # Формируем URL для запроса
        api_url = f"{OLLAMA_BASE_URL}/api/chat"
        
        # Формируем сообщения
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        # Улучшаем пользовательский запрос, добавляя инструкцию о точности
        enhanced_question = question
        # Если вопрос содержит "что такое" или похожие запросы, добавляем контекст
        if any(phrase in question.lower() for phrase in ["что такое", "объясни", "расскажи", "парадокс", "гипотеза"]):
            enhanced_question = f"{question}\n\nВажно: отвечай точно, основываясь на реальных фактах. Если не уверен, скажи об этом."
        messages.append({"role": "user", "content": enhanced_question})
        
        # Формируем payload для Ollama API
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": num_predict
            }
        }
        
        logger.info(f"Sending request to Ollama: {api_url}, model: {OLLAMA_MODEL}, temperature: {temperature}, num_ctx: {num_ctx}, num_predict: {num_predict}")
        logger.debug(f"Ollama payload: {payload}")
        
        # Отправляем POST запрос
        response = requests.post(
            api_url,
            json=payload,
            timeout=OLLAMA_TIMEOUT
        )
        
        logger.debug(f"Ollama response status: {response.status_code}")
        
        # Проверяем статус ответа
        response.raise_for_status()
        
        # Парсим ответ
        data = response.json()
        
        # Проверяем наличие ошибки в ответе
        if "error" in data:
            error_msg = data.get("error", "Неизвестная ошибка")
            logger.error(f"Ollama API error: {error_msg}, full response: {data}")
            raise ValueError(f"Ошибка модели: {error_msg}")
        
        # Извлекаем текст ответа из структуры Ollama
        # Формат ответа: {"message": {"content": "текст ответа"}}
        if "message" in data and "content" in data["message"]:
            answer = data["message"]["content"].strip()
            if answer:
                logger.info(f"Ollama response received, length: {len(answer)}")
                return answer
            else:
                logger.warning(f"Ollama returned empty content, full response: {data}")
                raise ValueError("Модель вернула пустой ответ")
        else:
            logger.warning(f"Unexpected Ollama response structure: {data}")
            raise ValueError("Неожиданный формат ответа от модели")
            
    except requests.exceptions.Timeout:
        logger.exception("Ollama request timeout")
        raise ConnectionError("Локальная модель недоступна (таймаут)")
    except requests.exceptions.ConnectionError:
        logger.exception("Ollama connection error")
        raise ConnectionError("Локальная модель недоступна (ошибка подключения)")
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if hasattr(e, 'response') and e.response else 'unknown'
        error_body = ""
        if hasattr(e, 'response') and e.response:
            try:
                error_body = e.response.text
                logger.error(f"Ollama HTTP error {status_code}: {error_body}")
            except:
                pass
        logger.exception(f"Ollama HTTP error: {status_code}")
        raise ConnectionError(f"Ошибка при обращении к локальной модели (HTTP {status_code})")
    except ValueError as e:
        # Передаем ValueError как есть (это ошибки от модели)
        logger.error(f"Ollama model error: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in send_to_ollama: {type(e).__name__}: {str(e)}")
        raise ConnectionError(f"Неожиданная ошибка при обращении к локальной модели: {str(e)}")


async def send_to_ollama_analyze(json_content: str, question: str) -> str:
    """Отправляет запрос в Ollama API для анализа JSON данных и возвращает ответ модели."""
    try:
        # Формируем URL для запроса
        api_url = f"{OLLAMA_BASE_URL}/api/chat"
        
        # Системный промпт для анализа логов
        system_prompt = "Ты — ассистент для анализа логов. Анализируй предоставленные JSON данные и отвечай на вопросы пользователя. Отвечай точно, кратко и только на русском языке."
        
        # Формируем сообщения
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"JSON данные:\n{json_content}\n\nВопрос: {question}"}
        ]
        
        # Формируем payload для Ollama API
        payload = {
            "model": ANALYZE_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": OLLAMA_TEMPERATURE,
                "num_ctx": OLLAMA_NUM_CTX,
                "num_predict": OLLAMA_NUM_PREDICT
            }
        }
        
        logger.info(f"Sending analyze request to Ollama: {api_url}, model: {ANALYZE_MODEL}")
        logger.debug(f"Ollama analyze payload: {payload}")
        
        # Отправляем POST запрос
        response = requests.post(
            api_url,
            json=payload,
            timeout=OLLAMA_TIMEOUT
        )
        
        logger.debug(f"Ollama analyze response status: {response.status_code}")
        
        # Проверяем статус ответа
        response.raise_for_status()
        
        # Парсим ответ
        data = response.json()
        
        # Проверяем наличие ошибки в ответе
        if "error" in data:
            error_msg = data.get("error", "Неизвестная ошибка")
            logger.error(f"Ollama API error: {error_msg}, full response: {data}")
            raise ValueError(f"Ошибка модели: {error_msg}")
        
        # Извлекаем текст ответа из структуры Ollama
        if "message" in data and "content" in data["message"]:
            answer = data["message"]["content"].strip()
            if answer:
                logger.info(f"Ollama analyze response received, length: {len(answer)}")
                return answer
            else:
                logger.warning(f"Ollama returned empty content, full response: {data}")
                raise ValueError("Модель вернула пустой ответ")
        else:
            logger.warning(f"Unexpected Ollama response structure: {data}")
            raise ValueError("Неожиданный формат ответа от модели")
            
    except requests.exceptions.Timeout:
        logger.exception("Ollama analyze request timeout")
        raise ConnectionError("Локальная модель недоступна (таймаут)")
    except requests.exceptions.ConnectionError:
        logger.exception("Ollama analyze connection error")
        raise ConnectionError("Локальная модель недоступна (ошибка подключения)")
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if hasattr(e, 'response') and e.response else 'unknown'
        error_body = ""
        if hasattr(e, 'response') and e.response:
            try:
                error_body = e.response.text
                logger.error(f"Ollama HTTP error {status_code}: {error_body}")
            except:
                pass
        logger.exception(f"Ollama HTTP error: {status_code}")
        raise ConnectionError(f"Ошибка при обращении к локальной модели (HTTP {status_code})")
    except ValueError as e:
        logger.error(f"Ollama model error: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in send_to_ollama_analyze: {type(e).__name__}: {str(e)}")
        raise ConnectionError(f"Неожиданная ошибка при обращении к локальной модели: {str(e)}")


def _get_ollama_settings_display(user_data: dict = None) -> str:
    """Формирует строку с текущими настройками модели."""
    temperature = float(user_data.get("ollama_temperature", OLLAMA_TEMPERATURE)) if user_data else OLLAMA_TEMPERATURE
    num_ctx = int(user_data.get("ollama_num_ctx", OLLAMA_NUM_CTX)) if user_data else OLLAMA_NUM_CTX
    num_predict = int(user_data.get("ollama_num_predict", OLLAMA_NUM_PREDICT)) if user_data else OLLAMA_NUM_PREDICT
    system_prompt = user_data.get("ollama_system_prompt", OLLAMA_SYSTEM_PROMPT) if user_data else OLLAMA_SYSTEM_PROMPT
    
    return (
        f"📊 Текущие настройки модели:\n"
        f"• Температура: {temperature}\n"
        f"• Контекстное окно: {num_ctx}\n"
        f"• Максимальная длина ответа: {num_predict}\n"
        f"• Системный промпт: {system_prompt[:50]}{'...' if len(system_prompt) > 50 else ''}"
    )


async def local_model_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /local_model - переключение в режим локальной модели Ollama или отправка запроса"""
    if not update.message:
        return
    
    # Если аргументов нет - переключаем режим
    if not context.args:
        chat_id = int(update.effective_chat.id) if update.effective_chat else 0
        context.user_data["mode"] = "local_model"
        reset_tz(context)
        reset_forest(context)
        
        settings_text = _get_ollama_settings_display(context.user_data)
        
        await safe_reply_text(
            update,
            f"✅ Режим локальной модели Ollama активирован.\n"
            f"Модель: {OLLAMA_MODEL}\n\n"
            f"{settings_text}\n\n"
            f"Теперь все ваши сообщения будут обрабатываться через локальную модель.\n"
            f"Для выхода из режима используйте /mode_text или другой режим.\n\n"
            f"💡 Команды для изменения настроек:\n"
            f"• \"изменить температуру 0.7\"\n"
            f"• \"изменить контекстное окно 4096\"\n"
            f"• \"изменить максимальную длину ответа 512\"\n"
            f"• \"показать текущие настройки модели\"\n"
            f"• \"сбросить настройки модели\""
        )
        return
    
    # Получаем текст команды
    text = " ".join(context.args).strip().lower()
    
    # Обработка словесных команд
    # Изменить температуру
    temp_match = re.search(r'изменить\s+температуру\s+([\d.]+)', text)
    if temp_match:
        try:
            new_temp = float(temp_match.group(1))
            if 0.0 <= new_temp <= 2.0:
                context.user_data["ollama_temperature"] = new_temp
                await safe_reply_text(update, f"✅ Температура изменена на {new_temp}")
            else:
                await safe_reply_text(update, "❌ Температура должна быть в диапазоне от 0.0 до 2.0")
            return
        except ValueError:
            await safe_reply_text(update, "❌ Неверный формат температуры")
            return
    
    # Изменить контекстное окно
    ctx_match = re.search(r'изменить\s+контекстное\s+окно\s+(\d+)', text)
    if ctx_match:
        try:
            new_ctx = int(ctx_match.group(1))
            if new_ctx > 0:
                context.user_data["ollama_num_ctx"] = new_ctx
                await safe_reply_text(update, f"✅ Контекстное окно изменено на {new_ctx}")
            else:
                await safe_reply_text(update, "❌ Контекстное окно должно быть больше 0")
            return
        except ValueError:
            await safe_reply_text(update, "❌ Неверный формат контекстного окна")
            return
    
    # Изменить максимальную длину ответа
    predict_match = re.search(r'изменить\s+максимальную\s+длину\s+ответа\s+(\d+)', text)
    if predict_match:
        try:
            new_predict = int(predict_match.group(1))
            if new_predict > 0:
                context.user_data["ollama_num_predict"] = new_predict
                await safe_reply_text(update, f"✅ Максимальная длина ответа изменена на {new_predict}")
            else:
                await safe_reply_text(update, "❌ Максимальная длина ответа должна быть больше 0")
            return
        except ValueError:
            await safe_reply_text(update, "❌ Неверный формат максимальной длины ответа")
            return
    
    # Показать текущие настройки
    if "показать текущие настройки модели" in text or "показать настройки" in text:
        settings_text = _get_ollama_settings_display(context.user_data)
        await safe_reply_text(update, settings_text)
        return
    
    # Сбросить настройки к значениям по умолчанию
    if "сбросить настройки модели" in text or "сбросить настройки" in text:
        # Удаляем пользовательские настройки
        context.user_data.pop("ollama_temperature", None)
        context.user_data.pop("ollama_num_ctx", None)
        context.user_data.pop("ollama_num_predict", None)
        context.user_data.pop("ollama_system_prompt", None)
        settings_text = _get_ollama_settings_display(context.user_data)
        await safe_reply_text(update, f"✅ Настройки сброшены к значениям по умолчанию:\n\n{settings_text}")
        return
    
    # Если это не команда - отправляем запрос в модель
    question = " ".join(context.args)
    
    try:
        answer = await send_to_ollama(question, context.user_data)
        await safe_reply_text(update, answer)
    except ValueError as e:
        # Ошибки валидации или от модели
        await safe_reply_text(update, f"❌ {str(e)}\n\n💡 Попробуйте сбросить настройки командой: сбросить настройки модели")
    except ConnectionError as e:
        await safe_reply_text(update, f"❌ {str(e)}")
    except Exception as e:
        await safe_reply_text(update, f"❌ Ошибка при обработке запроса: {str(e)}")


# -------------------- ANALYZE COMMAND --------------------

async def analyze_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /analyze - анализ JSON файлов с логами через Ollama"""
    if not update.message:
        return
    
    # Устанавливаем режим analyze
    context.user_data["mode"] = "analyze"
    # Очищаем предыдущие данные анализа
    context.user_data.pop("analyze_json_content", None)
    
    await safe_reply_text(update, "Отправь JSON файл с логами для анализа")


# -------------------- ME COMMAND (PERSONAL ASSISTANT) --------------------

async def me_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /me - переключение в режим персонального ассистента"""
    if not update.message:
        return
    
    # Переключаем режим на "me"
    context.user_data["mode"] = "me"
    reset_tz(context)
    reset_forest(context)
    
    # Загружаем профиль для проверки
    try:
        profile = load_user_profile()
        profile_info = ""
        if profile.get("name"):
            profile_info = f"\n👤 Имя: {profile['name']}"
        if profile.get("interests"):
            profile_info += f"\n🎯 Интересы: {', '.join(profile['interests'][:3])}"
            if len(profile['interests']) > 3:
                profile_info += "..."
    except Exception as e:
        logger.warning(f"Error loading profile in me_cmd: {e}")
        profile_info = "\n⚠️ Профиль не загружен. Используйте команду 'Обновить профиль' для создания профиля."
    
    await safe_reply_text(
        update,
        f"✅ Режим персонального ассистента активирован.\n"
        f"Модель: {ME_MODEL}\n"
        f"{profile_info}\n\n"
        f"Теперь все ваши сообщения будут обрабатываться через персонального ассистента.\n"
        f"Для выхода из режима используйте /mode_text или другой режим.\n\n"
        f"💡 Команды:\n"
        f"• \"Обновить профиль [текст]\" - обновить информацию о себе\n"
        f"• \"Кто я?\" - показать текущий профиль\n"
        f"• Обычные сообщения - общение с персональным ассистентом"
    )


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
        BotCommand("clear_embeddings", "Удалить все эмбеддинги"),
        BotCommand("weather_sub", "Подписка на погоду (пример: /weather_sub Москва 30)"),
        BotCommand("weather_sub_stop", "Остановить подписку на погоду (пример: /weather_sub_stop Москва)"),
        BotCommand("digest", "Утренняя сводка: погода + новости (пример: /digest Москва, технологии)"),
        BotCommand("embed_create", "Создать эмбеддинги из .md файла (сначала отправьте файл)"),
        BotCommand("embed_docs", "Создать эмбеддинги из всех файлов в папке docs/"),
        BotCommand("rag_model", "Режим RAG (используйте \"Ответь с RAG\" или \"Ответь без RAG\")"),
        BotCommand("register", "Регистрация (пример: /register Иванов Иван Иванович +79991234567)"),
        BotCommand("unregister", "Удалить свою регистрацию"),
        BotCommand("train_signup", "Запись на тренировку (пример: /train_signup 15-02-2026 18:00 [примечание])"),
        BotCommand("train_move", "Перенос записи (пример: /train_move 1 16-02-2026 19:00)"),
        BotCommand("train_cancel", "Отмена записи (пример: /train_cancel 1)"),
        BotCommand("support", "Поддержка с RAG (пример: /support можно перенести запись?)"),
        BotCommand("task_list", "Режим работы с задачами"),
        BotCommand("local_model", f"Режим локальной модели Ollama (переключение режима)"),
        BotCommand("analyze", "Анализ JSON логов через Ollama"),
        BotCommand("me", "Персональный ассистент"),
    ]
    
    if PR_REVIEW_AVAILABLE:
        cmds.append(BotCommand("review_pr", "Анализ Pull Request (пример: /review_pr 123)"))

    if MODEL_GLM:
        cmds.append(BotCommand("model_glm", f"Модель: {_short_model_name(MODEL_GLM)}"))
    if MODEL_GEMMA:
        cmds.append(BotCommand("model_gemma", f"Модель: {_short_model_name(MODEL_GEMMA)}"))

    await app.bot.set_my_commands(cmds)


def run() -> None:
    # Подавляем избыточные логи httpx (HTTP запросы к Telegram API)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    init_db()
    # Инициализируем таблицу для эмбеддингов
    from .embeddings import init_embeddings_table
    init_embeddings_table()

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
    app.add_handler(CommandHandler("clear_embeddings", clear_embeddings_cmd))

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
    if PR_REVIEW_AVAILABLE:
        app.add_handler(CommandHandler("review_pr", review_pr_cmd))
    app.add_handler(CommandHandler("embed_create", embed_create_cmd))
    app.add_handler(CommandHandler("embed_docs", embed_docs_cmd))
    app.add_handler(CommandHandler("rag_model", rag_model_cmd))
    app.add_handler(CommandHandler("register", register_cmd))
    app.add_handler(CommandHandler("unregister", unregister_cmd))
    app.add_handler(CommandHandler("train_signup", train_signup_cmd))
    app.add_handler(CommandHandler("train_move", train_move_cmd))
    app.add_handler(CommandHandler("train_cancel", train_cancel_cmd))
    app.add_handler(CommandHandler("support", support_cmd))
    app.add_handler(CommandHandler("task_list", task_list_cmd))
    app.add_handler(CommandHandler("deploy_bot", deploy_bot_cmd))
    app.add_handler(CommandHandler("stop_bot", stop_bot_cmd))
    app.add_handler(CommandHandler("local_model", local_model_cmd))
    app.add_handler(CommandHandler("analyze", analyze_cmd))
    app.add_handler(CommandHandler("me", me_cmd))

    app.add_handler(MessageHandler(filters.Document.ALL, on_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run()
