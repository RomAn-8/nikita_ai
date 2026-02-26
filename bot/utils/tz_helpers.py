"""Helper functions for TZ mode."""

import json
import re
from telegram import Update
from telegram.ext import ContextTypes

from ..core.errors import safe_reply_text
from ..core.prompts import SYSTEM_PROMPT_TZ
from ..services.llm import call_llm
from ..services.database import utc_now_iso


def extract_json_object(text: str) -> str:
    """Извлекает JSON объект из текста."""
    # Ищем JSON объект в тексте
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return text


def normalize_payload(data: dict) -> dict:
    """Нормализует payload для JSON режима."""
    payload = {
        "title": str(data.get("title", "")),
        "time": utc_now_iso(),
        "tag": str(data.get("tag", "tz_site")),
        "answer": str(data.get("answer", "")),
        "steps": list(data.get("steps", [])),
        "warnings": list(data.get("warnings", [])),
        "need_clarification": bool(data.get("need_clarification", False)),
        "clarifying_question": str(data.get("clarifying_question", "")),
    }
    return payload


def repair_json_with_model(system_prompt: str, broken_json: str, temperature: float, model: str | None) -> str:
    """Пытается исправить сломанный JSON через LLM."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Исправь этот JSON и верни только валидный JSON без текста:\n\n{broken_json}"}
    ]
    return (call_llm(messages, temperature=temperature, model=model) or "").strip()


async def send_final_tz_json(update: Update, context: ContextTypes.DEFAULT_TYPE, raw: str, temperature: float, model: str | None) -> None:
    """Отправляет финальный JSON для TZ режима."""
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
