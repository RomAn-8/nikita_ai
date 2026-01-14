import json
import re
from datetime import datetime, timezone

from telegram import Update
from telegram.error import TimedOut
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from telegram.request import HTTPXRequest

from .config import TELEGRAM_BOT_TOKEN
from .openrouter import chat_completion


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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


def extract_json_object(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
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

    normalized["steps"] = [str(x) for x in normalized["steps"] if str(x).strip()]
    normalized["warnings"] = [str(x) for x in normalized["warnings"] if str(x).strip()]

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


def repair_json_with_model(raw: str) -> str:
    repair_prompt = (
        SYSTEM_PROMPT_JSON
        + "\n\nИсправь следующий ответ так, чтобы он стал валидным JSON строго по схеме. Верни только JSON."
    )
    fixed = chat_completion([
        {"role": "system", "content": repair_prompt},
        {"role": "user", "content": raw},
    ])
    return fixed


def get_mode(context: ContextTypes.DEFAULT_TYPE) -> str:
    return context.user_data.get("mode", "text")  # default: text


async def safe_reply_text(update: Update, text: str) -> None:
    try:
        await update.message.reply_text(text)
    except TimedOut:
        return


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    mode = get_mode(context)
    await safe_reply_text(
        update,
        "Привет!\n"
        "Команды:\n"
        "/mode_text — обычный текст\n"
        "/mode_json — JSON-режим (бот отвечает JSON сразу)\n"
        f"Текущий режим: {mode}",
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await safe_reply_text(
        update,
        "Команды:\n"
        "/mode_text — обычный текст\n"
        "/mode_json — JSON-режим\n"
        "Напиши сообщение — отвечу.",
    )


async def mode_json_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["mode"] = "json"
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


async def mode_text_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["mode"] = "text"
    await safe_reply_text(update, "Ок. Режим установлен: text")


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()
    if not text:
        return

    await update.message.chat.send_action("typing")

    mode = get_mode(context)

    if mode == "text":
        try:
            answer = chat_completion([
                {"role": "system", "content": SYSTEM_PROMPT_TEXT},
                {"role": "user", "content": text},
            ])
        except Exception as e:
            await safe_reply_text(update, f"Ошибка запроса к LLM: {e}")
            return

        answer_text = (answer or "").strip() or "Пустой ответ от модели."
        await safe_reply_text(update, answer_text)
        return

    # mode == "json": bot returns JSON immediately
    raw = ""
    try:
        raw = chat_completion([
            {"role": "system", "content": SYSTEM_PROMPT_JSON},
            {"role": "user", "content": text},
        ])

        json_str = extract_json_object(raw)
        data = json.loads(json_str)
        payload = normalize_payload(data)

    except Exception:
        # one repair attempt
        try:
            fixed_raw = repair_json_with_model(raw or text)
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


def run() -> None:
    request = HTTPXRequest(
        connect_timeout=20.0,
        read_timeout=60.0,
        write_timeout=60.0,
        pool_timeout=20.0,
    )

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).request(request).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("mode_json", mode_json_cmd))
    app.add_handler(CommandHandler("mode_text", mode_text_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run()
