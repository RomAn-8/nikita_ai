import json
import re
from datetime import datetime, timezone

from telegram import Update, BotCommand
from telegram.error import TimedOut
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from telegram.request import HTTPXRequest

from .config import TELEGRAM_BOT_TOKEN
from .openrouter import chat_completion


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------- PROMPTS ----------

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

# Новый режим: "решай пошагово"
SYSTEM_PROMPT_THINKING = """
Ты решаешь задачи в режиме "пошаговое рассуждение".
Правила:
- Решай задачу пошагово.
- В конце дай короткий итоговый ответ отдельной строкой: "ИТОГ: ...".
- Пиши понятно и без воды.
"""

# Новый режим: "группа экспертов"
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


# ---------- HELPERS ----------

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


def repair_json_with_model(system_prompt: str, raw: str) -> str:
    repair_prompt = (
        system_prompt
        + "\n\nИсправь следующий ответ так, чтобы он стал валидным JSON строго по схеме. Верни только JSON."
    )
    fixed = chat_completion([
        {"role": "system", "content": repair_prompt},
        {"role": "user", "content": raw or ""},
    ])
    return fixed


def get_mode(context: ContextTypes.DEFAULT_TYPE) -> str:
    # text | json | tz | forest | thinking | experts
    return context.user_data.get("mode", "text")


async def safe_reply_text(update: Update, text: str) -> None:
    try:
        if update.message:
            await update.message.reply_text(text)
    except TimedOut:
        return


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


# ---------- COMMANDS ----------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    mode = get_mode(context)
    await safe_reply_text(
        update,
        "Привет!\n\n"
        "Команды:\n"
        "/mode_text — обычный текст\n"
        "/mode_json — JSON на каждое сообщение\n"
        "/tz_creation_site — требования для ТЗ (вопросы текстом, итог JSON)\n"
        "/forest_split — кто кому должен (вопросы текстом, итог текстом)\n"
        "/thinking_model — решай пошагово\n"
        "/expert_group_model — группа экспертов\n\n"
        f"Текущий режим: {mode}"
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await safe_reply_text(
        update,
        "Команды:\n"
        "/mode_text — обычный текст\n"
        "/mode_json — JSON на каждое сообщение\n"
        "/tz_creation_site — собрать ТЗ на сайт (в конце JSON)\n"
        "/forest_split — посчитать кто кому должен (в конце текст)\n"
        "/thinking_model — решать пошагово\n"
        "/expert_group_model — решить как группа экспертов\n"
    )


async def mode_text_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["mode"] = "text"
    reset_tz(context)
    reset_forest(context)
    await safe_reply_text(update, "Ок. Режим установлен: text")


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

    first = chat_completion([
        {"role": "system", "content": SYSTEM_PROMPT_TZ},
        {"role": "user", "content": "Начни. Задай первый вопрос, чтобы собрать требования для ТЗ на создание сайта."},
    ]) or ""

    if looks_like_json(first):
        await send_final_tz_json(update, context, first)
        return

    context.user_data["tz_questions"] = 1
    context.user_data["tz_history"].append({"role": "assistant", "content": first.strip()})
    await safe_reply_text(update, first.strip())


async def forest_split_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["mode"] = "forest"
    context.user_data["forest_history"] = []
    context.user_data["forest_questions"] = 0
    context.user_data["forest_done"] = False
    context.user_data.pop("forest_result", None)

    reset_tz(context)

    first = chat_completion([
        {"role": "system", "content": SYSTEM_PROMPT_FOREST},
        {"role": "user", "content": "Начни. Задай первый вопрос для расчёта кто кому сколько должен."},
    ]) or ""

    context.user_data["forest_questions"] = 1
    context.user_data["forest_history"].append({"role": "assistant", "content": first.strip()})
    await safe_reply_text(update, first.strip())


# ---------- TZ FLOW ----------

async def send_final_tz_json(update: Update, context: ContextTypes.DEFAULT_TYPE, raw: str) -> None:
    try:
        json_str = extract_json_object(raw)
        data = json.loads(json_str)
        payload = normalize_payload(data)
    except Exception:
        try:
            fixed_raw = repair_json_with_model(SYSTEM_PROMPT_TZ, raw)
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


async def handle_tz_message(update: Update, context: ContextTypes.DEFAULT_TYPE, user_text: str) -> None:
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
        raw = (chat_completion(messages) or "").strip()
    except Exception as e:
        await safe_reply_text(update, f"Ошибка запроса к LLM: {e}")
        return

    if looks_like_json(raw):
        await send_final_tz_json(update, context, raw)
        return

    history.append({"role": "assistant", "content": raw})
    context.user_data["tz_history"] = history
    context.user_data["tz_questions"] = questions_asked + 1
    await safe_reply_text(update, raw)


# ---------- FOREST FLOW ----------

async def handle_forest_message(update: Update, context: ContextTypes.DEFAULT_TYPE, user_text: str) -> None:
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
        raw = (chat_completion(messages) or "").strip()
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


# ---------- MAIN TEXT HANDLER ----------

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()
    if not text:
        return

    await update.message.chat.send_action("typing")

    mode = get_mode(context)

    if mode == "tz":
        await handle_tz_message(update, context, text)
        return

    if mode == "forest":
        await handle_forest_message(update, context, text)
        return

    if mode == "thinking":
        try:
            answer = chat_completion([
                {"role": "system", "content": SYSTEM_PROMPT_THINKING},
                {"role": "user", "content": text},
            ])
        except Exception as e:
            await safe_reply_text(update, f"Ошибка запроса к LLM: {e}")
            return
        await safe_reply_text(update, (answer or "").strip() or "Пустой ответ от модели.")
        return

    if mode == "experts":
        try:
            answer = chat_completion([
                {"role": "system", "content": SYSTEM_PROMPT_EXPERTS},
                {"role": "user", "content": text},
            ])
        except Exception as e:
            await safe_reply_text(update, f"Ошибка запроса к LLM: {e}")
            return
        await safe_reply_text(update, (answer or "").strip() or "Пустой ответ от модели.")
        return

    if mode == "text":
        try:
            answer = chat_completion([
                {"role": "system", "content": SYSTEM_PROMPT_TEXT},
                {"role": "user", "content": text},
            ])
        except Exception as e:
            await safe_reply_text(update, f"Ошибка запроса к LLM: {e}")
            return

        await safe_reply_text(update, (answer or "").strip() or "Пустой ответ от модели.")
        return

    # mode == "json": JSON на каждое сообщение
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
        try:
            fixed_raw = repair_json_with_model(SYSTEM_PROMPT_JSON, raw or text)
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


# ---------- BOT COMMANDS MENU (slash suggestions) ----------

async def post_init(app: Application) -> None:
    await app.bot.set_my_commands([
        BotCommand("start", "Старт"),
        BotCommand("help", "Справка"),
        BotCommand("mode_text", "Обычный текст"),
        BotCommand("mode_json", "JSON на каждое сообщение"),
        BotCommand("tz_creation_site", "Собрать ТЗ на сайт (итог JSON)"),
        BotCommand("forest_split", "Кто кому должен (итог текст)"),
        BotCommand("thinking_model", "Решать пошагово"),
        BotCommand("expert_group_model", "Группа экспертов"),
    ])


def run() -> None:
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

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("mode_text", mode_text_cmd))
    app.add_handler(CommandHandler("mode_json", mode_json_cmd))
    app.add_handler(CommandHandler("tz_creation_site", tz_creation_site_cmd))
    app.add_handler(CommandHandler("forest_split", forest_split_cmd))
    app.add_handler(CommandHandler("thinking_model", thinking_model_cmd))
    app.add_handler(CommandHandler("expert_group_model", expert_group_model_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run()
