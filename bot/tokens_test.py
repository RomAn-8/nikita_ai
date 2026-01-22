import os
import time
import requests
from telegram import Update
from telegram.ext import ContextTypes

from .config import OPENROUTER_API_KEY
from .openrouter import chat_completion_raw

TOKENS_TEST_KEY = "tokens_test_state"

STAGES = [
    ("short", "КОРОТКИЙ"),
    ("long", "ДЛИННЫЙ"),
    ("over", "ПЕРЕЛИМИТ"),
]

# -------------------- OVERLIMIT SETTINGS (не ломает старое) --------------------
# Чтобы "ПЕРЕЛИМИТ" реально тестировал ПЕРЕПОЛНЕНИЕ КОНТЕКСТА (вход), а не отказ модели по смыслу:
# 1) Лучше выбрать модель с маленьким контекстом и указать её тут:
#    OPENROUTER_MODEL_OVERLIMIT=mistralai/mistral-7b-instruct-v0.1
# 2) Размер раздувания входа (слов):
#    TOKENS_OVERLIMIT_WORDS=6000
#
# Важно: на некоторых моделях OpenRouter может "ужимать" промпт трансформами (middle-out),
# поэтому для этапа ПЕРЕЛИМИТ мы отправляем transforms=[] (отключаем).
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
ENV_OVERLIMIT_MODEL = "OPENROUTER_MODEL_OVERLIMIT"
ENV_OVERLIMIT_WORDS = "TOKENS_OVERLIMIT_WORDS"
DEFAULT_OVERLIMIT_WORDS = 6000
OVERLIMIT_MAX_TOKENS = 16  # чтобы ответ был коротким, если вдруг влезло


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }


def _chat_completion_raw_overlimit(
    messages,
    temperature: float,
    model: str,
    timeout: int = 60,
) -> dict:
    """
    Отдельный вызов ТОЛЬКО для этапа ПЕРЕЛИМИТ:
    - transforms=[] чтобы не было "авто-ужатия" промпта
    - max_tokens маленький
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(OVERLIMIT_MAX_TOKENS),
        "transforms": [],  # КЛЮЧ: отключаем transforms
    }

    r = requests.post(OPENROUTER_CHAT_URL, headers=_headers(), json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _get_usage(data: dict) -> tuple[int | None, int | None, int | None]:
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


def _get_finish_reason(data: dict) -> str:
    try:
        choice0 = (data.get("choices") or [{}])[0]
        fr = choice0.get("finish_reason")
        return str(fr) if fr is not None else ""
    except Exception:
        return ""


def _get_content(data: dict) -> str:
    try:
        choice0 = (data.get("choices") or [{}])[0]
        return ((choice0.get("message") or {}).get("content") or "").strip()
    except Exception:
        return ""


def _format_http_error(e: Exception) -> str:
    if isinstance(e, requests.HTTPError) and getattr(e, "response", None) is not None:
        r = e.response
        code = getattr(r, "status_code", None)
        try:
            j = r.json()
            # OpenRouter часто возвращает {"error": {"message": "...", "code": ...}}
            if isinstance(j, dict) and "error" in j:
                err = j.get("error") or {}
                msg = err.get("message") or j.get("message") or str(j)
                return f"HTTP {code}: {msg}"
            return f"HTTP {code}: {j}"
        except Exception:
            try:
                txt = (r.text or "").strip()
                return f"HTTP {code}: {txt}"
            except Exception:
                return f"HTTP {code}: {str(e)}"
    return str(e)


def _build_overlimit_text(words: int) -> str:
    # ASCII-слова чаще всего дают предсказуемый рост токенов
    # (важно: это ВХОД, чтобы переполнить контекст).
    return ("test " * max(1, int(words))).strip()


async def tokens_test_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /tokens_test — включает режим теста токенов.
    Далее любой текст -> отправляется в модель, бот отвечает ответом модели + статистикой токенов.
    /tokens_next — переключить этап (короткий/длинный/перелимит)
    /tokens_stop — выйти и показать сводку
    """
    if not update.message:
        return

    context.user_data[TOKENS_TEST_KEY] = {
        "active": True,
        "stage_idx": 0,
        "runs": {k: [] for k, _ in STAGES},  # список прогонов по этапам
    }

    msg = (
        "ТЕСТ ТОКЕНОВ — ВКЛ\n"
        "\n"
        "Как работает:\n"
        "- В этом режиме каждое твое сообщение отправляется в модель.\n"
        "- Я отвечаю ответом модели и следом показываю токены: запрос/ответ/всего.\n"
        "\n"
        "Этапы:\n"
        "1) КОРОТКИЙ\n"
        "2) ДЛИННЫЙ\n"
        "3) ПЕРЕЛИМИТ (переполнение КОНТЕКСТА/входа)\n"
        "\n"
        "Команды:\n"
        "/tokens_next — следующий этап\n"
        "/tokens_stop — выйти и показать сводку\n"
        "\n"
        "Примеры запросов:\n"
        "Короткий: Скажи одним предложением, что такое токен в LLM.\n"
        "Длинный: Сделай резюме этого текста в 7 пунктах + 10 терминов: <вставь большой текст>\n"
        "Перелимит: Напиши тему (например: «о лисе»). Я сам добавлю большой вход, чтобы переполнить контекст.\n"
        "\n"
        "Подсказка для перелимита (рекомендуется):\n"
        f"- В .env можно задать {ENV_OVERLIMIT_MODEL}=mistralai/mistral-7b-instruct-v0.1\n"
        f"- И {ENV_OVERLIMIT_WORDS}=6000 (если не сработало — увеличь)\n"
        "\n"
        "СЕЙЧАС: этап 1) КОРОТКИЙ — пришли свой короткий запрос."
    )
    await update.message.reply_text(msg)


async def tokens_next_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    st = context.user_data.get(TOKENS_TEST_KEY) or {}
    if not st.get("active"):
        await update.message.reply_text("ТЕСТ ТОКЕНОВ сейчас выключен. Включить: /tokens_test")
        return

    idx = int(st.get("stage_idx", 0))
    idx += 1

    if idx >= len(STAGES):
        # финал -> покажем сводку и выключим
        await _send_summary_and_stop(update, context)
        return

    st["stage_idx"] = idx
    context.user_data[TOKENS_TEST_KEY] = st

    _, label = STAGES[idx]
    await update.message.reply_text(f"Ок. СЕЙЧАС: этап {idx+1}) {label} — пришли свой запрос для этого этапа.")


async def tokens_stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    st = context.user_data.get(TOKENS_TEST_KEY) or {}
    if not st.get("active"):
        await update.message.reply_text("ТЕСТ ТОКЕНОВ уже выключен.")
        return
    await _send_summary_and_stop(update, context)


async def _send_summary_and_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    st = context.user_data.get(TOKENS_TEST_KEY) or {}
    runs = st.get("runs") or {}

    lines = []
    lines.append("ТЕСТ ТОКЕНОВ — СВОДКА")
    lines.append("")

    for i, (key, label) in enumerate(STAGES, start=1):
        arr = runs.get(key) or []
        if not arr:
            lines.append(f"{i}) {label}: нет данных")
            continue

        last = arr[-1]
        status = "OK" if last.get("ok") else "ОШИБКА"
        pt = last.get("prompt_tokens")
        ct = last.get("completion_tokens")
        tt = last.get("total_tokens")
        fr = last.get("finish_reason") or ""
        dt = last.get("time_ms")

        def fmt(x):
            return str(x) if isinstance(x, int) else "n/a"

        lines.append(
            f"{i}) {label}: {status} | time={dt:.0f} ms | "
            f"токены: запрос={fmt(pt)}, ответ={fmt(ct)}, всего={fmt(tt)}"
            + (f" | finish_reason={fr}" if fr else "")
            + (f" | прогонов={len(arr)}" if len(arr) > 1 else "")
        )

    context.user_data.pop(TOKENS_TEST_KEY, None)
    await update.message.reply_text("\n".join(lines))


async def tokens_test_intercept(update: Update, context: ContextTypes.DEFAULT_TYPE, user_text: str) -> bool:
    """
    Вызывается из main.py внутри on_text.
    Если тест активен — обрабатываем сообщение (модель отвечает + токены) и возвращаем True.
    """
    if not update.message:
        return False

    st = context.user_data.get(TOKENS_TEST_KEY) or {}
    if not st.get("active"):
        return False

    # deps из main.py (чтобы не дублировать логику температуры/модели/ответов)
    deps = (context.application.bot_data or {}).get("tokens_deps") or {}
    get_temperature = deps.get("get_temperature")
    get_model = deps.get("get_model")
    get_effective_model = deps.get("get_effective_model")
    system_prompt_text = deps.get("SYSTEM_PROMPT_TEXT") or "Ты ассистент. Отвечай кратко и по делу."
    safe_reply_text = deps.get("safe_reply_text")  # coroutine

    chat_id = int(update.effective_chat.id) if update.effective_chat else 0

    temperature = float(get_temperature(context, chat_id)) if callable(get_temperature) else 0.7
    model = (get_model(context, chat_id) or None) if callable(get_model) else None
    effective_model = (get_effective_model(context, chat_id) or "") if callable(get_effective_model) else (model or "")

    stage_idx = int(st.get("stage_idx", 0))
    stage_key, stage_label = STAGES[stage_idx]

    await update.message.chat.send_action("typing")

    # -------------------- SPECIAL: OVERLIMIT via huge INPUT --------------------
    if stage_key == "over":
        # можно задать отдельную модель именно для перелимита (лучше с маленьким контекстом)
        over_model = (os.getenv(ENV_OVERLIMIT_MODEL) or "").strip()
        if not over_model:
            # если не задано — используем текущую эффективную (как раньше), ничего не ломаем
            over_model = effective_model or (model or "")

        # раздуваем ВХОД, чтобы превысить контекст
        try:
            words = int(os.getenv(ENV_OVERLIMIT_WORDS) or str(DEFAULT_OVERLIMIT_WORDS))
        except Exception:
            words = DEFAULT_OVERLIMIT_WORDS

        big = _build_overlimit_text(words)
        over_user_text = (
            f"{user_text}\n\n"
            f"[OVERLIMIT_INPUT]\n"
            f"{big}\n"
            f"[/OVERLIMIT_INPUT]\n"
        )

        messages = [
            {"role": "system", "content": system_prompt_text},
            {"role": "user", "content": over_user_text},
        ]

        t0 = time.perf_counter()
        try:
            data = _chat_completion_raw_overlimit(messages, temperature=temperature, model=over_model)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            answer = _get_content(data) or "Пустой ответ от модели."
            pt, ct, tt = _get_usage(data)
            fr = _get_finish_reason(data)

            # 1) сначала ответ модели
            if callable(safe_reply_text):
                await safe_reply_text(update, answer)
            else:
                await update.message.reply_text(answer)

            # 2) затем токены
            report = []
            report.append("ТЕСТ ТОКЕНОВ")
            report.append(f"Этап: {stage_label} (переполнение контекста/входа)")
            report.append(f"Модель: {over_model}")
            report.append(f"Температура: {temperature}")
            report.append(f"Время: {dt_ms:.0f} ms")
            report.append(
                "Токены: "
                f"запрос={pt if pt is not None else 'n/a'}, "
                f"ответ={ct if ct is not None else 'n/a'}, "
                f"всего={tt if tt is not None else 'n/a'}"
            )
            if fr:
                report.append(f"finish_reason: {fr}")

            # если вдруг НЕ упали, значит контекст большой (или words мало)
            report.append(
                f"Примечание: если это НЕ ошибка переполнения — увеличь {ENV_OVERLIMIT_WORDS} "
                f"или задай {ENV_OVERLIMIT_MODEL} (модель с маленьким контекстом)."
            )

            if callable(safe_reply_text):
                await safe_reply_text(update, "\n".join(report))
            else:
                await update.message.reply_text("\n".join(report))

            run = {
                "ok": True,
                "time_ms": dt_ms,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": tt,
                "finish_reason": fr,
            }

        except Exception as e:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            err = _format_http_error(e)

            # Для перелимита это нормальный исход: API может отклонить запрос
            report = []
            report.append("ТЕСТ ТОКЕНОВ")
            report.append(f"Этап: {stage_label} (переполнение контекста/входа)")
            report.append(f"Модель: {over_model}")
            report.append(f"Температура: {temperature}")
            report.append(f"Время: {dt_ms:.0f} ms")
            report.append("ОШИБКА (это ожидаемо для перелимита контекста):")
            report.append(err)

            if callable(safe_reply_text):
                await safe_reply_text(update, "\n".join(report))
            else:
                await update.message.reply_text("\n".join(report))

            run = {
                "ok": False,
                "time_ms": dt_ms,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "finish_reason": "",
                "error": err,
            }

        # сохраняем прогон
        runs = st.get("runs") or {k: [] for k, _ in STAGES}
        runs.setdefault(stage_key, []).append(run)
        st["runs"] = runs
        context.user_data[TOKENS_TEST_KEY] = st

        hint = "Дальше: /tokens_next (следующий этап) или /tokens_stop (сводка и выход)."
        if callable(safe_reply_text):
            await safe_reply_text(update, hint)
        else:
            await update.message.reply_text(hint)

        return True

    # -------------------- NORMAL (short/long) --------------------
    messages = [
        {"role": "system", "content": system_prompt_text},
        {"role": "user", "content": user_text},
    ]

    t0 = time.perf_counter()
    try:
        data = chat_completion_raw(messages, temperature=temperature, model=model)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        answer = _get_content(data) or "Пустой ответ от модели."
        pt, ct, tt = _get_usage(data)
        fr = _get_finish_reason(data)

        # 1) сначала ответ модели (как у тебя уже было)
        if callable(safe_reply_text):
            await safe_reply_text(update, answer)
        else:
            await update.message.reply_text(answer)

        # 2) затем отдельным сообщением — токены
        report = []
        report.append("ТЕСТ ТОКЕНОВ")
        report.append(f"Этап: {stage_label}")
        report.append(f"Модель: {effective_model}")
        report.append(f"Температура: {temperature}")
        report.append(f"Время: {dt_ms:.0f} ms")
        report.append(
            "Токены: "
            f"запрос={pt if pt is not None else 'n/a'}, "
            f"ответ={ct if ct is not None else 'n/a'}, "
            f"всего={tt if tt is not None else 'n/a'}"
        )
        if fr:
            report.append(f"finish_reason: {fr}")

        if callable(safe_reply_text):
            await safe_reply_text(update, "\n".join(report))
        else:
            await update.message.reply_text("\n".join(report))

        run = {
            "ok": True,
            "time_ms": dt_ms,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": tt,
            "finish_reason": fr,
        }

    except Exception as e:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        err = _format_http_error(e)

        report = []
        report.append("ТЕСТ ТОКЕНОВ")
        report.append(f"Этап: {stage_label}")
        report.append(f"Модель: {effective_model}")
        report.append(f"Температура: {temperature}")
        report.append(f"Время: {dt_ms:.0f} ms")
        report.append("ОШИБКА запроса к OpenRouter:")
        report.append(err)

        if callable(safe_reply_text):
            await safe_reply_text(update, "\n".join(report))
        else:
            await update.message.reply_text("\n".join(report))

        run = {
            "ok": False,
            "time_ms": dt_ms,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "finish_reason": "",
            "error": err,
        }

    # сохраняем прогон
    runs = st.get("runs") or {k: [] for k, _ in STAGES}
    runs.setdefault(stage_key, []).append(run)
    st["runs"] = runs
    context.user_data[TOKENS_TEST_KEY] = st

    # подсказка “что дальше”
    hint = "Дальше: /tokens_next (следующий этап) или /tokens_stop (сводка и выход)."
    if callable(safe_reply_text):
        await safe_reply_text(update, hint)
    else:
        await update.message.reply_text(hint)

    return True
