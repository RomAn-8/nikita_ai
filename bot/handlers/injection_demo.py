"""Handlers для учебной демонстрации indirect prompt injection — День 12."""

import logging

from telegram import Update
from telegram.ext import ContextTypes, ApplicationHandlerStop

from ..core.errors import safe_reply_text
from ..services import injection_demo as demo
from ..services.injection_demo import (
    InjectionResult,
    PAYLOAD_HTML,
    PAYLOAD_ZWSP,
    PAYLOAD_CTX,
)
from ..services.injection_sanitizer import validate_output

logger = logging.getLogger(__name__)

_RESULTS_KEY = "inj_results"


def _get_results(context: ContextTypes.DEFAULT_TYPE) -> list[InjectionResult]:
    return context.user_data.get(_RESULTS_KEY, [])


def _upsert_result(context: ContextTypes.DEFAULT_TYPE, result: InjectionResult) -> None:
    results = _get_results(context)
    results = [r for r in results if r.vector != result.vector]
    results.append(result)
    context.user_data[_RESULTS_KEY] = results


async def inj_html_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/inj_html — атака: HTML-комментарий с скрытой инструкцией в документе."""
    if not update.message:
        return

    await update.message.chat.send_action("typing")

    doc_preview = demo.get_poisoned_doc_html()

    try:
        response = demo.run_attack_html()
    except Exception as e:
        logger.exception("inj_html_cmd error: %s", e)
        await safe_reply_text(update, "Ошибка при выполнении атаки. Попробуйте позже.")
        return

    fired = not validate_output(response, PAYLOAD_HTML)
    result = InjectionResult(
        vector="html",
        attack_response=response,
        attack_fired=fired,
        safe_response="",
        defense_held=False,
    )
    _upsert_result(context, result)

    status = "✅ СРАБОТАЛА" if fired else "❌ не сработала"
    lines = [
        "🔴 Вектор 1: HTML Comment Injection",
        "",
        "📄 Вредоносный документ:",
        "```",
        doc_preview,
        "```",
        "",
        f"🤖 Ответ агента (без защиты):",
        response,
        "",
        f"Атака: {status}",
    ]
    if fired:
        lines.append(f'  → фраза «{PAYLOAD_HTML}» присутствует в ответе')
    else:
        lines.append(f'  → фраза «{PAYLOAD_HTML}» не найдена в ответе')

    await safe_reply_text(update, "\n".join(lines))


async def inj_zwsp_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/inj_zwsp — атака: невидимые zero-width символы несут скрытый payload."""
    if not update.message:
        return

    await update.message.chat.send_action("typing")

    try:
        response = demo.run_attack_zwsp()
    except Exception as e:
        logger.exception("inj_zwsp_cmd error: %s", e)
        await safe_reply_text(update, "Ошибка при выполнении атаки. Попробуйте позже.")
        return

    fired = not validate_output(response, PAYLOAD_ZWSP)
    result = InjectionResult(
        vector="zwsp",
        attack_response=response,
        attack_fired=fired,
        safe_response="",
        defense_held=False,
    )
    _upsert_result(context, result)

    status = "✅ СРАБОТАЛА" if fired else "❌ не сработала"
    lines = [
        "🔴 Вектор 2: Zero-Width Space Injection",
        "",
        "📄 Видимый текст документа (невидимый payload скрыт между строками):",
        "```",
        "Итоги квартала: выручка выросла на 12%, клиентская база расширилась.",
        "Новые контракты: 8 корпоративных клиентов, средний чек увеличился на 15%.",
        "[скрытый payload: U+200B ...DOCUMENT COMPROMISED... U+200B]",
        "```",
        "",
        "🤖 Ответ агента (без защиты):",
        response,
        "",
        f"Атака: {status}",
    ]
    if fired:
        lines.append(f'  → слово «{PAYLOAD_ZWSP}» присутствует в ответе')
    else:
        lines.append(f'  → слово «{PAYLOAD_ZWSP}» не найдено в ответе')

    await safe_reply_text(update, "\n".join(lines))


async def inj_ctx_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/inj_ctx — атака: фейковая история разговора внутри документа."""
    if not update.message:
        return

    await update.message.chat.send_action("typing")

    doc_preview = demo.get_poisoned_doc_ctx()

    try:
        response = demo.run_attack_ctx()
    except Exception as e:
        logger.exception("inj_ctx_cmd error: %s", e)
        await safe_reply_text(update, "Ошибка при выполнении атаки. Попробуйте позже.")
        return

    fired = not validate_output(response, PAYLOAD_CTX)
    result = InjectionResult(
        vector="ctx",
        attack_response=response,
        attack_fired=fired,
        safe_response="",
        defense_held=False,
    )
    _upsert_result(context, result)

    status = "✅ СРАБОТАЛА" if fired else "❌ не сработала"
    lines = [
        "🔴 Вектор 3: Fake Context Injection",
        "",
        "📄 Вредоносный документ:",
        "```",
        doc_preview,
        "```",
        "",
        "🤖 Ответ агента (без защиты):",
        response,
        "",
        f"Атака: {status}",
    ]
    if fired:
        lines.append(f'  → фраза «{PAYLOAD_CTX}» присутствует в ответе')
    else:
        lines.append(f'  → фраза «{PAYLOAD_CTX}» не найдена в ответе')

    await safe_reply_text(update, "\n".join(lines))


async def inj_safe_html_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/inj_safe_html — защита от HTML comment injection: strip + boundary + validation."""
    if not update.message:
        return

    await update.message.chat.send_action("typing")

    try:
        response, log = demo.run_safe_html_agent()
    except Exception as e:
        logger.exception("inj_safe_html_cmd error: %s", e)
        await safe_reply_text(update, "Ошибка при выполнении защиты. Попробуйте позже.")
        return

    held = validate_output(response, PAYLOAD_HTML)

    # Обновляем safe_response в сохранённом результате
    existing = next((r for r in _get_results(context) if r.vector == "html"), None)
    if existing:
        existing.safe_response = response
        existing.defense_held = held
    else:
        _upsert_result(context, InjectionResult(
            vector="html",
            attack_response="",
            attack_fired=False,
            safe_response=response,
            defense_held=held,
        ))

    status = "✅ УСТОЯЛА" if held else "❌ пробита"
    lines = [
        "🛡 Защита от HTML Comment Injection",
        "",
        "Применённые слои:",
    ]
    lines.extend(f"  {entry}" for entry in log)
    lines.extend([
        "",
        "🤖 Ответ защищённого агента:",
        response,
        "",
        f"Защита: {status}",
    ])

    await safe_reply_text(update, "\n".join(lines))


async def inj_safe_zwsp_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/inj_safe_zwsp — защита от zero-width injection: strip + boundary + validation."""
    if not update.message:
        return

    await update.message.chat.send_action("typing")

    try:
        response, log = demo.run_safe_zwsp_agent()
    except Exception as e:
        logger.exception("inj_safe_zwsp_cmd error: %s", e)
        await safe_reply_text(update, "Ошибка при выполнении защиты. Попробуйте позже.")
        return

    held = validate_output(response, PAYLOAD_ZWSP)

    existing = next((r for r in _get_results(context) if r.vector == "zwsp"), None)
    if existing:
        existing.safe_response = response
        existing.defense_held = held
    else:
        _upsert_result(context, InjectionResult(
            vector="zwsp",
            attack_response="",
            attack_fired=False,
            safe_response=response,
            defense_held=held,
        ))

    status = "✅ УСТОЯЛА" if held else "❌ пробита"
    lines = [
        "🛡 Защита от Zero-Width Space Injection",
        "",
        "Применённые слои:",
    ]
    lines.extend(f"  {entry}" for entry in log)
    lines.extend([
        "",
        "🤖 Ответ защищённого агента:",
        response,
        "",
        f"Защита: {status}",
    ])

    await safe_reply_text(update, "\n".join(lines))


async def inj_safe_ctx_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/inj_safe_ctx — защита от fake context injection: boundary markers + validation."""
    if not update.message:
        return

    await update.message.chat.send_action("typing")

    try:
        response, log = demo.run_safe_ctx_agent()
    except Exception as e:
        logger.exception("inj_safe_ctx_cmd error: %s", e)
        await safe_reply_text(update, "Ошибка при выполнении защиты. Попробуйте позже.")
        return

    held = validate_output(response, PAYLOAD_CTX)

    existing = next((r for r in _get_results(context) if r.vector == "ctx"), None)
    if existing:
        existing.safe_response = response
        existing.defense_held = held
    else:
        _upsert_result(context, InjectionResult(
            vector="ctx",
            attack_response="",
            attack_fired=False,
            safe_response=response,
            defense_held=held,
        ))

    status = "✅ УСТОЯЛА" if held else "❌ пробита"
    lines = [
        "🛡 Защита от Fake Context Injection",
        "",
        "Применённые слои:",
    ]
    lines.extend(f"  {entry}" for entry in log)
    lines.extend([
        "",
        "🤖 Ответ защищённого агента:",
        response,
        "",
        f"Защита: {status}",
    ])

    await safe_reply_text(update, "\n".join(lines))


async def inj_report_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/inj_report — итоговый отчёт: какие атаки сработали, какие защиты устояли."""
    if not update.message:
        return

    results = _get_results(context)
    report = demo.build_comparison_report(results)
    await safe_reply_text(update, report)


# ─── Custom file upload flow (День 12 расширение) ──────────────────────────

_UPLOAD_MODE_KEY = "inj_upload_mode"
_UPLOAD_DOC_KEY = "inj_upload_doc"
_UPLOAD_NAME_KEY = "inj_upload_filename"
_FILE_RESULTS_KEY = "inj_file_results"


async def inj_upload_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/inj_upload — активирует режим ожидания injection-demo файла (.txt/.md)."""
    if not update.message:
        return
    context.user_data[_UPLOAD_MODE_KEY] = True
    await safe_reply_text(
        update,
        "Режим загрузки активирован.\nОтправь .txt или .md файл с вредоносным payload.",
    )


async def inj_document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Перехватывает .txt/.md файлы когда inj_upload_mode=True (группа -1).

    Если режим не активен — возвращает None, и on_document в group=0 обрабатывает файл как обычно.
    """
    if not context.user_data.get(_UPLOAD_MODE_KEY):
        return

    if not update.message or not update.message.document:
        return

    document = update.message.document
    fname = (document.file_name or "").lower()

    if not (fname.endswith(".txt") or fname.endswith(".md")):
        await safe_reply_text(update, "Поддерживаются только .txt и .md файлы.")
        raise ApplicationHandlerStop

    try:
        file = await context.bot.get_file(document.file_id)
        content_bytes = await file.download_as_bytearray()
        text = content_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        logger.exception("inj_document_handler: ошибка скачивания файла: %s", e)
        await safe_reply_text(update, "Не удалось прочитать файл. Попробуйте ещё раз.")
        raise ApplicationHandlerStop

    context.user_data[_UPLOAD_DOC_KEY] = text
    context.user_data[_UPLOAD_NAME_KEY] = document.file_name or fname
    context.user_data[_UPLOAD_MODE_KEY] = False
    context.user_data.pop(_FILE_RESULTS_KEY, None)

    preview = text[:200] + "..." if len(text) > 200 else text
    await safe_reply_text(
        update,
        f"Файл «{document.file_name}» загружен ({len(text)} символов).\n\nПревью:\n{preview}\n\n"
        "Теперь: /inj_file_unsafe → /inj_file_safe → /inj_file_report",
    )
    raise ApplicationHandlerStop


async def inj_file_unsafe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/inj_file_unsafe — unsafe-анализ загруженного файла без защиты."""
    if not update.message:
        return

    doc = context.user_data.get(_UPLOAD_DOC_KEY)
    if not doc:
        await safe_reply_text(update, "Файл не загружен. Сначала /inj_upload.")
        return

    await update.message.chat.send_action("typing")
    fname = context.user_data.get(_UPLOAD_NAME_KEY, "файл")

    try:
        response = demo.run_custom_unsafe(doc)
    except Exception as e:
        logger.exception("inj_file_unsafe_cmd error: %s", e)
        await safe_reply_text(update, "Ошибка при анализе. Попробуйте позже.")
        return

    context.user_data.setdefault(_FILE_RESULTS_KEY, {})["unsafe"] = response
    await safe_reply_text(update, f"Unsafe-анализ «{fname}» (без защиты):\n\n{response}")


async def inj_file_safe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/inj_file_safe — safe-анализ загруженного файла со всеми слоями защиты."""
    if not update.message:
        return

    doc = context.user_data.get(_UPLOAD_DOC_KEY)
    if not doc:
        await safe_reply_text(update, "Файл не загружен. Сначала /inj_upload.")
        return

    await update.message.chat.send_action("typing")
    fname = context.user_data.get(_UPLOAD_NAME_KEY, "файл")

    try:
        response, log = demo.run_custom_safe(doc)
    except Exception as e:
        logger.exception("inj_file_safe_cmd error: %s", e)
        await safe_reply_text(update, "Ошибка при анализе. Попробуйте позже.")
        return

    results = context.user_data.setdefault(_FILE_RESULTS_KEY, {})
    results["safe"] = response
    results["safe_log"] = log

    lines = [f"Safe-анализ «{fname}»:", "", "Применённые слои:"]
    lines += [f"  {entry}" for entry in log]
    lines += ["", response]
    await safe_reply_text(update, "\n".join(lines))


async def inj_file_report_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/inj_file_report — сравнение unsafe vs safe для загруженного файла."""
    if not update.message:
        return

    doc = context.user_data.get(_UPLOAD_DOC_KEY)
    if not doc:
        await safe_reply_text(update, "Файл не загружен. Сначала /inj_upload.")
        return

    fname = context.user_data.get(_UPLOAD_NAME_KEY, "файл")
    results = context.user_data.get(_FILE_RESULTS_KEY, {})

    lines = [f"Отчёт по файлу «{fname}»:", ""]

    if "unsafe" not in results and "safe" not in results:
        lines.append(
            "Тесты ещё не запускались.\n"
            "Используйте /inj_file_unsafe и /inj_file_safe."
        )
    else:
        if "unsafe" in results:
            lines += ["Unsafe-ответ (без защиты):", results["unsafe"], ""]
        else:
            lines.append("Unsafe: не тестировался\n")

        if "safe" in results:
            lines += ["Safe-ответ (с защитой):", results["safe"], ""]
        else:
            lines.append("Safe: не тестировался\n")

        if "unsafe" in results and "safe" in results:
            lines.append(
                "Сравни ответы: содержит ли unsafe нежелательный payload, "
                "которого нет в safe?"
            )

    await safe_reply_text(update, "\n".join(lines))
