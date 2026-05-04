"""Handlers для LLM Gateway — День 13.

Команды:
  /gw_prompt [текст]  — отправить промпт через gateway (или активировать ожидание)
  /gw_mode <режим>    — переключить режим (block | redact | restore)
  /gw_audit           — последние записи audit log
  /gw_stats           — статистика сессии и за сегодня
"""

import logging

from telegram import Update
from telegram.ext import ContextTypes

from ..core.errors import safe_reply_text
from ..services import gateway as gw_service
from ..services.gateway_audit import read_recent_events, read_today_stats

logger = logging.getLogger(__name__)

_VALID_MODES = ("block", "redact", "restore")
_DEFAULT_MODE = "redact"
_MODE_KEY = "gw_mode"
_WAITING_KEY = "gw_waiting"

_CANCEL_PHRASES = frozenset(["/cancel", "cancel", "отмена", "выход"])


def _get_mode(context: ContextTypes.DEFAULT_TYPE) -> str:
    return context.user_data.get(_MODE_KEY, _DEFAULT_MODE)


async def _run_and_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str) -> None:
    """Запускает gateway pipeline и отправляет результат пользователю."""
    mode = _get_mode(context)
    user_id = update.effective_user.id if update.effective_user else 0

    await update.message.chat.send_action("typing")

    try:
        result = gw_service.run_gateway(
            prompt=prompt,
            mode=mode,
            user_id=user_id,
            user_data=context.user_data,
        )
    except Exception as e:
        logger.exception("gateway _run_and_reply error: %s", e)
        await safe_reply_text(update, "Ошибка Gateway. Попробуйте позже.")
        return

    lines = [f"[Gateway / режим: {mode}]", ""]

    if result.input_findings:
        types = ", ".join(f.type for f in result.input_findings)
        lines.append(f"Обнаружено в промпте: {types}")
        if result.redacted:
            lines.append("Секреты заменены плейсхолдерами перед отправкой в LLM.")
        lines.append("")

    lines.append(result.response)

    if result.restored:
        lines += ["", "Оригинальные значения восстановлены в ответе."]

    if result.output_findings:
        lines += ["", f"Output guard: {len(result.output_findings)} проблем в ответе LLM."]

    cost_str = f"${result.cost_usd:.6f}" if result.cost_usd else "—"
    lines += [
        "",
        f"Токены: {result.prompt_tokens}p / {result.completion_tokens}c | Стоимость: {cost_str}",
    ]

    await safe_reply_text(update, "\n".join(lines))


async def gw_handle_intercepted(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
    """Обрабатывает перехваченное текстовое сообщение как gateway-промпт.

    Вызывается из on_text в main.py когда gw_waiting=True.
    Флаг уже сброшен вызывающей стороной до вызова этой функции.
    """
    if not update.message:
        return

    if text.lower().strip() in _CANCEL_PHRASES:
        await safe_reply_text(update, "Gateway-режим отменён.")
        return

    await _run_and_reply(update, context, text)


async def gw_prompt_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/gw_prompt [текст] — отправить промпт через LLM Gateway.

    Без текста активирует режим ожидания: следующее сообщение уйдёт в gateway.
    С текстом обрабатывает промпт немедленно.
    """
    if not update.message:
        return

    prompt = " ".join(context.args).strip() if context.args else ""

    if not prompt:
        # Режим ожидания: следующее сообщение перехватит on_text
        context.user_data[_WAITING_KEY] = True
        mode = _get_mode(context)
        await safe_reply_text(
            update,
            f"Gateway-режим активирован (режим: {mode}).\n"
            "Отправьте следующий prompt для проверки.\n"
            "Для отмены: /cancel",
        )
        return

    await _run_and_reply(update, context, prompt)


async def gw_mode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/gw_mode <block|redact|restore> — переключить режим Gateway."""
    if not update.message:
        return

    if not context.args:
        current = _get_mode(context)
        await safe_reply_text(
            update,
            f"Текущий режим: {current}\n\n"
            "Доступные режимы:\n"
            "  block   — блокировать запросы с секретами\n"
            "  redact  — заменить секреты плейсхолдерами (LLM видит [REDACTED_...])\n"
            "  restore — как redact, но LLM-ответ получает оригинальные значения обратно\n\n"
            "Пример: /gw_mode redact",
        )
        return

    new_mode = context.args[0].strip().lower()
    if new_mode not in _VALID_MODES:
        await safe_reply_text(
            update,
            f"Неизвестный режим: {new_mode}\nДоступно: {', '.join(_VALID_MODES)}",
        )
        return

    context.user_data[_MODE_KEY] = new_mode
    # Сбрасываем ожидание если режим поменяли во время него
    context.user_data.pop(_WAITING_KEY, None)
    await safe_reply_text(update, f"Режим Gateway: {new_mode}")


async def gw_audit_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/gw_audit — последние 10 записей audit log."""
    if not update.message:
        return

    events = read_recent_events(n=10)
    if not events:
        await safe_reply_text(update, "Audit log пуст (нет событий за сегодня).")
        return

    lines = [f"Последние {len(events)} событий Gateway:"]
    for e in reversed(events):
        ts = e.get("ts", "?")[:19].replace("T", " ")
        mode = e.get("mode", "?")
        findings = e.get("input_findings", [])
        blocked = "🚫" if e.get("blocked") else "✅"
        redacted = " redact" if e.get("redacted") else ""
        restored = " restore" if e.get("restored") else ""
        cost = f"${e.get('cost_usd', 0):.6f}"
        f_str = f" [{', '.join(findings)}]" if findings else ""
        lines.append(f"{blocked} {ts} | {mode}{redacted}{restored}{f_str} | {cost}")

    await safe_reply_text(update, "\n".join(lines))


async def gw_stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/gw_stats — статистика Gateway за сегодня и текущей сессии."""
    if not update.message:
        return

    today = read_today_stats()
    session = context.user_data.get("gw_session_stats", {})
    mode = _get_mode(context)

    lines = [
        f"Gateway статистика | режим: {mode}",
        "",
        "Сегодня (audit log):",
        f"  Запросов: {today['total']}",
        f"  Заблокировано: {today['blocked']}",
        f"  Redacted: {today['redacted']}",
        f"  Output guard срабатываний: {today['output_triggered']}",
        f"  Всего findings: {today['findings_total']}",
        f"  Токены: {today['prompt_tokens']}p / {today['completion_tokens']}c",
        f"  Стоимость: ${today['cost_usd']:.6f}",
    ]

    if session:
        lines += [
            "",
            "Сессия (в памяти):",
            f"  Запросов: {session.get('requests', 0)}",
            f"  Заблокировано: {session.get('blocked', 0)}",
            f"  Redacted: {session.get('redacted', 0)}",
            f"  Всего findings: {session.get('findings_total', 0)}",
            f"  Стоимость: ${session.get('cost_usd', 0.0):.6f}",
        ]

    await safe_reply_text(update, "\n".join(lines))
