"""Telegram handlers for Day 14 security loop MVP."""

from __future__ import annotations

import logging
from typing import Any

from telegram import Update
from telegram.ext import ContextTypes

from ..core.errors import safe_reply_text
from ..services.day14_security_loop import Day14LoopError, run_security_loop

logger = logging.getLogger(__name__)

_LAST_RESULT_KEY = "sec14_last_result"


def _format_result_lines(result: dict[str, Any]) -> list[str]:
    """Build a compact human-readable summary."""
    lines = [
        f"Run ID: {result['run_id']}",
        f"Статус: {result['status']}",
        f"Задача: {result['task']}",
        f"Gateway режим: {result['gateway_mode']}",
        f"Попыток генерации: {result['attempts_used']}",
        f"Security repair attempts: {result['security_repairs_used']}",
        f"Итог: {result['summary']}",
        f"Target project: {result['workspace_root']}",
        f"Log: {result['log_path']}",
    ]

    if result.get("changed_paths"):
        lines.extend(["", "Изменённые файлы:"])
        lines.extend(f"  - {path}" for path in result["changed_paths"])

    if result.get("gateway_hits"):
        lines.extend(["", "Gateway:"])
        lines.extend(f"  - {entry}" for entry in result["gateway_hits"])

    if result.get("warnings"):
        lines.extend(["", "Warnings:"])
        lines.extend(f"  - {warning}" for warning in result["warnings"])

    if result.get("security_findings"):
        lines.extend(["", "Security findings:"])
        for finding in result["security_findings"]:
            lines.append(
                "  - {severity} | {path} | {title}".format(
                    severity=finding.get("severity") or "?",
                    path=finding.get("path") or "?",
                    title=finding.get("title") or "?",
                )
            )

    if result.get("missed_by_both"):
        lines.extend(["", "Missed by both:"])
        lines.extend(f"  - {item}" for item in result["missed_by_both"])

    validation = result.get("validation") or {}
    build_code = validation.get("build_returncode")
    test_code = validation.get("test_returncode")
    if build_code is not None or test_code is not None:
        lines.extend(
            [
                "",
                f"Build rc: {build_code if build_code is not None else '—'}",
                f"Test rc: {test_code if test_code is not None else '—'}",
            ]
        )
    return lines


def _to_summary_dict(loop_result) -> dict[str, Any]:
    """Reduce the service result to Telegram/session-safe data."""
    validation_dict: dict[str, Any] = {}
    if loop_result.validation and loop_result.status != "failed_generation":
        validation_dict = {
            "build_returncode": loop_result.validation.build.returncode,
            "test_returncode": loop_result.validation.test.returncode,
        }

    return {
        "run_id": loop_result.run_id,
        "task": loop_result.task,
        "status": loop_result.status,
        "summary": loop_result.summary,
        "workspace_root": str(loop_result.workspace_root),
        "log_path": str(loop_result.log_path),
        "attempts_used": loop_result.attempts_used,
        "security_repairs_used": loop_result.security_repairs_used,
        "gateway_mode": loop_result.gateway_mode,
        "gateway_hits": loop_result.gateway_hits,
        "security_findings": loop_result.security_findings,
        "warnings": loop_result.warnings,
        "changed_paths": loop_result.changed_paths,
        "missed_by_both": loop_result.missed_by_both,
        "validation": validation_dict,
    }


async def sec14_run_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/sec14_run <task> — run the narrow Day 14 MVP loop."""
    if not update.message:
        return

    task = " ".join(context.args).strip() if context.args else ""
    if not task:
        await safe_reply_text(update, "Использование: /sec14_run <задача>")
        return

    await update.message.chat.send_action("typing")
    try:
        result = run_security_loop(
            task=task,
            user_id=update.effective_user.id if update.effective_user else 0,
            user_data=context.user_data,
        )
    except Day14LoopError as exc:
        logger.exception("sec14_run_cmd loop error: %s", exc)
        await safe_reply_text(update, f"Day 14 loop завершился ошибкой: {exc}")
        return
    except Exception as exc:
        logger.exception("sec14_run_cmd unexpected error: %s", exc)
        await safe_reply_text(update, "Не удалось выполнить Day 14 loop. Проверьте логи.")
        return

    summary = _to_summary_dict(result)
    context.user_data[_LAST_RESULT_KEY] = summary
    await safe_reply_text(update, "\n".join(_format_result_lines(summary)))


async def sec14_status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/sec14_status — show the last Day 14 run status."""
    if not update.message:
        return

    summary = context.user_data.get(_LAST_RESULT_KEY)
    if not summary:
        await safe_reply_text(update, "Для этой сессии ещё нет запусков Day 14.")
        return

    lines = [
        f"Последний Day 14 run: {summary['run_id']}",
        f"Статус: {summary['status']}",
        f"Задача: {summary['task']}",
        f"Попыток генерации: {summary['attempts_used']}",
        f"Security repair attempts: {summary['security_repairs_used']}",
        f"Target project: {summary['workspace_root']}",
    ]
    await safe_reply_text(update, "\n".join(lines))


async def sec14_report_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/sec14_report — show the last Day 14 report."""
    if not update.message:
        return

    summary = context.user_data.get(_LAST_RESULT_KEY)
    if not summary:
        await safe_reply_text(update, "Для этой сессии ещё нет отчёта Day 14.")
        return

    await safe_reply_text(update, "\n".join(_format_result_lines(summary)))
