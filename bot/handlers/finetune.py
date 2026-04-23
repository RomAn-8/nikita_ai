"""Fine-tuning pipeline Telegram command handlers."""

import logging

from telegram import Update
from telegram.ext import ContextTypes

from ..core.errors import safe_reply_text
from ..services import finetune_service

logger = logging.getLogger(__name__)


async def ft_status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показать метаданные файлов датасета fine-tuning."""
    if not update.message:
        return

    await update.message.chat.send_action("typing")

    try:
        status = finetune_service.get_dataset_status()
    except Exception as e:
        logger.exception("ft_status_cmd error: %s", e)
        await safe_reply_text(update, "Ошибка при чтении датасета.")
        return

    lines = ["📊 Датасет fine-tuning:"]
    for key, info in status.items():
        if not info.get("exists"):
            lines.append(f"  {info['filename']} — файл не найден")
        else:
            lines.append(f"  {info['filename']} — {info['lines']} строк, {info['size_kb']} KB")

    await safe_reply_text(update, "\n".join(lines))


async def ft_validate_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Валидировать train.jsonl и eval.jsonl."""
    if not update.message:
        return

    await update.message.chat.send_action("typing")

    try:
        result = finetune_service.validate_datasets()
    except Exception as e:
        logger.exception("ft_validate_cmd error: %s", e)
        await safe_reply_text(update, "Ошибка при валидации датасета.")
        return

    lines = ["🔍 Валидация датасета:"]
    for key, info in result.items():
        filename = info["filename"]
        if not info.get("exists"):
            lines.append(f"  ❓ {filename} — файл не найден")
            continue
        errors = info["errors"]
        total = info["total"]
        if not errors:
            lines.append(f"  ✅ {filename} — {total} строк, ошибок нет")
        else:
            lines.append(f"  ❌ {filename} — {total} строк, {len(errors)} ошибок:")
            for err in errors[:5]:
                lines.append(f"      Строка {err['line']}: {err['reason']}")
            if len(errors) > 5:
                lines.append(f"      ... и ещё {len(errors) - 5}")

    await safe_reply_text(update, "\n".join(lines))


async def ft_baseline_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Краткая статистика по baseline_results.jsonl."""
    if not update.message:
        return

    await update.message.chat.send_action("typing")

    try:
        summary = finetune_service.get_baseline_summary()
    except Exception as e:
        logger.exception("ft_baseline_cmd error: %s", e)
        await safe_reply_text(update, "Ошибка при чтении baseline результатов.")
        return

    if not summary.get("exists"):
        await safe_reply_text(update, "Файл baseline_results.jsonl не найден.")
        return

    count = summary.get("count", 0)
    if count == 0:
        await safe_reply_text(update, "baseline_results.jsonl пустой.")
        return

    lines = [
        f"📋 Baseline results ({count} примеров):",
        f"  baseline_response:  avg {summary['baseline_avg']} симв. ({summary['baseline_min']}–{summary['baseline_max']})",
        f"  expected_response:  avg {summary['expected_avg']} симв. ({summary['expected_min']}–{summary['expected_max']})",
    ]

    await safe_reply_text(update, "\n".join(lines))


async def ft_dryrun_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Проверить параметры fine-tuning pipeline без API-вызовов."""
    if not update.message:
        return

    await update.message.chat.send_action("typing")

    try:
        info = finetune_service.run_finetune_dryrun()
    except FileNotFoundError as e:
        await safe_reply_text(update, str(e))
        return
    except Exception as e:
        logger.exception("ft_dryrun_cmd error: %s", e)
        await safe_reply_text(update, "Ошибка при проверке параметров pipeline.")
        return

    lines = [
        "🔍 DRY-RUN fine-tuning pipeline:",
        f"  Файл:     {info['file']} ({info['lines']} строк, {info['size_kb']} KB)",
        f"  Модель:   {info['model']}",
        f"  Интервал: {info['interval']} сек",
        "  API-запросы не отправлялись.",
    ]

    await safe_reply_text(update, "\n".join(lines))
