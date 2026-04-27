"""Multi-stage inference command handlers — День 9."""

import logging

from telegram import Update
from telegram.ext import ContextTypes

from ..core.errors import safe_reply_text
from ..config import ROUTING_SMALL_MODEL, OPENROUTER_MODEL
from ..services import multi_stage as ms

logger = logging.getLogger(__name__)


async def iq_post_monolithic_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/iq_post_monolithic <описание главы> — один запрос, один готовый VK-анонс."""
    if not update.message:
        return

    user_prompt = " ".join(context.args) if context.args else ""
    if not user_prompt.strip():
        await safe_reply_text(
            update,
            "Укажи описание главы.\nПример: /iq_post_monolithic Глава 5: Арморн обнаружил залежи руды.",
        )
        return

    await update.message.chat.send_action("typing")

    try:
        result = ms.monolithic_post(user_prompt)
    except Exception as e:
        logger.exception("iq_post_monolithic_cmd error: %s", e)
        await safe_reply_text(update, "Ошибка при генерации. Попробуйте позже.")
        return

    constraint = result.constraint_check
    link_icon = "✓" if constraint.get("has_link") else "✗"
    status_icon = "✅" if constraint.get("ok") else "❌"
    model_short = OPENROUTER_MODEL.split("/")[-1]

    lines = ["🔵 Monolithic inference:", ""]

    if result.post:
        lines.append("📝 VK-анонс:")
        lines.append(result.post)
        lines.append("")

    lines.append(f"{status_icon} Constraint-check: {'OK' if constraint.get('ok') else 'FAIL'}  |  {constraint.get('body_length', '—')} симв.  |  Ссылка: {link_icon}")

    if not constraint.get("ok"):
        issues = constraint.get("issues", [])
        if issues:
            lines.append("⚠️ Проблемы: " + "; ".join(issues))

    lines.append(f"🤖 Модель: {model_short}  |  ⏱ {result.latency_ms / 1000:.1f} сек  |  токены: ~{result.tokens_used}")

    await safe_reply_text(update, "\n".join(lines))


async def iq_post_multistage_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/iq_post_multistage <описание главы> — 3-этапная генерация: анализ → стратегия → пост."""
    if not update.message:
        return

    user_prompt = " ".join(context.args) if context.args else ""
    if not user_prompt.strip():
        await safe_reply_text(
            update,
            "Укажи описание главы.\nПример: /iq_post_multistage Глава 5: Арморн обнаружил залежи руды.",
        )
        return

    await update.message.chat.send_action("typing")

    try:
        result = ms.multistage_post(user_prompt)
    except Exception as e:
        logger.exception("iq_post_multistage_cmd error: %s", e)
        await safe_reply_text(update, "Ошибка при multi-stage генерации. Попробуйте позже.")
        return

    stage1, stage2, stage3 = result.stages
    a = stage1.output if isinstance(stage1.output, dict) else {}
    s = stage2.output if isinstance(stage2.output, dict) else {}
    constraint = result.constraint_check
    link_icon = "✓" if constraint.get("has_link") else "✗"
    status_icon = "✅" if constraint.get("ok") else "❌"
    small_short = ROUTING_SMALL_MODEL.split("/")[-1]
    main_short = OPENROUTER_MODEL.split("/")[-1]

    lines = ["🧩 Multi-stage inference (3 этапа):", ""]

    lines.append(f"📊 Этап 1 — Анализ ({small_short}, {stage1.latency_ms / 1000:.1f} сек):")
    lines.append(f"  • Тон: {a.get('tone', '—')}")
    lines.append(f"  • Риск спойлера: {a.get('spoiler_risk', '—')}")
    lines.append(f"  • Дуга: {a.get('chapter_arc', '—')}")
    lines.append(f"  • Фокус: {a.get('key_focus', '—')}")
    lines.append("")

    lines.append(f"🎯 Этап 2 — Стратегия ({small_short}, {stage2.latency_ms / 1000:.1f} сек):")
    lines.append(f"  • Длина: {s.get('length', '—')}")
    lines.append(f"  • Открытие: {s.get('opening_type', '—')}")
    lines.append(f"  • Стиль: {s.get('style_hint', '—')}")
    if s.get("avoid"):
        lines.append(f"  • Избегать: {s['avoid']}")
    lines.append("")

    lines.append(f"📝 Этап 3 — Пост ({main_short}, {stage3.latency_ms / 1000:.1f} сек):")
    if result.final_post:
        lines.append(result.final_post)
    else:
        lines.append("(нет текста)")
    lines.append("")

    lines.append(f"{status_icon} Constraint-check: {'OK' if constraint.get('ok') else 'FAIL'}  |  {constraint.get('body_length', '—')} симв.  |  Ссылка: {link_icon}")
    if not constraint.get("ok"):
        issues = constraint.get("issues", [])
        if issues:
            lines.append("⚠️ Проблемы: " + "; ".join(issues))
    lines.append(f"⏱ {result.total_latency_ms / 1000:.1f} сек итого  |  токены: ~{result.total_tokens}")

    await safe_reply_text(update, "\n".join(lines))


async def iq_post_compare_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/iq_post_compare <описание главы> — сравнение monolithic vs multi-stage."""
    if not update.message:
        return

    user_prompt = " ".join(context.args) if context.args else ""
    if not user_prompt.strip():
        await safe_reply_text(
            update,
            "Укажи описание главы.\nПример: /iq_post_compare Глава 5: Арморн обнаружил залежи руды.",
        )
        return

    await update.message.chat.send_action("typing")

    try:
        result = ms.compare_posts(user_prompt)
    except Exception as e:
        logger.exception("iq_post_compare_cmd error: %s", e)
        await safe_reply_text(update, "Ошибка при сравнении. Попробуйте позже.")
        return

    mono = result.monolithic
    multi = result.multistage
    model_short = OPENROUTER_MODEL.split("/")[-1]

    mono_icon = "✅" if mono.constraint_check.get("ok") else "❌"
    multi_icon = "✅" if multi.constraint_check.get("ok") else "❌"
    mono_link = "✓" if mono.constraint_check.get("has_link") else "✗"
    multi_link = "✓" if multi.constraint_check.get("has_link") else "✗"

    # Diff latency и токенов
    lat_diff = round((multi.total_latency_ms - mono.latency_ms) / max(mono.latency_ms, 1) * 100)
    tok_diff = round((multi.total_tokens - mono.tokens_used) / max(mono.tokens_used, 1) * 100)
    len_diff = (multi.constraint_check.get("body_length", 0) or 0) - (mono.constraint_check.get("body_length", 0) or 0)

    lines = ["⚖️ Monolithic vs Multi-stage:", ""]

    lines.append(f"🔵 Monolithic ({model_short}, {mono.latency_ms / 1000:.1f} сек, ~{mono.tokens_used} токенов):")
    if mono.post:
        lines.append(mono.post)
    else:
        lines.append("(нет текста)")
    lines.append(f"{mono_icon} {mono.constraint_check.get('body_length', '—')} симв. | Ссылка: {mono_link}")
    lines.append("")

    lines.append(f"🟢 Multi-stage (3 вызова, {multi.total_latency_ms / 1000:.1f} сек, ~{multi.total_tokens} токенов):")
    if multi.final_post:
        lines.append(multi.final_post)
    else:
        lines.append("(нет текста)")
    lines.append(f"{multi_icon} {multi.constraint_check.get('body_length', '—')} симв. | Ссылка: {multi_link}")
    lines.append("")

    sign = "+" if lat_diff >= 0 else ""
    tok_sign = "+" if tok_diff >= 0 else ""
    len_sign = "+" if len_diff >= 0 else ""

    lines.append("📊 Сравнение:")
    lines.append(f"  Latency:  mono {mono.latency_ms / 1000:.1f} с  vs  multi {multi.total_latency_ms / 1000:.1f} с  ({sign}{lat_diff}%)")
    lines.append(f"  Токены:   mono ~{mono.tokens_used}  vs  multi ~{multi.total_tokens}  ({tok_sign}{tok_diff}%)")
    lines.append(f"  Длина:    mono {mono.constraint_check.get('body_length', '—')}  vs  multi {multi.constraint_check.get('body_length', '—')}  ({len_sign}{len_diff} симв.)")
    lines.append(f"  Формат:   mono {mono_icon}  vs  multi {multi_icon}")

    await safe_reply_text(update, "\n".join(lines))
