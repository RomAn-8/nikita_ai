"""Micro-model first command handlers — День 10."""

import logging

from telegram import Update
from telegram.ext import ContextTypes

from ..core.errors import safe_reply_text
from ..config import ROUTING_SMALL_MODEL, ROUTING_LARGE_MODEL, MICRO_CONFIDENCE_THRESHOLD
from ..services import micro_model as mm
from ..services.multi_stage import MonolithicResult, MultiStageResult

logger = logging.getLogger(__name__)


async def iq_post_micro_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/iq_post_micro <описание главы> — micro-model классифицирует вход, выбирает стратегию."""
    if not update.message:
        return

    user_prompt = " ".join(context.args) if context.args else ""
    if not user_prompt.strip():
        await safe_reply_text(
            update,
            "Укажи описание главы.\nПример: /iq_post_micro Глава 5: Арморн обнаружил залежи руды.",
        )
        return

    await update.message.chat.send_action("typing")

    try:
        result = mm.micro_first_post(user_prompt)
    except Exception as e:
        logger.exception("iq_post_micro_cmd error: %s", e)
        await safe_reply_text(update, "Ошибка при micro-model генерации. Попробуйте позже.")
        return

    cls = result.classification
    label = cls.get("label", "—")
    confidence = float(cls.get("confidence", 0.0))
    status = cls.get("status", "—")
    reason = cls.get("reason", "—")
    small_short = ROUTING_SMALL_MODEL.split("/")[-1]
    large_short = ROUTING_LARGE_MODEL.split("/")[-1]
    status_icon = "✅" if status == "OK" else "⚠️"

    lines = ["🔬 Micro-model first:", ""]

    if result.fallback_used:
        lines.append(f"🏷️ Micro-классификация ({small_short}):")
        lines.append(f"  • Уверенность: {confidence:.2f} < {MICRO_CONFIDENCE_THRESHOLD} — UNSURE ⚠️")
        lines.append(f"  ⬆️ Fallback: переклассификация через {large_short}")
        lines.append("")
        lines.append(f"🏷️ Fallback-классификация ({large_short}):")
    else:
        lines.append(f"🏷️ Классификация ({small_short}):")

    lines.append(f"  • Метка: {label}")
    lines.append(f"  • Уверенность: {confidence:.2f}  |  Статус: {status} {status_icon}")
    lines.append(f"  • Причина: {reason}")
    lines.append("")

    strategy_note = "micro справилась" if not result.fallback_used else "после fallback"
    lines.append(f"⚙️ Стратегия: {result.strategy_used} ({strategy_note})")
    lines.append("")

    pr = result.post_result

    if isinstance(pr, MonolithicResult):
        constraint = pr.constraint_check
        link_icon = "✓" if constraint.get("has_link") else "✗"
        c_icon = "✅" if constraint.get("ok") else "❌"

        lines.append("📝 VK-анонс:")
        lines.append(pr.post if pr.post else "(нет текста)")
        lines.append("")
        lines.append(f"{c_icon} Constraint-check: {'OK' if constraint.get('ok') else 'FAIL'}  |  {constraint.get('body_length', '—')} симв.  |  Ссылка: {link_icon}")
        if not constraint.get("ok"):
            issues = constraint.get("issues", [])
            if issues:
                lines.append("⚠️ Проблемы: " + "; ".join(issues))
    else:
        # MultiStageResult
        stage1, stage2, stage3 = pr.stages
        a = stage1.output if isinstance(stage1.output, dict) else {}
        s = stage2.output if isinstance(stage2.output, dict) else {}
        constraint = pr.constraint_check
        link_icon = "✓" if constraint.get("has_link") else "✗"
        c_icon = "✅" if constraint.get("ok") else "❌"

        lines.append(f"📊 Этап 1 — Анализ ({small_short}, {stage1.latency_ms / 1000:.1f} сек):")
        lines.append(f"  • Тон: {a.get('tone', '—')}  |  Риск спойлера: {a.get('spoiler_risk', '—')}")
        lines.append(f"  • Фокус: {a.get('key_focus', '—')}")
        lines.append("")
        lines.append(f"🎯 Этап 2 — Стратегия ({small_short}, {stage2.latency_ms / 1000:.1f} сек):")
        lines.append(f"  • Длина: {s.get('length', '—')}  |  Открытие: {s.get('opening_type', '—')}")
        lines.append("")
        lines.append(f"📝 Этап 3 — Пост ({stage3.latency_ms / 1000:.1f} сек):")
        lines.append(pr.final_post if pr.final_post else "(нет текста)")
        lines.append("")
        lines.append(f"{c_icon} Constraint-check: {'OK' if constraint.get('ok') else 'FAIL'}  |  {constraint.get('body_length', '—')} симв.  |  Ссылка: {link_icon}")
        if not constraint.get("ok"):
            issues = constraint.get("issues", [])
            if issues:
                lines.append("⚠️ Проблемы: " + "; ".join(issues))

    micro_tok = result.micro_tokens
    gen_tok = result.total_tokens - micro_tok
    lines.append(f"⏱ {result.total_latency_ms / 1000:.1f} сек  |  токены: ~{result.total_tokens} (micro ~{micro_tok} + gen ~{gen_tok})")

    await safe_reply_text(update, "\n".join(lines))


async def iq_micro_stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/iq_micro_stats — статистика micro-model: micro vs fallback, latency."""
    if not update.message:
        return

    stats = mm.get_micro_stats()
    total = stats["total"]

    if total == 0:
        await safe_reply_text(update, "Статистика пуста — команда /iq_post_micro ещё не использовалась.")
        return

    def pct(n: int) -> str:
        return f"{round(n / total * 100)}%" if total else "—"

    lines = [
        "🔬 Micro-model статистика:",
        f"  Всего запросов: {total}",
        f"  ✅ Micro справилась: {stats['micro_handled']} ({pct(stats['micro_handled'])})",
        f"  ⬆️ Fallback использован: {stats['fallback_used']} ({pct(stats['fallback_used'])})",
        f"  ⚙️ Стратегий: monolithic {stats['strategy_monolithic']} | multistage {stats['strategy_multistage']}",
        f"  Токены: ~{stats['total_tokens']}  |  Среднее время: {stats['avg_latency_ms'] / 1000:.1f} сек",
    ]

    await safe_reply_text(update, "\n".join(lines))
