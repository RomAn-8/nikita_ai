"""Inference Quality Control command handlers — День 7."""

import logging

from telegram import Update
from telegram.ext import ContextTypes

from ..core.errors import safe_reply_text
from ..services import inference_quality as iq

logger = logging.getLogger(__name__)


async def iq_post_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/iq_post <описание главы> — VK-анонс с constraint-check + self-check + scoring."""
    if not update.message:
        return

    user_prompt = " ".join(context.args) if context.args else ""
    if not user_prompt.strip():
        await safe_reply_text(
            update,
            "Укажи описание главы.\nПример: /iq_post Глава 5: Арморн обнаружил залежи руды в шахте.",
        )
        return

    await update.message.chat.send_action("typing")

    try:
        result = iq.generate_with_confidence(user_prompt)
    except Exception as e:
        logger.exception("iq_post_cmd error: %s", e)
        await safe_reply_text(update, "Ошибка при генерации анонса. Попробуйте позже.")
        return

    sc = result.checks.get("self_check", {})
    constraint = result.checks.get("constraint", {})

    status_icon = {"OK": "✅", "UNSURE": "⚠️", "FAIL": "❌"}.get(result.status, "❓")
    link_icon = "✓" if constraint.get("has_link") else "✗"
    sc_verdict = sc.get("verdict", "—")
    sc_issues = sc.get("issues", [])
    sc_line = "без спойлеров, стиль ✓" if sc_verdict == "OK" else ", ".join(sc_issues) if sc_issues else sc_verdict

    lines = []

    if result.status != "FAIL" and result.text:
        lines.append("📝 VK-анонс:")
        lines.append(result.text)
        lines.append("")

    lines.append(f"{status_icon} Статус: {result.status}  |  Уверенность: {result.confidence:.2f}")
    lines.append(f"📏 Длина: {constraint.get('body_length', '—')} симв. | Ссылка: {link_icon}")
    lines.append(f"🔍 Self-check: {sc_line}")
    lines.append(f"🔁 Попыток: {result.retries + 1}  |  ⏱ {result.latency_ms / 1000:.1f} сек")

    if result.status == "FAIL":
        issues = constraint.get("issues", [])
        if issues:
            lines.append("⚠️ Проблемы: " + "; ".join(issues))

    await safe_reply_text(update, "\n".join(lines))


async def iq_redundancy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/iq_redundancy <описание главы> — 3 прогона + LLM-судья: сравнение по смыслу."""
    if not update.message:
        return

    user_prompt = " ".join(context.args) if context.args else ""
    if not user_prompt.strip():
        await safe_reply_text(
            update,
            "Укажи описание главы.\nПример: /iq_redundancy Глава 5: Арморн в шахте.",
        )
        return

    await update.message.chat.send_action("typing")

    try:
        result = iq.redundancy_check(user_prompt)
    except Exception as e:
        logger.exception("iq_redundancy_cmd error: %s", e)
        await safe_reply_text(update, "Ошибка при redundancy check. Попробуйте позже.")
        return

    n = result["n"]
    cv_pct = result["length_variation_pct"]
    status_icon = {"CONSISTENT": "✅", "VARIABLE": "⚠️", "UNSTABLE": "❌"}.get(result["status"], "❓")

    judge = result.get("judge", {})
    judge_verdict = judge.get("verdict", "UNSURE")
    judge_conf = judge.get("confidence", 0.0)
    judge_issues = judge.get("issues", [])
    judge_icon = {"CONSISTENT": "✅", "UNSURE": "⚠️", "DIVERGENT": "❌"}.get(judge_verdict, "❓")

    lines = [f"🔁 Redundancy check ({n} прогона):"]

    for i, run in enumerate(result["runs"], 1):
        ok_icon = "✅" if run["constraint"]["ok"] else "❌"
        length = run.get("length", 0)
        lines.append(f"\n📝 Вариант #{i} ({length} симв. — {ok_icon}):")
        if run["text"]:
            lines.append(run["text"])
        else:
            lines.append("(нет текста)")
        if not run["constraint"]["ok"]:
            issues_str = "; ".join(run["constraint"].get("issues", []))
            if issues_str:
                lines.append(f"⚠️ {issues_str}")

    lines.append(f"\n🧑‍⚖️ Judge: {judge_verdict} {judge_icon}  |  confidence: {judge_conf:.2f}")
    if judge_issues:
        lines.append("Замечания судьи: " + "; ".join(judge_issues))

    lines.append(f"Статус формата: {result['status']} {status_icon}  |  разброс длин: {cv_pct}%")
    lines.append(f"⏱ {result['latency_ms'] / 1000:.1f} сек  |  токены: ~{result['total_tokens']}")

    await safe_reply_text(update, "\n".join(lines))


async def iq_eval_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/iq_eval — оценка 10 eval-примеров из Дня 6 через constraint-check."""
    if not update.message:
        return

    await update.message.chat.send_action("typing")

    try:
        result = iq.eval_baseline_quality()
    except FileNotFoundError as e:
        await safe_reply_text(update, str(e))
        return
    except Exception as e:
        logger.exception("iq_eval_cmd error: %s", e)
        await safe_reply_text(update, "Ошибка при оценке baseline. Попробуйте позже.")
        return

    total = result["total"]
    passed = result["passed"]
    failed = result["failed"]
    rate = result["pass_rate"]

    lines = [
        f"📊 Eval baseline (День 6): {total} примеров",
        f"  ✅ Прошли constraint-check: {passed}/{total} ({rate}%)",
        f"  ❌ Не прошли: {failed}/{total}",
        "",
        "Детали:",
    ]

    for d in result["details"]:
        ok_icon = "✅" if d["ok"] else "❌"
        length_info = f"{d['body_length']} симв." if d["body_length"] else "нет тела"
        issues_str = "; ".join(d["issues"]) if d["issues"] else "ок"
        lines.append(f"  {ok_icon} #{d['example_id']}: {length_info} — {issues_str}")

    await safe_reply_text(update, "\n".join(lines))


async def iq_stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/iq_stats — статистика текущей сессии."""
    if not update.message:
        return

    stats = iq.get_session_stats()
    total = stats["total"]

    if total == 0:
        await safe_reply_text(update, "Статистика пуста — команды /iq_post и /iq_compare ещё не использовались.")
        return

    def pct(n: int) -> str:
        return f"{round(n / total * 100)}%" if total else "—"

    lines = [
        "📈 Статистика сессии:",
        f"  Всего запросов: {total}",
        f"  ✅ OK:     {stats['ok']} ({pct(stats['ok'])})",
        f"  ⚠️ UNSURE: {stats['unsure']} ({pct(stats['unsure'])})",
        f"  ❌ FAIL:   {stats['fail']} ({pct(stats['fail'])})",
        f"  Повторных inference: {stats['retried']}",
        f"  Токены: ~{stats['total_tokens']}  |  Среднее время: {stats['avg_latency_ms']} мс",
    ]

    await safe_reply_text(update, "\n".join(lines))
