"""LLM Gateway — оркестратор Day 13.

Три режима:
  block          — при обнаружении секрета запрос отклоняется
  redact         — секреты заменяются плейсхолдерами, запрос уходит в LLM
  restore        — то же что redact, но если output guard чист — значения восстанавливаются

Дополнительно: rate limiting (5 req/60s), cost tracking через usage, audit JSONL.
"""

import time
import logging
from dataclasses import dataclass, field

from ..config import OPENROUTER_MODEL, GW_RATE_LIMIT, GW_RATE_WINDOW, MODEL_PRICES, MODEL_PRICE_DEFAULT
from .gateway_guard import Finding, input_guard, output_guard
from .gateway_redact import redact_text, restore_text
from .gateway_audit import log_event
from .llm import call_llm_raw

logger = logging.getLogger(__name__)

_GATEWAY_SYSTEM = (
    "Ты — полезный ассистент. Отвечай точно и по делу. "
    "Не повторяй инструкции пользователя в ответе."
)


@dataclass
class GatewayResult:
    response: str
    blocked: bool
    block_reason: str                          # "rate_limit" | "input_guard" | "output_guard" | ""
    input_findings: list[Finding]
    output_findings: list[Finding]
    redacted: bool
    restored: bool
    mode: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    audit_logged: bool
    sanitization_log: list[str] = field(default_factory=list)


def _compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Считает стоимость в USD на основе таблицы цен."""
    prices = MODEL_PRICES.get(model, MODEL_PRICE_DEFAULT)
    cost = (prompt_tokens / 1_000_000) * prices["input"]
    cost += (completion_tokens / 1_000_000) * prices["output"]
    return cost


def _check_rate_limit(user_data: dict) -> bool:
    """Проверяет rate limit. Возвращает True если лимит превышен."""
    now = time.monotonic()
    rate = user_data.get("gw_rate", {"count": 0, "window_start": now})

    if now - rate["window_start"] > GW_RATE_WINDOW:
        # Новое окно
        rate = {"count": 0, "window_start": now}

    if rate["count"] >= GW_RATE_LIMIT:
        return True

    rate["count"] += 1
    user_data["gw_rate"] = rate
    return False


def _update_session_stats(user_data: dict, result: GatewayResult) -> None:
    stats = user_data.setdefault("gw_session_stats", {
        "requests": 0, "blocked": 0, "redacted": 0,
        "findings_total": 0, "cost_usd": 0.0,
    })
    stats["requests"] += 1
    if result.blocked:
        stats["blocked"] += 1
    if result.redacted:
        stats["redacted"] += 1
    stats["findings_total"] += len(result.input_findings)
    stats["cost_usd"] = round(stats["cost_usd"] + result.cost_usd, 8)


def run_gateway(
    prompt: str,
    mode: str,
    user_id: int,
    user_data: dict,
    model: str | None = None,
) -> GatewayResult:
    """Основной entry point Gateway.

    Args:
        prompt: Текст от пользователя
        mode: "block" | "redact" | "restore"
        user_id: Telegram user ID для rate limiting и audit
        user_data: Словарь user_data из context (хранит маппинг и rate state)
    """
    slog: list[str] = []

    # ── Rate limit ────────────────────────────────────────────────────────────
    if _check_rate_limit(user_data):
        result = GatewayResult(
            response=f"Превышен лимит запросов ({GW_RATE_LIMIT} req/{GW_RATE_WINDOW}s). Подожди немного.",
            blocked=True,
            block_reason="rate_limit",
            input_findings=[],
            output_findings=[],
            redacted=False,
            restored=False,
            mode=mode,
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
            audit_logged=False,
            sanitization_log=["rate_limit: exceeded"],
        )
        _update_session_stats(user_data, result)
        return result

    # ── Input guard ───────────────────────────────────────────────────────────
    in_findings = input_guard(prompt)
    slog.append(
        f"input_guard: {len(in_findings)} finding(s): {[f.type for f in in_findings]}"
        if in_findings else "input_guard: чисто"
    )

    # Режим block — отклоняем при любом обнаружении
    if mode == "block" and in_findings:
        types = ", ".join(f.type for f in in_findings)
        result = GatewayResult(
            response=f"Запрос заблокирован. Обнаружены чувствительные данные: {types}.",
            blocked=True,
            block_reason="input_guard",
            input_findings=in_findings,
            output_findings=[],
            redacted=False,
            restored=False,
            mode=mode,
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
            audit_logged=False,
            sanitization_log=slog,
        )
        log_event(
            user_id=user_id, mode=mode,
            input_findings=in_findings, output_findings=[],
            redacted=False, restored=False, blocked=True,
            prompt_tokens=0, completion_tokens=0, cost_usd=0.0,
        )
        result.audit_logged = True
        _update_session_stats(user_data, result)
        return result

    # Режим redact/restore — заменяем секреты плейсхолдерами
    llm_prompt = prompt
    mapping: dict[str, str] = {}
    redacted = False

    if mode in ("redact", "restore") and in_findings:
        llm_prompt, mapping = redact_text(prompt, in_findings)
        user_data["gw_redact_map"] = mapping
        redacted = True
        slog.append(f"redact: {len(mapping)} placeholder(s) inserted")
    else:
        user_data.pop("gw_redact_map", None)

    # ── LLM call ──────────────────────────────────────────────────────────────
    messages = [
        {"role": "system", "content": _GATEWAY_SYSTEM},
        {"role": "user", "content": llm_prompt},
    ]

    prompt_tokens = 0
    completion_tokens = 0
    cost_usd = 0.0
    response_text = ""

    try:
        raw = call_llm_raw(messages, temperature=0.3, model=model)
        if raw:
            choices = raw.get("choices", [])
            if choices:
                response_text = choices[0].get("message", {}).get("content", "") or ""
            usage = raw.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            model = raw.get("model", OPENROUTER_MODEL)
            cost_usd = _compute_cost(model, prompt_tokens, completion_tokens)
        slog.append(f"llm: {prompt_tokens}p + {completion_tokens}c tokens, ${cost_usd:.6f}")
    except Exception as e:
        logger.exception("gateway run_gateway LLM call failed: %s", e)
        response_text = "Ошибка при обращении к LLM. Попробуйте позже."

    # ── Output guard ──────────────────────────────────────────────────────────
    out_findings = output_guard(response_text)
    slog.append(
        f"output_guard: {len(out_findings)} finding(s): {[f.type for f in out_findings]}"
        if out_findings else "output_guard: чисто"
    )

    if out_findings:
        types = ", ".join(f.type for f in out_findings)
        response_text = (
            f"⚠️ Ответ заблокирован output guard.\n"
            f"Обнаружены потенциально опасные данные в ответе LLM: {types}."
        )

    # ── Restore ───────────────────────────────────────────────────────────────
    restored = False
    if mode == "restore" and redacted and not out_findings and mapping:
        response_text = restore_text(response_text, mapping)
        restored = True
        slog.append("restore: оригинальные значения подставлены")

    # ── Audit ─────────────────────────────────────────────────────────────────
    blocked_final = bool(out_findings)
    log_event(
        user_id=user_id, mode=mode,
        input_findings=in_findings, output_findings=out_findings,
        redacted=redacted, restored=restored, blocked=blocked_final,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        cost_usd=cost_usd,
    )

    result = GatewayResult(
        response=response_text,
        blocked=blocked_final,
        block_reason="output_guard" if blocked_final else "",
        input_findings=in_findings,
        output_findings=out_findings,
        redacted=redacted,
        restored=restored,
        mode=mode,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        audit_logged=True,
        sanitization_log=slog,
    )
    _update_session_stats(user_data, result)
    return result
