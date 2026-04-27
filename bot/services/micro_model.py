"""Micro-model first — День 10.

ROUTING_SMALL_MODEL классифицирует входное описание главы (CLEAR / VAGUE / SPOILER_RISK).
При UNSURE или уверенности ниже порога — переклассификация через ROUTING_LARGE_MODEL.
Выбранная метка направляет на monolithic_post() или multistage_post() из Day 9.
"""

import json
import logging
import time
from dataclasses import dataclass

from .llm import call_llm_raw
from .multi_stage import monolithic_post, multistage_post, MonolithicResult, MultiStageResult
from ..config import ROUTING_SMALL_MODEL, ROUTING_LARGE_MODEL, MICRO_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

_MICRO_SYSTEM = (
    "Ты — классификатор описаний глав книги «Арморн — Сталь вместо заклинаний».\n"
    "Определи качество описания для генерации VK-анонса.\n"
    "CLEAR — конкретное описание, достаточно деталей, нет спойлеров.\n"
    "VAGUE — расплывчатое, мало деталей или слишком короткое.\n"
    "SPOILER_RISK — раскрывает ключевые сюжетные повороты.\n"
    "Верни строгий JSON без markdown-обёртки."
)

_micro_stats: dict = {
    "total": 0,
    "micro_handled": 0,
    "fallback_used": 0,
    "strategy_monolithic": 0,
    "strategy_multistage": 0,
    "total_tokens": 0,
    "latencies_ms": [],
}


@dataclass
class MicroResult:
    classification: dict                                 # label, confidence, status, reason
    fallback_used: bool                                  # была ли переклассификация через большую модель
    strategy_used: str                                   # "monolithic" | "multistage"
    post_result: MonolithicResult | MultiStageResult
    micro_tokens: int                                    # токены шага классификации (micro)
    total_tokens: int
    total_latency_ms: int


def _classify_input(user_prompt: str, model: str) -> tuple[dict, int]:
    """Один LLM-вызов. Возвращает (classification_dict, tokens_used)."""
    fallback = {
        "label": "VAGUE",
        "confidence": 0.5,
        "status": "UNSURE",
        "reason": "не удалось классифицировать",
    }
    messages = [
        {"role": "system", "content": _MICRO_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Описание главы: {user_prompt}\n\n"
                "Классифицируй (без пояснений):\n"
                '{"label": "CLEAR|VAGUE|SPOILER_RISK", '
                '"confidence": 0.0-1.0, '
                '"status": "OK|UNSURE", '
                '"reason": "до 80 символов"}'
            ),
        },
    ]
    try:
        raw = call_llm_raw(messages, temperature=0.1, model=model)
        if not raw:
            return fallback, 0
        text = raw["choices"][0]["message"]["content"]
        tokens = raw.get("usage", {}).get("total_tokens", 0)
        clean = text.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        try:
            parsed = json.loads(clean.strip())
        except json.JSONDecodeError:
            logger.warning("_classify_input: JSON parse failed for model %s, using fallback", model)
            return fallback, tokens
        return parsed, tokens
    except Exception as e:
        logger.exception("_classify_input error (model=%s): %s", model, e)
        return fallback, 0


def _validate_classification(cls: dict) -> bool:
    """Проверяет корректность структуры classification dict."""
    if cls.get("label") not in {"CLEAR", "VAGUE", "SPOILER_RISK"}:
        return False
    if cls.get("status") not in {"OK", "UNSURE"}:
        return False
    try:
        conf = float(cls.get("confidence", -1))
        if not (0.0 <= conf <= 1.0):
            return False
    except (TypeError, ValueError):
        return False
    return True


def _choose_strategy(label: str) -> str:
    """CLEAR → monolithic. VAGUE / SPOILER_RISK → multistage."""
    return "monolithic" if label == "CLEAR" else "multistage"


def micro_first_post(user_prompt: str) -> MicroResult:
    """Полный pipeline: классификация входа → выбор стратегии → генерация поста."""
    t = time.perf_counter()

    # Шаг 1: micro-model (ROUTING_SMALL_MODEL)
    micro_cls, micro_tokens = _classify_input(user_prompt, ROUTING_SMALL_MODEL)
    valid = _validate_classification(micro_cls)
    confidence = float(micro_cls.get("confidence", 0.0))
    status = micro_cls.get("status", "UNSURE")
    fallback_used = False
    fallback_tokens = 0

    # Шаг 2: fallback через ROUTING_LARGE_MODEL если UNSURE или низкая уверенность
    if not valid or status == "UNSURE" or confidence < MICRO_CONFIDENCE_THRESHOLD:
        fallback_used = True
        fallback_cls, fallback_tokens = _classify_input(user_prompt, ROUTING_LARGE_MODEL)
        if _validate_classification(fallback_cls):
            micro_cls = fallback_cls

    label = micro_cls.get("label", "VAGUE")
    strategy = _choose_strategy(label)

    # Шаг 3: генерация поста выбранной стратегией
    if strategy == "monolithic":
        post_result = monolithic_post(user_prompt)
        gen_tokens = post_result.tokens_used
    else:
        post_result = multistage_post(user_prompt)
        gen_tokens = post_result.total_tokens

    total_latency_ms = int((time.perf_counter() - t) * 1000)
    total_tokens = micro_tokens + fallback_tokens + gen_tokens

    _micro_stats["total"] += 1
    if fallback_used:
        _micro_stats["fallback_used"] += 1
    else:
        _micro_stats["micro_handled"] += 1
    if strategy == "monolithic":
        _micro_stats["strategy_monolithic"] += 1
    else:
        _micro_stats["strategy_multistage"] += 1
    _micro_stats["total_tokens"] += total_tokens
    _micro_stats["latencies_ms"].append(total_latency_ms)

    return MicroResult(
        classification=micro_cls,
        fallback_used=fallback_used,
        strategy_used=strategy,
        post_result=post_result,
        micro_tokens=micro_tokens,
        total_tokens=total_tokens,
        total_latency_ms=total_latency_ms,
    )


def get_micro_stats() -> dict:
    total = _micro_stats["total"]
    latencies = _micro_stats["latencies_ms"]
    avg_latency_ms = round(sum(latencies) / len(latencies)) if latencies else 0
    return {
        "total": total,
        "micro_handled": _micro_stats["micro_handled"],
        "fallback_used": _micro_stats["fallback_used"],
        "strategy_monolithic": _micro_stats["strategy_monolithic"],
        "strategy_multistage": _micro_stats["strategy_multistage"],
        "total_tokens": _micro_stats["total_tokens"],
        "avg_latency_ms": avg_latency_ms,
    }
