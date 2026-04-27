"""Multi-stage inference — День 9.

Сравнение двух подходов к генерации VK-анонсов:
- Monolithic: один большой запрос → готовый пост
- Multi-stage: анализ → стратегия → генерация (три коротких строгих вызова)
"""

import json
import logging
import time
from dataclasses import dataclass, field

from .llm import call_llm, call_llm_raw
from .inference_quality import VK_SYSTEM_PROMPT, check_constraints
from ..config import ROUTING_SMALL_MODEL, OPENROUTER_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Системные промпты этапов
# ---------------------------------------------------------------------------

_ANALYSIS_SYSTEM = (
    "Ты — контент-аналитик книжного SMM.\n"
    "Извлеки структуру описания главы.\n"
    "Верни строгий JSON без markdown-обёртки."
)

_STRATEGY_SYSTEM = (
    "Ты — редактор VK-постов о книге «Арморн — Сталь вместо заклинаний».\n"
    "На основе анализа главы прими решение о параметрах поста.\n"
    "Верни строгий JSON без markdown-обёртки."
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    stage: int
    output: dict | str      # dict для этапов 1–2, str для этапа 3
    raw_text: str
    tokens_used: int
    latency_ms: int


@dataclass
class MultiStageResult:
    stages: list[StageResult]   # [stage1, stage2, stage3]
    final_post: str
    constraint_check: dict
    total_tokens: int
    total_latency_ms: int


@dataclass
class MonolithicResult:
    post: str
    constraint_check: dict
    tokens_used: int
    latency_ms: int


@dataclass
class CompareResult:
    monolithic: MonolithicResult
    multistage: MultiStageResult
    total_latency_ms: int
    total_tokens: int


# ---------------------------------------------------------------------------
# Вспомогательные: парсинг JSON с fallback
# ---------------------------------------------------------------------------

def _parse_json(raw: str, fallback: dict) -> dict:
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    try:
        return json.loads(clean.strip())
    except json.JSONDecodeError:
        logger.warning("_parse_json: не удалось распарсить JSON, используем fallback")
        return fallback


# ---------------------------------------------------------------------------
# Этапы multi-stage
# ---------------------------------------------------------------------------

def _analyze_chapter(user_prompt: str) -> StageResult:
    """Этап 1: классификация — тон, риск спойлера, дуга, фокус."""
    t = time.perf_counter()
    fallback = {
        "tone": "atmospheric",
        "spoiler_risk": "medium",
        "chapter_arc": "action",
        "key_focus": user_prompt[:60],
    }
    messages = [
        {"role": "system", "content": _ANALYSIS_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Описание главы:\n{user_prompt}\n\n"
                "Верни JSON с анализом (без пояснений):\n"
                '{"tone": "tense|atmospheric|action|emotional|revelatory", '
                '"spoiler_risk": "low|medium|high", '
                '"chapter_arc": "setup|action|climax|revelation", '
                '"key_focus": "главный момент до 60 символов"}'
            ),
        },
    ]
    try:
        raw = call_llm_raw(messages, temperature=0.1, model=ROUTING_SMALL_MODEL)
        if not raw:
            return StageResult(1, fallback, "", 0, int((time.perf_counter() - t) * 1000))
        text = raw["choices"][0]["message"]["content"]
        tokens = raw.get("usage", {}).get("total_tokens", 0)
        output = _parse_json(text, fallback)
        if output.get("tone") not in ("tense", "atmospheric", "action", "emotional", "revelatory"):
            output["tone"] = fallback["tone"]
        if output.get("spoiler_risk") not in ("low", "medium", "high"):
            output["spoiler_risk"] = fallback["spoiler_risk"]
        if output.get("chapter_arc") not in ("setup", "action", "climax", "revelation"):
            output["chapter_arc"] = fallback["chapter_arc"]
        return StageResult(1, output, text, tokens, int((time.perf_counter() - t) * 1000))
    except Exception as e:
        logger.exception("_analyze_chapter error: %s", e)
        return StageResult(1, fallback, "", 0, int((time.perf_counter() - t) * 1000))


def _decide_strategy(stage1: StageResult) -> StageResult:
    """Этап 2: решение о параметрах поста на основе анализа."""
    t = time.perf_counter()
    fallback = {
        "length": "medium",
        "opening_type": "atmosphere",
        "style_hint": "сдержанный, атмосферный",
        "avoid": "",
    }
    messages = [
        {"role": "system", "content": _STRATEGY_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Анализ главы:\n{json.dumps(stage1.output, ensure_ascii=False)}\n\n"
                "Реши параметры VK-поста (без пояснений):\n"
                '{"length": "short|medium", '
                '"opening_type": "question|statement|atmosphere|action", '
                '"style_hint": "до 60 символов", '
                '"avoid": "что не упоминать, до 80 символов"}'
            ),
        },
    ]
    try:
        raw = call_llm_raw(messages, temperature=0.1, model=ROUTING_SMALL_MODEL)
        if not raw:
            return StageResult(2, fallback, "", 0, int((time.perf_counter() - t) * 1000))
        text = raw["choices"][0]["message"]["content"]
        tokens = raw.get("usage", {}).get("total_tokens", 0)
        output = _parse_json(text, fallback)
        if output.get("length") not in ("short", "medium"):
            output["length"] = fallback["length"]
        if output.get("opening_type") not in ("question", "statement", "atmosphere", "action"):
            output["opening_type"] = fallback["opening_type"]
        return StageResult(2, output, text, tokens, int((time.perf_counter() - t) * 1000))
    except Exception as e:
        logger.exception("_decide_strategy error: %s", e)
        return StageResult(2, fallback, "", 0, int((time.perf_counter() - t) * 1000))


def _generate_post(user_prompt: str, stage1: StageResult, stage2: StageResult) -> StageResult:
    """Этап 3: творческая генерация поста с явными параметрами."""
    t = time.perf_counter()
    a = stage1.output  # analysis
    s = stage2.output  # strategy

    length_hint = "160–250 символов тела" if s.get("length") == "short" else "300–400 символов тела"
    avoid_line = f"Не упоминай: {s['avoid']}." if s.get("avoid") else ""
    spoiler_note = "будь особенно осторожен" if a.get("spoiler_risk") == "high" else "пиши свободно"

    user_content = (
        f"Описание главы: {user_prompt}\n\n"
        f"Параметры (уже определены):\n"
        f"- Тон: {a.get('tone', '—')}\n"
        f"- Риск спойлера: {a.get('spoiler_risk', '—')} — {spoiler_note}\n"
        f"- Фокус: {a.get('key_focus', '—')}\n"
        f"- Длина тела: {length_hint}\n"
        f"- Тип открытия: {s.get('opening_type', '—')}\n"
        f"- Стиль: {s.get('style_hint', '—')}\n"
        f"{avoid_line}\n\n"
        "Напиши VK-анонс строго по параметрам. Отвечай ТОЛЬКО текстом поста."
    )
    messages = [
        {"role": "system", "content": VK_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    try:
        raw = call_llm_raw(messages, temperature=0.7, model=OPENROUTER_MODEL)
        if not raw:
            return StageResult(3, "", "", 0, int((time.perf_counter() - t) * 1000))
        text = raw["choices"][0]["message"]["content"].strip()
        tokens = raw.get("usage", {}).get("total_tokens", 0)
        return StageResult(3, text, text, tokens, int((time.perf_counter() - t) * 1000))
    except Exception as e:
        logger.exception("_generate_post error: %s", e)
        return StageResult(3, "", "", 0, int((time.perf_counter() - t) * 1000))


# ---------------------------------------------------------------------------
# Публичные точки входа
# ---------------------------------------------------------------------------

def monolithic_post(user_prompt: str) -> MonolithicResult:
    """Один большой запрос → готовый VK-пост."""
    t = time.perf_counter()
    messages = [
        {"role": "system", "content": VK_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{user_prompt}\n\n"
                "Определи тон и риск спойлеров, затем напиши анонс. "
                "Отвечай ТОЛЬКО текстом поста."
            ),
        },
    ]
    try:
        raw = call_llm_raw(messages, temperature=0.7, model=OPENROUTER_MODEL)
        if not raw:
            return MonolithicResult("", {"ok": False, "issues": ["Нет ответа"]}, 0, int((time.perf_counter() - t) * 1000))
        text = raw["choices"][0]["message"]["content"].strip()
        tokens = raw.get("usage", {}).get("total_tokens", 0)
        constraint = check_constraints(text)
        return MonolithicResult(
            post=text,
            constraint_check=constraint,
            tokens_used=tokens,
            latency_ms=int((time.perf_counter() - t) * 1000),
        )
    except Exception as e:
        logger.exception("monolithic_post error: %s", e)
        return MonolithicResult("", {"ok": False, "issues": [str(e)]}, 0, int((time.perf_counter() - t) * 1000))


def multistage_post(user_prompt: str) -> MultiStageResult:
    """3 этапа: анализ → стратегия → генерация."""
    t = time.perf_counter()
    stage1 = _analyze_chapter(user_prompt)
    stage2 = _decide_strategy(stage1)
    stage3 = _generate_post(user_prompt, stage1, stage2)

    final_post = stage3.output if isinstance(stage3.output, str) else ""
    constraint = check_constraints(final_post)
    total_tokens = stage1.tokens_used + stage2.tokens_used + stage3.tokens_used

    return MultiStageResult(
        stages=[stage1, stage2, stage3],
        final_post=final_post,
        constraint_check=constraint,
        total_tokens=total_tokens,
        total_latency_ms=int((time.perf_counter() - t) * 1000),
    )


def compare_posts(user_prompt: str) -> CompareResult:
    """Запускает оба подхода и возвращает сравнение."""
    t = time.perf_counter()
    mono = monolithic_post(user_prompt)
    multi = multistage_post(user_prompt)
    return CompareResult(
        monolithic=mono,
        multistage=multi,
        total_latency_ms=int((time.perf_counter() - t) * 1000),
        total_tokens=mono.tokens_used + multi.total_tokens,
    )
