"""Inference Quality Control — День 7.

Контроль уверенности при генерации VK-анонсов глав «Арморн».
Подходы: constraint-based, self-check, redundancy, scoring.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from statistics import mean, stdev

from .llm import call_llm, call_llm_raw
from ..config import IQ_MAX_RETRIES, IQ_CONFIDENCE_THRESHOLD, IQ_REDUNDANCY_N, FT_DATA_PATH

logger = logging.getLogger(__name__)

BOOK_LINK = "author.today/reader/566381"
LINK_MARKER = "Приятного чтения:"
BODY_MIN_LEN = 150
BODY_MAX_LEN = 550

# Тот же system_prompt, что в finetune/data/eval.jsonl (День 6)
VK_SYSTEM_PROMPT = (
    "Ты — SMM-автор книги «Арморн — Сталь вместо заклинаний».\n"
    "Пишешь короткие анонсы глав для ВКонтакте.\n\n"
    "Стиль: сдержанный, атмосферный, без лишнего пафоса.\n"
    "Без спойлеров — только интрига и настроение главы.\n"
    f"Длина: {BODY_MIN_LEN}–{BODY_MAX_LEN} символов текста без ссылки.\n"
    f"Ссылку {BOOK_LINK} добавляй в конце через «{LINK_MARKER}».\n"
    "Хэштеги и эмодзи — по усмотрению, для финальных или особых глав допустимы."
)

_SELF_CHECK_SYSTEM = (
    "Ты — строгий редактор VK-постов о книге «Арморн — Сталь вместо заклинаний».\n"
    "Оцени анонс главы. Верни ответ строго в JSON без markdown-обёртки."
)

_JUDGE_SYSTEM = (
    "Ты — редактор серии «Арморн — Сталь вместо заклинаний».\n"
    "Сравни несколько вариантов VK-анонса одной главы.\n"
    "Оцени, насколько они согласованы по смыслу, стилю и содержанию.\n"
    "Верни ответ строго в JSON без markdown-обёртки."
)

# Глобальная статистика сессии (сбрасывается при рестарте)
_session_stats: dict = {
    "total": 0,
    "ok": 0,
    "unsure": 0,
    "fail": 0,
    "retried": 0,
    "total_tokens": 0,
    "latencies_ms": [],
}


@dataclass
class InferenceResult:
    text: str
    status: str                        # "OK" | "UNSURE" | "FAIL"
    confidence: float                  # 0.0–1.0
    checks: dict = field(default_factory=dict)
    retries: int = 0
    tokens_used: int = 0
    latency_ms: int = 0


# ---------------------------------------------------------------------------
# Constraint-based (без LLM)
# ---------------------------------------------------------------------------

def check_constraints(text: str) -> dict:
    """Проверяет формат VK-поста: длина тела и наличие ссылки."""
    issues: list[str] = []

    has_link_marker = LINK_MARKER in text
    has_link = BOOK_LINK in text

    if has_link_marker:
        body = text[: text.index(LINK_MARKER)].strip()
    else:
        body = text.strip()

    body_len = len(body)

    if not has_link_marker:
        issues.append(f"Нет маркера «{LINK_MARKER}»")
    if not has_link:
        issues.append(f"Нет ссылки {BOOK_LINK}")
    if body_len < BODY_MIN_LEN:
        issues.append(f"Тело слишком короткое: {body_len} симв. (минимум {BODY_MIN_LEN})")
    elif body_len > BODY_MAX_LEN:
        issues.append(f"Тело слишком длинное: {body_len} симв. (максимум {BODY_MAX_LEN})")

    return {
        "ok": len(issues) == 0,
        "body_length": body_len,
        "has_link": has_link and has_link_marker,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Self-check (один дополнительный LLM-вызов)
# ---------------------------------------------------------------------------

def self_check(system: str, user: str, answer: str, model: str | None = None) -> dict:
    """Просит модель оценить собственный ответ. Возвращает verdict/confidence/issues."""
    checker_user = (
        f"Системный промпт автора:\n{system}\n\n"
        f"Задание:\n{user}\n\n"
        f"Созданный анонс:\n{answer}\n\n"
        "Оцени анонс по трём критериям:\n"
        "1. Нет явных спойлеров (исходы битв, гибель персонажей, раскрытие секретов)\n"
        "2. Стиль сдержанный, атмосферный, без пафоса\n"
        "3. Читается как интрига, а не пересказ\n\n"
        "Верни строго JSON (без ```markdown```), пример:\n"
        '{"verdict": "OK", "confidence": 0.9, "issues": []}'
    )
    messages = [
        {"role": "system", "content": _SELF_CHECK_SYSTEM},
        {"role": "user", "content": checker_user},
    ]
    try:
        raw_text = call_llm(messages, temperature=0.1, model=model)
        if not raw_text:
            return {"verdict": "UNSURE", "confidence": 0.5, "issues": ["Self-check не вернул ответ"]}

        # Пробуем распарсить JSON, удаляя возможные markdown-обёртки
        clean = raw_text.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        result = json.loads(clean.strip())

        verdict = result.get("verdict", "UNSURE")
        if verdict not in ("OK", "UNSURE", "FAIL"):
            verdict = "UNSURE"
        confidence = float(result.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        issues = result.get("issues", [])
        return {"verdict": verdict, "confidence": confidence, "issues": issues}

    except json.JSONDecodeError:
        logger.warning("self_check: не удалось распарсить JSON ответ")
        return {"verdict": "UNSURE", "confidence": 0.4, "issues": ["Self-check: не JSON ответ"]}
    except Exception as e:
        logger.exception("self_check error: %s", e)
        return {"verdict": "UNSURE", "confidence": 0.4, "issues": [f"Self-check ошибка: {e}"]}


# ---------------------------------------------------------------------------
# Redundancy judge: LLM сравнивает N текстов по смыслу, стилю, спойлерам
# ---------------------------------------------------------------------------

def _judge_redundancy(texts: list[str], model: str | None = None) -> dict:
    """Один LLM-вызов: сравнивает тексты между собой. Возвращает verdict/confidence/issues."""
    non_empty = [t for t in texts if t and t.strip()]
    if not non_empty:
        return {"verdict": "UNSURE", "confidence": 0.0, "issues": ["Нет текстов для сравнения"]}

    numbered = "\n\n".join(f"Вариант {i + 1}:\n{t}" for i, t in enumerate(non_empty))
    judge_user = (
        f"Три варианта анонса одной главы:\n\n{numbered}\n\n"
        "Оцени, насколько варианты согласованы:\n"
        "1. Похож ли общий смысл и интрига (нет ли расхождений)\n"
        "2. Одинаков ли стиль (сдержанный, без пафоса)\n"
        "3. Нет ли спойлера только в одном из вариантов\n"
        "4. Нет ли сильного расхождения по содержанию\n\n"
        "Верни строго JSON (без ```markdown```):\n"
        '{"verdict": "CONSISTENT", "confidence": 0.9, "issues": []}'
    )
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": judge_user},
    ]
    try:
        raw_text = call_llm(messages, temperature=0.1, model=model)
        if not raw_text:
            return {"verdict": "UNSURE", "confidence": 0.5, "issues": ["Judge не вернул ответ"]}

        clean = raw_text.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        result = json.loads(clean.strip())

        verdict = result.get("verdict", "UNSURE")
        if verdict not in ("CONSISTENT", "UNSURE", "DIVERGENT"):
            verdict = "UNSURE"
        confidence = float(result.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        issues = result.get("issues", [])
        return {"verdict": verdict, "confidence": confidence, "issues": issues}

    except json.JSONDecodeError:
        logger.warning("_judge_redundancy: не удалось распарсить JSON")
        return {"verdict": "UNSURE", "confidence": 0.4, "issues": ["Judge: не JSON ответ"]}
    except Exception as e:
        logger.exception("_judge_redundancy error: %s", e)
        return {"verdict": "UNSURE", "confidence": 0.4, "issues": [f"Judge ошибка: {e}"]}


# ---------------------------------------------------------------------------
# Scoring: объединяет constraint + self_check → финальный статус
# ---------------------------------------------------------------------------

def _compute_status(constraint: dict, sc: dict) -> tuple[str, float]:
    if not constraint["ok"]:
        return "FAIL", 0.0
    if sc["verdict"] == "FAIL":
        return "FAIL", 0.1
    if sc["verdict"] == "UNSURE":
        return "UNSURE", min(sc.get("confidence", 0.5), 0.65)
    return "OK", max(sc.get("confidence", 0.8), 0.75)


# ---------------------------------------------------------------------------
# Основной pipeline: generate → constraint → self-check → retry
# ---------------------------------------------------------------------------

def generate_with_confidence(
    user_prompt: str,
    max_retries: int = IQ_MAX_RETRIES,
) -> InferenceResult:
    """Генерирует VK-анонс с контролем quality. Constraint-based + self-check + scoring."""
    t_start = time.perf_counter()
    total_tokens = 0
    attempt = 0
    last_result: InferenceResult | None = None

    messages = [
        {"role": "system", "content": VK_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    while attempt <= max_retries:
        try:
            raw = call_llm_raw(messages, temperature=0.7)
            if not raw:
                attempt += 1
                continue

            text = raw["choices"][0]["message"]["content"].strip()
            tokens = raw.get("usage", {}).get("total_tokens", 0)
            total_tokens += tokens

            constraint = check_constraints(text)

            # Если constraint не прошёл и есть ещё попытки — сразу retry
            if not constraint["ok"] and attempt < max_retries:
                attempt += 1
                continue

            sc = self_check(VK_SYSTEM_PROMPT, user_prompt, text)
            total_tokens += raw.get("usage", {}).get("completion_tokens", 0)  # approx

            status, confidence = _compute_status(constraint, sc)

            last_result = InferenceResult(
                text=text,
                status=status,
                confidence=confidence,
                checks={"constraint": constraint, "self_check": sc},
                retries=attempt,
                tokens_used=total_tokens,
                latency_ms=int((time.perf_counter() - t_start) * 1000),
            )

            # Если confidence выше порога — принимаем
            if confidence >= IQ_CONFIDENCE_THRESHOLD:
                break

            # Иначе retry если есть попытки
            if attempt < max_retries:
                attempt += 1
                continue
            break

        except Exception as e:
            logger.exception("generate_with_confidence attempt %d error: %s", attempt, e)
            attempt += 1

    if last_result is None:
        last_result = InferenceResult(
            text="",
            status="FAIL",
            confidence=0.0,
            checks={"error": "Все попытки провалились"},
            retries=attempt,
            tokens_used=total_tokens,
            latency_ms=int((time.perf_counter() - t_start) * 1000),
        )

    _update_stats(last_result)
    return last_result


# ---------------------------------------------------------------------------
# Redundancy: N прогонов, сравнение стабильности
# ---------------------------------------------------------------------------

def redundancy_check(
    user_prompt: str,
    n: int = IQ_REDUNDANCY_N,
) -> dict:
    """Запускает N прогонов и проверяет стабильность результатов."""
    messages = [
        {"role": "system", "content": VK_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    runs: list[dict] = []
    total_tokens = 0
    t_start = time.perf_counter()

    for i in range(n):
        try:
            raw = call_llm_raw(messages, temperature=0.7)
            if not raw:
                runs.append({"text": "", "constraint": {"ok": False, "issues": ["Нет ответа"]}, "length": 0})
                continue
            text = raw["choices"][0]["message"]["content"].strip()
            total_tokens += raw.get("usage", {}).get("total_tokens", 0)
            constraint = check_constraints(text)
            # Длина тела для сравнения
            if LINK_MARKER in text:
                body_len = len(text[: text.index(LINK_MARKER)].strip())
            else:
                body_len = len(text)
            runs.append({"text": text, "constraint": constraint, "length": body_len})
        except Exception as e:
            logger.exception("redundancy_check run %d error: %s", i, e)
            runs.append({"text": "", "constraint": {"ok": False, "issues": [str(e)]}, "length": 0})

    constraint_pass = [r["constraint"]["ok"] for r in runs]
    lengths = [r["length"] for r in runs if r["length"] > 0]

    # Коэффициент вариации длин: std/mean
    if len(lengths) >= 2:
        cv = stdev(lengths) / mean(lengths) if mean(lengths) > 0 else 0.0
    else:
        cv = 0.0

    all_pass = all(constraint_pass)
    consistent = all_pass and cv < 0.30

    status = "CONSISTENT" if consistent else ("UNSTABLE" if not all_pass else "VARIABLE")

    judge = _judge_redundancy([r["text"] for r in runs])

    return {
        "runs": runs,
        "n": n,
        "constraint_pass": constraint_pass,
        "pass_count": sum(constraint_pass),
        "lengths": lengths,
        "length_variation_pct": round(cv * 100),
        "consistent": consistent,
        "status": status,
        "judge": judge,
        "total_tokens": total_tokens,
        "latency_ms": int((time.perf_counter() - t_start) * 1000),
    }


# ---------------------------------------------------------------------------
# Связь с Днём 6: проверка baseline через constraint-based
# ---------------------------------------------------------------------------

def eval_baseline_quality() -> dict:
    """Прогоняет baseline_results.jsonl через constraint-check. Связь с Днём 6."""
    baseline_path = FT_DATA_PATH / "baseline_results.jsonl"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Файл не найден: {baseline_path}")

    total = 0
    passed = 0
    details: list[dict] = []

    with open(baseline_path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue

            example_id = obj.get("example_id", total + 1)
            baseline_text = obj.get("baseline_response", "")
            constraint = check_constraints(baseline_text)
            total += 1
            if constraint["ok"]:
                passed += 1
            details.append({
                "example_id": example_id,
                "ok": constraint["ok"],
                "body_length": constraint.get("body_length", 0),
                "issues": constraint.get("issues", []),
            })

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total * 100) if total else 0,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Session statistics
# ---------------------------------------------------------------------------

def get_session_stats() -> dict:
    """Возвращает статистику текущей сессии."""
    total = _session_stats["total"]
    latencies = _session_stats["latencies_ms"]
    avg_latency = round(sum(latencies) / len(latencies)) if latencies else 0
    return {
        "total": total,
        "ok": _session_stats["ok"],
        "unsure": _session_stats["unsure"],
        "fail": _session_stats["fail"],
        "retried": _session_stats["retried"],
        "total_tokens": _session_stats["total_tokens"],
        "avg_latency_ms": avg_latency,
    }


def _update_stats(result: InferenceResult) -> None:
    _session_stats["total"] += 1
    _session_stats[result.status.lower()] = _session_stats.get(result.status.lower(), 0) + 1
    if result.retries > 0:
        _session_stats["retried"] += result.retries
    _session_stats["total_tokens"] += result.tokens_used
    _session_stats["latencies_ms"].append(result.latency_ms)
