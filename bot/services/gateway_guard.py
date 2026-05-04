"""Input/output guard для LLM Gateway — День 13.

Обнаруживает чувствительные данные в тексте промпта и ответа LLM.
Паттерны: API-ключи, email, телефон, банковские карты, base64-секреты.
"""

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Finding:
    """Одно обнаруженное вхождение чувствительных данных."""
    type: str    # "aws_key" | "openai_key" | "github_token" | "email" | "phone" | "card" | "base64"
    value: str   # оригинальное значение
    start: int   # начало в исходном тексте
    end: int     # конец в исходном тексте


# ─── Input guard patterns ─────────────────────────────────────────────────────

_INPUT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("aws_key",       re.compile(r"AKIA[0-9A-Z]{16}")),
    ("openai_key",    re.compile(r"sk-[a-zA-Z0-9_\-]{20,}")),
    ("github_token",  re.compile(r"ghp_[a-zA-Z0-9]{36}")),
    ("email",         re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")),
    ("phone",         re.compile(r"\+7[\s\-\(\)0-9]{10,14}")),
    # base64: минимум 40 символов высокой энтропии; не проверяем JWT здесь —
    # покрывается тем же паттерном
    ("base64",        re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")),
]

# Карты обрабатываются отдельно через Luhn; ищем 16-значные числа (с пробелами/-)
_CARD_RE = re.compile(r"\b(?:\d[ \-]?){15}\d\b")


# ─── Output guard patterns ────────────────────────────────────────────────────

# Те же секретные паттерны + специфика выходного контента
_OUTPUT_EXTRA_PATTERNS: list[tuple[str, re.Pattern]] = [
    # URL с IP-адресом (потенциально вредоносный)
    ("suspicious_url", re.compile(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")),
    # Признаки утечки системного промпта
    ("system_prompt_leak", re.compile(
        r"(?i)(you are a helpful|my system prompt|my instructions (say|are)|"
        r"system: you|<system>|忽略之前的)",
    )),
    # Шелл-команды с явным деструктивным характером
    ("shell_cmd", re.compile(r"(?i)(rm\s+-rf|curl\s+.+\|\s*bash|wget\s+.+\|\s*sh)")),
]


# ─── Luhn algorithm ───────────────────────────────────────────────────────────

def luhn_valid(number: str) -> bool:
    """Проверяет номер карты по алгоритму Луна."""
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# ─── Guard functions ──────────────────────────────────────────────────────────

def _apply_patterns(text: str, patterns: list[tuple[str, re.Pattern]]) -> list[Finding]:
    """Применяет список паттернов к тексту и возвращает список Finding."""
    findings: list[Finding] = []
    for ftype, pattern in patterns:
        for m in pattern.finditer(text):
            findings.append(Finding(
                type=ftype,
                value=m.group(),
                start=m.start(),
                end=m.end(),
            ))
    return findings


def input_guard(text: str) -> list[Finding]:
    """Ищет чувствительные данные во входящем промпте.

    Возвращает список Finding. Пустой список = чисто.
    """
    findings = _apply_patterns(text, _INPUT_PATTERNS)

    # Карты — только если Luhn проходит
    for m in _CARD_RE.finditer(text):
        if luhn_valid(m.group()):
            findings.append(Finding(
                type="card",
                value=m.group(),
                start=m.start(),
                end=m.end(),
            ))

    # Сортируем по позиции для предсказуемого порядка
    findings.sort(key=lambda f: f.start)
    logger.debug("input_guard: %d findings", len(findings))
    return findings


def output_guard(text: str) -> list[Finding]:
    """Ищет проблемы в ответе LLM: секреты, утечки промпта, подозрительные URL.

    Возвращает список Finding. Пустой список = чисто.
    """
    # Те же паттерны, что и для input (LLM мог галлюцинировать секреты)
    findings = _apply_patterns(text, _INPUT_PATTERNS)

    # Карты
    for m in _CARD_RE.finditer(text):
        if luhn_valid(m.group()):
            findings.append(Finding(
                type="card",
                value=m.group(),
                start=m.start(),
                end=m.end(),
            ))

    # Специфика вывода
    findings += _apply_patterns(text, _OUTPUT_EXTRA_PATTERNS)

    findings.sort(key=lambda f: f.start)
    logger.debug("output_guard: %d findings", len(findings))
    return findings
