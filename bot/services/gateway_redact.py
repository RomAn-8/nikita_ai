"""Redact/restore логика для LLM Gateway — День 13.

redact_text() — заменяет найденные секреты на плейсхолдеры [REDACTED_TYPE_N].
restore_text() — подставляет оригинальные значения обратно.

Маппинг хранится только в user_data и никогда не попадает в audit log.
"""

import logging
from .gateway_guard import Finding

logger = logging.getLogger(__name__)


def redact_text(text: str, findings: list[Finding]) -> tuple[str, dict[str, str]]:
    """Заменяет секреты в тексте на плейсхолдеры.

    Сортирует findings по позиции в обратном порядке,
    чтобы замены не смещали смежные индексы.

    Возвращает (redacted_text, mapping), где mapping = {placeholder: original_value}.
    """
    if not findings:
        return text, {}

    # Обратная сортировка по start — заменяем с конца
    sorted_findings = sorted(findings, key=lambda f: f.start, reverse=True)

    mapping: dict[str, str] = {}
    counter = 0
    result = text

    for finding in sorted_findings:
        counter += 1
        placeholder = f"[REDACTED_{finding.type.upper()}_{counter}]"
        mapping[placeholder] = finding.value
        result = result[: finding.start] + placeholder + result[finding.end :]

    logger.debug("redact_text: %d replacements", counter)
    return result, mapping


def restore_text(text: str, mapping: dict[str, str]) -> str:
    """Заменяет плейсхолдеры обратно на оригинальные значения.

    Вызывается только когда output_guard вернул пустой список.
    """
    if not mapping:
        return text
    result = text
    for placeholder, original in mapping.items():
        result = result.replace(placeholder, original)
    logger.debug("restore_text: %d restorations", len(mapping))
    return result
