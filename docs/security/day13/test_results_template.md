# LLM Gateway — Test Results

**Дата тестирования:** ___________  
**Режим бота:** ___________  (`block` / `redact` / `restore`)  
**Тестировал:** ___________

---

## Результаты

| Test Case | Detected | Action Taken | Passed / Missed | Notes |
|-----------|----------|--------------|-----------------|-------|
| TC-01: AWS Access Key | ☐ Yes ☐ No | ☐ Blocked ☐ Redacted ☐ Restored ☐ None | ☐ Passed ☐ Missed | |
| TC-02: OpenAI-Like Key | ☐ Yes ☐ No | ☐ Blocked ☐ Redacted ☐ Restored ☐ None | ☐ Passed ☐ Missed | |
| TC-03: GitHub Token | ☐ Yes ☐ No | ☐ Blocked ☐ Redacted ☐ Restored ☐ None | ☐ Passed ☐ Missed | |
| TC-04: Email Address | ☐ Yes ☐ No | ☐ Blocked ☐ Redacted ☐ Restored ☐ None | ☐ Passed ☐ Missed | |
| TC-05: Phone Number | ☐ Yes ☐ No | ☐ Blocked ☐ Redacted ☐ Restored ☐ None | ☐ Passed ☐ Missed | |
| TC-06: Card Number | ☐ Yes ☐ No | ☐ Blocked ☐ Redacted ☐ Restored ☐ None | ☐ Passed ☐ Missed | |
| TC-07: Base64 Token | ☐ Yes ☐ No | ☐ Blocked ☐ Redacted ☐ Restored ☐ None | ☐ Passed ☐ Missed | |
| TC-08: Split Secret | ☐ Yes ☐ No | ☐ Blocked ☐ Redacted ☐ Restored ☐ None | ☐ Passed ☐ Missed | |
| TC-09: Clean Prompt | — (n/a) | ☐ Pass-through ☐ Wrongly blocked | ☐ Passed ☐ Missed | |
| TC-10: Output Hallucination | ☐ Yes ☐ No | ☐ Blocked ☐ Warning ☐ None | ☐ Passed ☐ Missed | |
| TC-11: Suspicious URL | ☐ Yes ☐ No | ☐ Blocked ☐ Warning ☐ None | ☐ Passed ☐ Missed | |
| TC-12: System Prompt Leak | ☐ Yes ☐ No | ☐ Blocked ☐ Warning ☐ None | ☐ Passed ☐ Missed | |

---

## Сводка

| Метрика | Значение |
|---------|----------|
| Всего кейсов | 12 |
| Пройдено (Passed) | |
| Провалено (Missed) | |
| Ложные срабатывания (False Positive) | |
| Ложные пропуски (False Negative) | |

---

## Замечания по отдельным кейсам

**TC-08 (Split Secret)** — ожидаемый False Negative: split-секрет намеренно обходит regex. Не считать провалом для MVP.

**TC-09 (Clean Prompt)** — если blocked: ложное срабатывание. Проверить regex-паттерны guard.

---

## Дополнительные наблюдения

<!-- Запиши здесь всё нестандартное поведение, неожиданные ответы LLM, edge cases -->

