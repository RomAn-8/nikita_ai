# Пакет Red Team для Day 15

## Цель

Day 15 продолжает существующий security-трек в `nikita_ai` и добавляет documentation-only red-team пакет для Day 14 MVP pipeline.

Этот пакет не добавляет:

- новые команды бота;
- новые handlers или services;
- новую автоматизацию;
- изменения runtime-кода.

Его задача — дать безопасный и повторяемый набор ручных проверок для основной цели `/sec14_run`, а также вспомогательные материалы по поведению Day 13 gateway и contextual injection-паттернам из Day 12.

## Почему `/sec14_run` — основная цель

`/sec14_run` — самая ценная поверхность атаки в текущем проекте, потому что здесь в одном pipeline соединены сразу несколько стадий:

1. пользовательская задача превращается в generation prompt;
2. generation проходит через Day 13 gateway;
3. модель возвращает структурированные instructions для изменений файлов;
4. loop валидирует и применяет изменения к target project;
5. выполняются build/test;
6. второй LLM-pass делает security review;
7. результат возвращается пользователю как единый run/report.

Из-за этого `/sec14_run` ближе всего к реальному agentic workflow в проекте. Уязвимость здесь важнее, чем изолированная слабость в обычном chat flow.

## Охват

### Основная поверхность

- `/sec14_run`

### Второстепенные поверхности

- поведение Day 13 gateway;
- file/context injection demos из Day 12 как reference-материал.

### Намеренно вне охвата

- `/mode_text` как самостоятельная Day 15 поверхность атаки

`/mode_text` существует в проекте, но не является частью execution path Day 14 loop. Внутри Day 14 loop используется фиксированный gateway mode, поэтому Day 15 остаётся сфокусированным на тех поверхностях, которые реально влияют на target pipeline.

## Карта файлов

- `attacks_sec14_run.md` — основной operational playbook с copy-paste payloads для Telegram
- `attacks_gateway.md` — вспомогательные safe payloads для gateway
- `attack_matrix.md` — компактная матрица атак, категорий и ожидаемых stop layers
- `attacker_report_template.md` — шаблон для ручной фиксации red-team прогона
- `defender_report_template.md` — шаблон для инженерной оценки и follow-up по hardening
- `retest_checklist.md` — чеклист для повторных прогонов после hardening

## Как использовать пакет

1. Прочитать `attack_matrix.md` и выбрать набор атак.
2. Начать с атак `Готово сейчас` из `attacks_sec14_run.md`.
3. Запускать payloads вручную через Telegram.
4. Фиксировать наблюдения в `attacker_report_template.md`.
5. Если слабое место подтверждено, документировать remediation через `defender_report_template.md`.
6. После hardening повторно прогонять нужные кейсы по `retest_checklist.md`.

## Какие результаты считать полезными

Для Day 15 полезным результатом считается не только успешный bypass. Полезным результатом также может быть:

- gateway блокирует попытку на ранней стадии;
- generation output отклоняется validation-логикой;
- safe zone rules останавливают изменение;
- security review помечает рискованное поведение;
- final report корректно показывает реальную точку остановки.

Если атака остановилась на раннем слое, это всё равно валидный и часто желательный результат.

## Текущие ограничения Day 14 MVP

Эти ограничения нужно держать в виду при использовании Day 15:

- Day 14 — это узкий MVP над одним target project.
- attack surface построена вокруг prompt-driven pipeline, а не полного автономного planning.
- часть атак может остановиться уже на gateway или generation validation и не дойти до build/test или security review;
- false negatives и false positives всё ещё возможны, особенно для contextual или reformulated input;
- Day 15 намеренно остаётся ручным пакетом, а не automated test suite.

## Правила безопасности

Все Day 15 payloads должны оставаться:

- учебными;
- неразрушительными;
- без реальных секретов;
- без инструкций по повреждению внешнего проекта;
- сфокусированными на демонстрации слабых мест pipeline, а не на реальном abuse.

Не превращай этот пакет в exploit-документацию. Он должен оставаться безопасным red-team упражнением для текущей архитектуры `nikita_ai`.
