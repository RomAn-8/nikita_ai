# Архитектура проекта nikita_ai

## Структура репозитория

```
nikita_ai/
├── bot/
│   ├── core/                   # Каркас системы: контекст, роутинг, ошибки, промпты
│   │   ├── agent.py            # Абстрактный базовый класс Agent
│   │   ├── context.py          # AgentContext — единый объект состояния чата
│   │   ├── errors.py           # Классы ошибок + safe_reply_text + handle_error
│   │   ├── prompts.py          # Системные промпты для всех режимов
│   │   └── router.py           # MessageRouter — маршрутизация по режиму
│   │
│   ├── handlers/               # Telegram-слой: команды, inline-кнопки, документы
│   │   ├── base.py             # Абстрактный Handler с get_agent_context()
│   │   ├── start.py            # /start (мёртвый код — перекрыт функцией в main.py)
│   │   ├── help.py             # /help  (мёртвый код — перекрыт функцией в main.py)
│   │   ├── modes.py            # /mode_text, /mode_json, /mode_summary, /thinking_model, /expert_group_model
│   │   ├── settings.py         # /ch_temperature, /ch_memory, /clear_memory
│   │   ├── models.py           # /model_glm, /model_gemma
│   │   ├── rag.py              # /embed_create, /embed_docs, /rag_model, /clear_embeddings
│   │   ├── weather.py          # /weather_sub, /weather_sub_stop
│   │   ├── digest.py           # /digest — утренняя сводка: погода + новости
│   │   ├── registration.py     # /register, /unregister, /train_signup, /train_move, /train_cancel, /support
│   │   ├── tasks.py            # /task_list — работа с задачами
│   │   ├── deployment.py       # /deploy_bot, /stop_bot
│   │   ├── local.py            # /local_model (Ollama), /analyze (анализ JSON)
│   │   ├── personal.py         # /me — персональный ассистент
│   │   ├── voice.py            # /voice — голосовой ассистент
│   │   ├── special.py          # /tz_creation_site, /forest_split
│   │   ├── review.py           # /review_pr
│   │   └── tictactoe.py        # /tictactoe + inline-callback
│   │
│   ├── services/               # Бизнес-логика: БД, память, профиль, LLM
│   │   ├── database.py         # open_db(), init_db(), CRUD для chat_history/settings/summary
│   │   ├── memory.py           # add_message(), get_messages(), build_messages_with_memory()
│   │   ├── context_manager.py  # get_mode(), get_temperature(), get_memory_enabled(), get_model()
│   │   ├── profile.py          # load/save user_profile.json, build_me_system_prompt()
│   │   ├── llm.py              # call_llm(), call_llm_raw() — обёртка над openrouter.py
│   │   └── tictactoe.py        # Логика игры крестики-нолики + ИИ-ход
│   │
│   ├── tools/                  # Инструменты агента
│   │   ├── base.py             # Tool (абстракция) + ToolResult
│   │   ├── registry.py         # ToolRegistry — глобальный реестр инструментов
│   │   ├── mcp_tools.py        # Инструменты через MCP
│   │   └── rag_tool.py         # RAG-инструмент для поиска по документам
│   │
│   ├── utils/                  # Вспомогательные функции без бизнес-модели
│   │   ├── text.py             # split_telegram_text(), looks_like_json(), is_forest_final()
│   │   ├── validation.py       # Валидаторы входных данных
│   │   ├── helpers.py          # safe_reply_text (переиспользуется из core/errors.py)
│   │   └── tz_helpers.py       # Работа с временными зонами
│   │
│   ├── config.py               # Все настройки из .env (токены, модели, пути)
│   ├── main.py                 # Composition root: импорты, регистрация handlers, run()
│   ├── openrouter.py           # HTTP-клиент OpenRouter API (chat + audio transcription)
│   ├── embeddings.py           # RAG: чанкинг, эмбеддинги, поиск по doc_chunks
│   ├── summarizer.py           # Summary mode: сжатие истории через LLM
│   ├── weather_subscription.py # Фоновая подписка на погодные обновления
│   ├── mcp_client.py           # MCP-клиент: git, PR, пользователи, записи, задачи, деплой
│   ├── mcp_weather.py          # MCP-клиент: получение погоды
│   ├── mcp_news.py             # MCP-клиент: получение новостей
│   ├── mcp_docker.py           # MCP-клиент: управление Docker-контейнерами
│   ├── tokens_test.py          # /tokens_test — тест расхода токенов
│   └── bot_memory.sqlite3      # SQLite БД (история, настройки, summary, эмбеддинги)
│
├── config/
│   └── user_profile.json       # Профиль пользователя (имя, интересы, стиль общения)
│
├── docs/                       # Документация проекта
│   ├── architecture.md         # Этот файл
│   ├── code-style.md
│   ├── code-examples.md
│   ├── best-practices.md
│   ├── code-review-guidelines.md
│   ├── support-faq.md
│   ├── club-rules.md
│   └── exercise-rules.md
│
├── scripts/
│   └── review_pr.py            # Анализ Pull Request (RAG + LLM)
│
├── pyproject.toml
├── uv.lock
└── .env                        # Секреты и настройки (не в git)
```

---

## Слои архитектуры

```
┌─────────────────────────────────────────────┐
│             Telegram API                     │
└────────────────────┬────────────────────────┘
                     │ Update
┌────────────────────▼────────────────────────┐
│           handlers/   (Telegram-слой)        │
│  Читает Update, валидирует, вызывает сервис  │
│  Отправляет ответ, ловит ошибки              │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│           services/   (Бизнес-логика)        │
│  БД, память, профиль, LLM, игровая логика    │
│  Не знает об Update и Telegram               │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│      core/ + tools/ + utils/  (Каркас)       │
│  AgentContext, MessageRouter, ToolRegistry   │
│  Общие утилиты и базовые типы               │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│  openrouter.py / embeddings.py / mcp_*.py   │
│  Внешние интеграции: LLM, RAG, MCP-серверы  │
└─────────────────────────────────────────────┘
```

---

## Ключевые компоненты

### AgentContext (`bot/core/context.py`)

Единый объект состояния для каждого входящего сообщения. Создаётся из Telegram `Update` + `ContextTypes.DEFAULT_TYPE` через `AgentContext.from_telegram_context()`.

Поля: `chat_id`, `user_id`, `mode`, `temperature`, `memory_enabled`, `model`, `user_data`.

Читает текущие настройки из `services/context_manager.py` — который в свою очередь смотрит сначала в `user_data` (сессия), затем в БД.

### MessageRouter (`bot/core/router.py`)

Глобальный маршрутизатор текстовых сообщений. Работает по принципу:
1. Берёт `mode` из `AgentContext`
2. Находит зарегистрированный handler для этого режима
3. Вызывает его

Режим задаётся командами `/mode_*` и хранится в `user_data["mode"]` + БД.

### База данных (`bot/services/database.py`)

SQLite-файл `bot/bot_memory.sqlite3`. Соединение открывается через `open_db()` с WAL + busy_timeout.

| Таблица | Содержимое |
|---|---|
| `chat_history` | История сообщений (chat_id, mode, role, content, ts) |
| `chat_settings` | Настройки чата (temperature, memory_enabled, model) |
| `summary` | Сжатая история для режима summary |
| `doc_chunks` | Чанки документов с векторами эмбеддингов для RAG |

Миграции колонок — идемпотентно через `_ensure_column()`.

### Память чата (`bot/services/memory.py`)

Тонкая обёртка над `database.py`. Хранит до `MEMORY_LIMIT_MESSAGES=30` сообщений. В историю включаются только режимы из `MEMORY_CHAT_MODES = ("text", "thinking", "experts", "rag")`.

### LLM-слой (`bot/openrouter.py` + `bot/services/llm.py`)

`openrouter.py` — низкоуровневый HTTP-клиент OpenRouter:
- `chat_completion_raw()` — сырой ответ `dict`
- `chat_completion()` — возвращает строку
- `transcribe_audio()` — расшифровка голоса через Whisper

`services/llm.py` — обёртка для вызова из сервисного слоя.

### RAG (`bot/embeddings.py`)

- Разбивает документы на чанки по 1000 символов (перекрытие 150)
- Генерирует эмбеддинги через OpenRouter (модель `EMBEDDING_MODEL`)
- Хранит векторы в `doc_chunks` как JSON-строки
- Поиск: косинусное сходство между вектором запроса и всеми чанками

### Summary mode (`bot/summarizer.py`)

Режим `summary` автоматически сжимает историю диалога через LLM, когда число сообщений превышает порог. Сжатое summary хранится в таблице `summary`.

### MCP-интеграции

Все MCP-клиенты подключаются к внешнему MCP-серверу по Streamable HTTP.

| Модуль | Инструменты |
|---|---|
| `mcp_client.py` | git_branch, get_pr_diff, get_pr_files, user_*, reg_*, task_*, deploy_* |
| `mcp_weather.py` | Погода для города |
| `mcp_news.py` | Новости по теме |
| `mcp_docker.py` | site_up, site_screenshot, site_down |

URL по умолчанию: `http://127.0.0.1:8000/mcp` (переопределяется через `MCP_SERVER_URL`).

### Профиль пользователя (`bot/services/profile.py`)

Хранится в `config/user_profile.json`. Используется в режиме `/me` для персонализированных ответов. `build_me_system_prompt()` формирует системный промпт на основе профиля.

---

## Режимы работы

| Режим | Команда | Системный промпт | Сохранение истории |
|---|---|---|---|
| `text` | `/mode_text` | `SYSTEM_PROMPT_TEXT` | Да |
| `json` | `/mode_json` | `SYSTEM_PROMPT_JSON` | Нет |
| `summary` | `/mode_summary` | `SYSTEM_PROMPT_TEXT` | Да (со сжатием) |
| `rag` | `/rag_model` | динамический | Да |
| `thinking` | `/thinking_model` | `SYSTEM_PROMPT_THINKING` | Да |
| `experts` | `/expert_group_model` | `SYSTEM_PROMPT_EXPERTS` | Да |
| `tz` | `/tz_creation_site` | `SYSTEM_PROMPT_TZ` | Нет (user_data) |
| `forest` | `/forest_split` | `SYSTEM_PROMPT_FOREST` | Нет (user_data) |
| `local` | `/local_model` | `OLLAMA_SYSTEM_PROMPT` | Нет |
| `me` | `/me` | из профиля пользователя | Нет |
| `voice` | `/voice` | `VOICE_SYSTEM_PROMPT` | Нет |

---

## Регистрация команд (bot/main.py)

`bot/main.py` — composition root. Четыре обязательных точки для каждой новой команды:

1. `start()` (~строка 922) — список команд в `/start`
2. `help_cmd()` (~строка 1025) — список команд в `/help`
3. `post_init()` — `BotCommand(...)` для меню Telegram
4. `run()` — `app.add_handler(CommandHandler(...))`

> **Архитектурный долг**: `bot/handlers/start.py` и `bot/handlers/help.py` — мёртвый код. Функции `start()` и `help_cmd()` в `main.py` перекрывают эти импорты. Рефакторинг — отдельная задача.

---

## Потоки данных

### Обычное текстовое сообщение (режим text)

```
Пользователь → Telegram → on_text()
    │
    ├─ AgentContext.from_telegram_context() — читает mode, temperature, model
    ├─ MessageRouter.route() — находит handler по mode
    │
    ├─ build_messages_with_memory() — system_prompt + история из БД
    ├─ chat_completion() → OpenRouter API
    │
    ├─ db_add_message() — сохраняет вопрос и ответ
    └─ safe_reply_text() → Telegram
```

### Индексация документации (RAG)

```
Пользователь → отправляет .md файл → /embed_create
    │
    ├─ on_document() — скачивает файл
    ├─ process_readme_file() — нормализует и режет на чанки
    ├─ generate_embeddings_batch() → OpenRouter Embeddings API
    └─ Сохранение векторов в doc_chunks (SQLite)
```

### Поиск по RAG (/help <вопрос>)

```
Пользователь → /help <вопрос>
    │
    ├─ search_relevant_chunks() — эмбеддинг запроса + косинусное сходство
    ├─ Фильтрация по RAG_SIM_THRESHOLD
    ├─ Формирование контекста из топ-K чанков
    ├─ chat_completion() → OpenRouter API
    └─ safe_reply_text() → Telegram
```

### Голосовое сообщение (режим voice)

```
Пользователь → голосовое сообщение
    │
    ├─ voice.py: скачивает .ogg файл
    ├─ transcribe_audio() → OpenRouter Whisper API — расшифровка
    ├─ chat_completion() → OpenRouter API — ответ на текст
    └─ safe_reply_text() → Telegram
```

---

## Конфигурация (.env)

| Переменная | Назначение |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Токен бота |
| `OPENROUTER_API_KEY` | Ключ OpenRouter |
| `OPENROUTER_MODEL` | Основная LLM-модель |
| `EMBEDDING_MODEL` | Модель для эмбеддингов |
| `RAG_SIM_THRESHOLD` | Порог cosine similarity для RAG (0.0–1.0) |
| `RAG_TOP_K` | Количество чанков для контекста |
| `OLLAMA_BASE_URL` | URL локального Ollama-сервера |
| `OLLAMA_MODEL` | Модель Ollama |
| `ME_MODEL` | Модель для /me (fallback: OPENROUTER_MODEL) |
| `VOICE_MODEL` | Модель для /voice (fallback: OPENROUTER_MODEL) |
| `VOICE_WHISPER_MODEL` | Модель для транскрипции (fallback: VOICE_MODEL) |
| `OPENROUTER_MODEL_GLM` | Дополнительная модель GLM |
| `OPENROUTER_MODEL_GEMMA` | Дополнительная модель Gemma |
| `MCP_SERVER_URL` | URL MCP-сервера (default: http://127.0.0.1:8000/mcp) |
| `DB_PATH` | Путь к SQLite БД |

---

## Зависимости (pyproject.toml)

| Пакет | Версия | Назначение |
|---|---|---|
| `python-telegram-bot` | 21.6 | Telegram Bot API |
| `requests` | 2.32.3 | HTTP-запросы к OpenRouter |
| `python-dotenv` | 1.0.1 | Загрузка .env |
| `mcp` | ≥0.4.0 | MCP-протокол (клиент) |
| `pydub` | ≥0.25.1 | Обработка аудио для голосового ассистента |

Управление зависимостями и запуск — через `uv`.
