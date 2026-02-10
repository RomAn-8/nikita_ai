# Примеры кода для nikita_ai

## Создание команды бота

### Базовая структура команды

```python
async def my_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Описание команды."""
    if not update.message:
        return
    
    await update.message.chat.send_action("typing")
    
    try:
        # Логика команды
        result = "Результат"
        await safe_reply_text(update, result)
    except Exception as e:
        logger.exception(f"Error in my_command: {e}")
        await safe_reply_text(update, f"Ошибка: {e}")
```

### Команда с аргументами

```python
async def command_with_args(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    
    if not context.args:
        await safe_reply_text(update, "Использование: /command <аргумент>")
        return
    
    arg = " ".join(context.args)
    await safe_reply_text(update, f"Получен аргумент: {arg}")
```

## Работа с базой данных

### Чтение данных

```python
from .embeddings import open_db

def get_data(chat_id: int) -> list[dict]:
    with open_db() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM table WHERE chat_id = ?",
            (chat_id,)
        )
        return [dict(row) for row in cursor.fetchall()]
```

### Запись данных

```python
def save_data(chat_id: int, data: str) -> None:
    with open_db() as conn:
        conn.execute(
            "INSERT INTO table (chat_id, data) VALUES (?, ?)",
            (chat_id, data)
        )
        conn.commit()
```

## Работа с RAG

### Поиск релевантных фрагментов

```python
from .embeddings import search_relevant_chunks, RAG_TOP_K, RAG_SIM_THRESHOLD

# Поиск с фильтром
chunks = search_relevant_chunks(
    query_text="Как работает система?",
    top_k=RAG_TOP_K,
    min_similarity=RAG_SIM_THRESHOLD,
    apply_threshold=True
)

# Использование результатов
for chunk in chunks:
    print(f"Документ: {chunk['doc_name']}")
    print(f"Текст: {chunk['text']}")
    print(f"Похожесть: {chunk['similarity']:.4f}")
```

### Создание эмбеддингов для файла

```python
from .embeddings import process_readme_file

result = process_readme_file(
    file_content=file_content,
    doc_name="example.md",
    replace_existing=True
)

if result["success"]:
    print(f"Создано {result['chunks_count']} чанков")
else:
    print(f"Ошибка: {result['error']}")
```

## Работа с MCP

### Вызов MCP инструмента

```python
from .mcp_client import get_git_branch

# Получение git ветки
branch = await get_git_branch()
if branch:
    print(f"Текущая ветка: {branch}")
```

### Использование MCP в команде

```python
async def my_mcp_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    
    await update.message.chat.send_action("typing")
    
    try:
        result = await get_git_branch()
        if result:
            await safe_reply_text(update, f"Результат: {result}")
        else:
            await safe_reply_text(update, "Не удалось получить данные")
    except Exception as e:
        logger.exception(f"Error: {e}")
        await safe_reply_text(update, f"Ошибка: {e}")
```

## Работа с LLM

### Простой запрос к LLM

```python
from .openrouter import chat_completion

messages = [
    {"role": "system", "content": "Ты помощник."},
    {"role": "user", "content": "Вопрос пользователя"}
]

response = chat_completion(messages, temperature=0.7)
print(response)
```

### Запрос с памятью

```python
from .summarizer import build_messages_with_db_memory

system_prompt = "Ты помощник."
chat_id = 12345

messages = build_messages_with_db_memory(system_prompt, chat_id=chat_id)
messages.append({"role": "user", "content": "Вопрос"})

response = chat_completion(messages)
```

## Обработка документов

### Скачивание и обработка файла

```python
async def process_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.document:
        return
    
    document = update.message.document
    file_name = document.file_name or ""
    
    # Скачиваем файл
    file = await context.bot.get_file(document.file_id)
    file_content_bytes = await file.download_as_bytearray()
    file_content = file_content_bytes.decode("utf-8", errors="replace")
    
    # Обрабатываем содержимое
    # ...
```
