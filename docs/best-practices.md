# Best Practices для nikita_ai

## Общие принципы

### 1. Принцип единственной ответственности (SRP)

Каждая функция должна делать одну вещь и делать её хорошо.

❌ **Плохо:**
```python
async def process_and_save(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Скачивает файл, обрабатывает, создает эмбеддинги, сохраняет в БД, отправляет ответ
    ...
```

✅ **Хорошо:**
```python
async def process_file(file_content: str) -> list[dict]:
    """Обрабатывает файл и возвращает чанки."""
    ...

async def save_chunks(chunks: list[dict]) -> None:
    """Сохраняет чанки в БД."""
    ...

async def embed_create_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда для создания эмбеддингов."""
    file_content = await download_file(...)
    chunks = await process_file(file_content)
    await save_chunks(chunks)
```

### 2. DRY (Don't Repeat Yourself)

Избегайте дублирования кода. Выносите повторяющуюся логику в функции.

❌ **Плохо:**
```python
async def command1(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.chat.send_action("typing")
    try:
        # Логика 1
    except Exception as e:
        logger.exception(f"Error: {e}")
        await safe_reply_text(update, f"Ошибка: {e}")

async def command2(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.chat.send_action("typing")
    try:
        # Логика 2
    except Exception as e:
        logger.exception(f"Error: {e}")
        await safe_reply_text(update, f"Ошибка: {e}")
```

✅ **Хорошо:**
```python
def command_wrapper(func):
    """Декоратор для стандартной обработки команд."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        await update.message.chat.send_action("typing")
        try:
            await func(update, context)
        except Exception as e:
            logger.exception(f"Error in {func.__name__}: {e}")
            await safe_reply_text(update, f"Ошибка: {e}")
    return wrapper

@command_wrapper
async def command1(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Логика 1

@command_wrapper
async def command2(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Логика 2
```

### 3. Fail Fast

Проверяйте входные данные в начале функции и возвращайтесь сразу при ошибке.

❌ **Плохо:**
```python
async def my_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Много кода
    if not update.message:
        return  # Поздно!
    # Еще код
```

✅ **Хорошо:**
```python
async def my_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return  # Сразу в начале
    
    if not context.args:
        await safe_reply_text(update, "Использование: /command <arg>")
        return
    
    # Основная логика
```

## Работа с асинхронностью

### 1. Правильное использование async/await

Всегда используйте `await` для асинхронных операций.

❌ **Плохо:**
```python
async def my_function():
    result = some_async_function()  # Забыли await
    return result
```

✅ **Хорошо:**
```python
async def my_function():
    result = await some_async_function()
    return result
```

### 2. Параллельное выполнение

Используйте `asyncio.gather()` для параллельного выполнения независимых операций.

❌ **Плохо:**
```python
result1 = await operation1()
result2 = await operation2()
result3 = await operation3()
```

✅ **Хорошо:**
```python
result1, result2, result3 = await asyncio.gather(
    operation1(),
    operation2(),
    operation3()
)
```

### 3. Таймауты

Всегда устанавливайте таймауты для внешних запросов.

✅ **Хорошо:**
```python
async with httpx.AsyncClient(timeout=30.0) as client:
    response = await client.get(url)
```

## Работа с базой данных

### 1. Использование транзакций

Группируйте связанные операции в транзакции.

✅ **Хорошо:**
```python
with open_db() as conn:
    conn.execute("INSERT INTO table1 ...")
    conn.execute("INSERT INTO table2 ...")
    conn.commit()  # Оба изменения сохранятся вместе
```

### 2. Индексы

Убедитесь, что запросы используют индексы для производительности.

✅ **Хорошо:**
```python
# Создание индекса
conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_id ON chat_history(chat_id)")

# Запрос использует индекс
cursor = conn.execute("SELECT * FROM chat_history WHERE chat_id = ?", (chat_id,))
```

## Обработка ошибок

### 1. Специфичные исключения

Ловите конкретные исключения, а не общий `Exception`.

❌ **Плохо:**
```python
try:
    number = int(user_input)
except Exception:
    # Слишком широко
    pass
```

✅ **Хорошо:**
```python
try:
    number = int(user_input)
except ValueError:
    await safe_reply_text(update, "Введите число")
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    await safe_reply_text(update, "Произошла ошибка")
```

### 2. Понятные сообщения об ошибках

Показывайте пользователю понятные сообщения, но логируйте детали.

✅ **Хорошо:**
```python
try:
    result = await complex_operation()
except ValueError as e:
    logger.exception(f"Validation error in complex_operation: {e}")
    await safe_reply_text(update, "Некорректные данные. Проверьте ввод.")
except ConnectionError as e:
    logger.exception(f"Connection error: {e}")
    await safe_reply_text(update, "Не удалось подключиться к сервису. Попробуйте позже.")
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    await safe_reply_text(update, "Произошла ошибка. Мы уже работаем над её исправлением.")
```

## Производительность

### 1. Избегайте N+1 запросов

❌ **Плохо:**
```python
users = get_all_users()
for user in users:
    posts = get_posts_for_user(user.id)  # N запросов
```

✅ **Хорошо:**
```python
users = get_all_users()
user_ids = [user.id for user in users]
posts = get_posts_for_users(user_ids)  # 1 запрос
```

### 2. Используйте генераторы для больших данных

✅ **Хорошо:**
```python
def process_large_file(file_path: str):
    with open(file_path) as f:
        for line in f:
            yield process_line(line)  # Не загружает весь файл в память
```

### 3. Кэширование

Кэшируйте результаты дорогих операций.

✅ **Хорошо:**
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_operation(param: str) -> str:
    # Дорогая операция
    return result
```

## Безопасность

### 1. Валидация входных данных

Всегда валидируйте данные от пользователя.

✅ **Хорошо:**
```python
def validate_pr_number(pr_number_str: str) -> int | None:
    try:
        pr_number = int(pr_number_str)
        if pr_number < 1:
            return None
        return pr_number
    except ValueError:
        return None
```

### 2. Защита от SQL injection

Всегда используйте параметризованные запросы.

✅ **Хорошо:**
```python
conn.execute("SELECT * FROM users WHERE name = ?", (user_name,))
```

### 3. Не логируйте чувствительные данные

❌ **Плохо:**
```python
logger.info(f"User token: {token}")
```

✅ **Хорошо:**
```python
logger.info(f"User token: {token[:8]}...")  # Только первые 8 символов
```

## Тестируемость

### 1. Разделение логики и I/O

Выносите бизнес-логику в отдельные функции, которые легко тестировать.

✅ **Хорошо:**
```python
def calculate_similarity(embedding1: list[float], embedding2: list[float]) -> float:
    """Чистая функция, легко тестируется."""
    # Логика вычисления
    return similarity

async def search_chunks(query: str) -> list[dict]:
    """Использует чистую функцию."""
    query_embedding = await generate_embedding(query)
    # Использует calculate_similarity
    ...
```

### 2. Зависимости через параметры

Передавайте зависимости через параметры, а не импортируйте напрямую.

✅ **Хорошо:**
```python
async def process_data(data: str, db_conn: sqlite3.Connection) -> None:
    # Легко подменить db_conn для тестов
    db_conn.execute("INSERT ...")
```

## Документация

### 1. Docstrings

Добавляйте docstrings для всех публичных функций.

✅ **Хорошо:**
```python
async def search_relevant_chunks(
    query_text: str,
    model: str = EMBEDDING_MODEL,
    top_k: int = 3,
    min_similarity: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Ищет релевантные чанки по запросу пользователя.
    
    Args:
        query_text: Текст запроса пользователя
        model: Модель эмбеддингов
        top_k: Количество возвращаемых чанков
        min_similarity: Минимальная cosine similarity
        
    Returns:
        Список словарей с ключами: text, chunk_index, similarity, doc_name
    """
    ...
```

### 2. Комментарии

Комментируйте "почему", а не "что".

❌ **Плохо:**
```python
# Увеличиваем счетчик
counter += 1
```

✅ **Хорошо:**
```python
# Увеличиваем счетчик для отслеживания количества обработанных запросов
# Это нужно для мониторинга производительности
counter += 1
```
