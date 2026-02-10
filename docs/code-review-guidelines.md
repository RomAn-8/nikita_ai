# Руководство по Code Review для nikita_ai

## Общие принципы

### Что проверять в PR

1. **Функциональность**
   - Код работает как задумано?
   - Нет ли очевидных багов?
   - Обработаны ли edge cases?

2. **Безопасность**
   - Нет ли SQL injection (используются параметризованные запросы)?
   - Нет ли утечек токенов/ключей в коде?
   - Правильно ли обрабатываются пользовательские данные?

3. **Производительность**
   - Нет ли неоптимальных запросов к БД?
   - Правильно ли используются async/await?
   - Нет ли лишних циклов или неэффективных операций?

4. **Читаемость и поддерживаемость**
   - Код понятен и самодокументирован?
   - Используются правильные имена переменных и функций?
   - Есть ли docstrings для сложных функций?

5. **Соответствие стандартам проекта**
   - Следует ли код правилам из `code-style.md`?
   - Соответствует ли архитектуре проекта?
   - Используются ли правильные паттерны?

## Типичные проблемы и как их находить

### 1. Обработка ошибок

❌ **Плохо:**
```python
async def my_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    result = await some_async_operation()
    await safe_reply_text(update, result)
```

✅ **Хорошо:**
```python
async def my_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    
    try:
        result = await some_async_operation()
        await safe_reply_text(update, result)
    except Exception as e:
        logger.exception(f"Error in my_command: {e}")
        await safe_reply_text(update, f"Ошибка: {e}")
```

**Что проверять:**
- Всегда ли есть try/except для асинхронных операций?
- Логируются ли исключения через `logger.exception()`?
- Показываются ли пользователю понятные сообщения об ошибках?

### 2. Работа с базой данных

❌ **Плохо:**
```python
conn.execute(f"SELECT * FROM users WHERE id = {user_id}")
```

✅ **Хорошо:**
```python
with open_db() as conn:
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        "SELECT * FROM users WHERE id = ?",
        (user_id,)
    )
    result = cursor.fetchone()
```

**Что проверять:**
- Используются ли параметризованные запросы (защита от SQL injection)?
- Используется ли контекстный менеджер `with open_db()`?
- Делается ли `conn.commit()` после изменений?
- Используется ли `conn.row_factory = sqlite3.Row` для удобного доступа к данным?

### 3. Асинхронные операции

❌ **Плохо:**
```python
def sync_function():
    response = requests.get(url)  # Блокирующий вызов
    return response.json()
```

✅ **Хорошо:**
```python
async def async_function():
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()
```

**Что проверять:**
- Используется ли `httpx.AsyncClient` вместо `requests`?
- Все ли I/O операции асинхронные?
- Правильно ли используется `await`?

### 4. Проверка входных данных

❌ **Плохо:**
```python
async def command_with_args(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    arg = context.args[0]  # Может быть IndexError
    number = int(arg)  # Может быть ValueError
```

✅ **Хорошо:**
```python
async def command_with_args(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    
    if not context.args or len(context.args) < 1:
        await safe_reply_text(update, "Использование: /command <аргумент>")
        return
    
    try:
        arg = context.args[0]
        number = int(arg)
    except ValueError:
        await safe_reply_text(update, "Аргумент должен быть числом")
        return
```

**Что проверять:**
- Проверяется ли наличие `update.message`?
- Валидируются ли аргументы команд?
- Обрабатываются ли исключения при преобразовании типов?

### 5. Типизация

❌ **Плохо:**
```python
def process_data(data):
    return data["field"]
```

✅ **Хорошо:**
```python
def process_data(data: dict[str, Any]) -> str:
    return data["field"]
```

**Что проверять:**
- Есть ли type hints для всех функций?
- Указан ли возвращаемый тип?
- Используются ли типы из `typing` (`Any`, `Optional`, `list`, `dict`)?

### 6. Именование

**Что проверять:**
- Функции и переменные: `snake_case`
- Команды бота: заканчиваются на `_cmd`
- Константы: `UPPER_CASE`
- Классы: `PascalCase`

### 7. Импорты

❌ **Плохо:**
```python
import os
import json
from bot.config import TELEGRAM_BOT_TOKEN
import requests
from bot.embeddings import search_relevant_chunks
```

✅ **Хорошо:**
```python
import os
import json

import requests

from bot.config import TELEGRAM_BOT_TOKEN
from bot.embeddings import search_relevant_chunks
```

**Что проверять:**
- Группируются ли импорты (стандартная библиотека, сторонние, локальные)?
- Используются ли относительные импорты для локальных модулей (`from .module import ...`)?

## Специфичные проверки для nikita_ai

### Команды бота

**Обязательные элементы:**
1. Проверка `if not update.message: return`
2. `await update.message.chat.send_action("typing")` для длительных операций
3. Обработка ошибок с логированием
4. Использование `safe_reply_text()` для ответов

**Пример правильной команды:**
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

### Работа с MCP

**Что проверять:**
- Обрабатываются ли ошибки подключения к MCP серверу?
- Показываются ли понятные сообщения, если сервер недоступен?
- Используется ли правильный URL (`MCP_SERVER_URL` из переменных окружения)?

**Пример:**
```python
try:
    result = await get_git_branch()
    if result:
        await safe_reply_text(update, f"Ветка: {result}")
    else:
        await safe_reply_text(update, "Не удалось получить ветку")
except ValueError as e:
    # ValueError содержит понятное сообщение об ошибке
    await safe_reply_text(update, f"Ошибка MCP: {e}")
```

### Работа с RAG

**Что проверять:**
- Правильно ли используется `search_relevant_chunks()`?
- Учитывается ли `RAG_SIM_THRESHOLD` и `RAG_TOP_K`?
- Обрабатывается ли случай, когда релевантные фрагменты не найдены?

## Частые баги

### 1. Утечка ресурсов

❌ **Плохо:**
```python
conn = open_db()
cursor = conn.execute("SELECT ...")
# Забыли закрыть соединение
```

✅ **Хорошо:**
```python
with open_db() as conn:
    cursor = conn.execute("SELECT ...")
    # Автоматически закроется
```

### 2. Race conditions

❌ **Плохо:**
```python
if not exists_in_db():
    insert_into_db()  # Может быть вставлено дважды
```

✅ **Хорошо:**
```python
try:
    insert_into_db()
except sqlite3.IntegrityError:
    # Уже существует
    pass
```

### 3. Неправильная обработка None

❌ **Плохо:**
```python
result = await get_data()
return result["field"]  # Может быть KeyError или TypeError
```

✅ **Хорошо:**
```python
result = await get_data()
if result and "field" in result:
    return result["field"]
return None
```

## Рекомендации по улучшению кода

1. **Добавляйте docstrings** для сложных функций
2. **Используйте константы** вместо магических чисел
3. **Разбивайте большие функции** на меньшие
4. **Избегайте дублирования кода** (DRY принцип)
5. **Добавляйте комментарии** для неочевидных решений
6. **Тестируйте edge cases** (пустые значения, None, большие числа)

## Формат комментариев в ревью

При комментировании кода:
- Будьте конкретны: указывайте файл и строку
- Предлагайте решения, а не только указывайте на проблемы
- Будьте вежливы и конструктивны
- Объясняйте "почему", а не только "что"

**Пример хорошего комментария:**
```
bot/main.py:123 - Использование f-строки здесь может привести к SQL injection. 
Рекомендую использовать параметризованный запрос:
conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
```
