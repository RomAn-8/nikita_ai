# Правила стиля кода для nikita_ai

## Именование

### Функции и переменные
- Используйте `snake_case` для функций и переменных
- Примеры:
  ```python
  async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
      chat_id = int(update.effective_chat.id)
  ```

### Команды бота
- Команды бота должны заканчиваться на `_cmd`
- Примеры:
  ```python
  async def embed_create_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
      ...
  ```

### Константы
- Используйте `UPPER_CASE` для констант
- Примеры:
  ```python
  CHUNK_SIZE = 1000
  RAG_SIM_THRESHOLD = 0.5
  ```

## Структура кода

### Обработчики команд
- Всегда проверяйте `if not update.message: return`
- Используйте `await update.message.chat.send_action("typing")` для длительных операций
- Обрабатывайте ошибки с помощью try/except и логирования

Пример:
```python
async def my_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    
    await update.message.chat.send_action("typing")
    
    try:
        # Ваш код
        await safe_reply_text(update, "Успех")
    except Exception as e:
        logger.exception(f"Error in my_command: {e}")
        await safe_reply_text(update, f"Ошибка: {e}")
```

### Работа с базой данных
- Используйте контекстный менеджер `with open_db() as conn:`
- Всегда делайте `conn.commit()` после изменений
- Используйте параметризованные запросы для безопасности

Пример:
```python
with open_db() as conn:
    conn.execute(
        "INSERT INTO table (field) VALUES (?)",
        (value,)
    )
    conn.commit()
```

## Импорты

- Группируйте импорты: стандартная библиотека, сторонние, локальные
- Используйте относительные импорты для локальных модулей: `from .embeddings import ...`

## Обработка ошибок

- Всегда логируйте исключения: `logger.exception(...)`
- Показывайте пользователю понятные сообщения об ошибках
- Не проглатывайте исключения без логирования

## Типизация

- Используйте type hints для всех функций
- Указывайте возвращаемый тип: `-> None`, `-> str`, `-> dict[str, Any]`

## Комментарии и документация

- Используйте docstrings для функций
- Комментарии должны объяснять "почему", а не "что"
- Примеры:
  ```python
  async def process_file(file_content: str) -> dict[str, Any]:
      """
      Обрабатывает файл и создает эмбеддинги.
      
      Args:
          file_content: Содержимое файла
          
      Returns:
          Словарь со статистикой обработки
      """
      ...
  ```
