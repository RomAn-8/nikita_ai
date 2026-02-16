FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

RUN pip install --no-cache-dir uv

# Сначала зависимости (для кеша)
COPY pyproject.toml uv.lock* /app/
RUN uv sync --frozen --no-dev

# Потом код
COPY . /app/

ENV PATH="/app/.venv/bin:$PATH"

# Вариант A (если у тебя точка входа как модуль):
CMD ["python", "-m", "bot.main"]
# Если у тебя запускается иначе — поменяй на:
# CMD ["python", "bot/main.py"]