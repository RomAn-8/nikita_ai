import os
from pathlib import Path
from dotenv import load_dotenv

# Определяем путь к .env файлу относительно корня проекта
# config.py находится в bot/config.py, .env в корне проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Загружаем .env с явным указанием пути
load_dotenv(dotenv_path=ENV_FILE)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()

# RAG настройки
RAG_SIM_THRESHOLD = float(os.getenv("RAG_SIM_THRESHOLD", "0.5"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

# Модель эмбеддингов (по умолчанию используем text-embedding-ada-002, так как она наиболее стабильна через OpenRouter)
# Альтернативы: openai/text-embedding-3-small, openai/text-embedding-3-large, jina/jina-embeddings-v3-small
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-ada-002").strip()

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is missing in .env")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is missing in .env")
