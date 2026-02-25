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

# Ollama настройки
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b").strip()
# Таймаут в секундах (диапазон 120-300)
OLLAMA_TIMEOUT_RAW = int(os.getenv("OLLAMA_TIMEOUT", "120"))
OLLAMA_TIMEOUT = max(120, min(300, OLLAMA_TIMEOUT_RAW))  # Ограничиваем диапазон 120-300

# Параметры модели Ollama
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.5"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))
OLLAMA_SYSTEM_PROMPT = os.getenv("OLLAMA_SYSTEM_PROMPT", "Ты — ассистент Никита. Отвечай точно, кратко и только на русском языке. Если ты не уверен в ответе или не знаешь точной информации, честно скажи об этом. Не выдумывай факты. Отвечай только на основе реальных знаний.").strip()

# Модель для анализа JSON логов
ANALYZE_MODEL = os.getenv("ANALYZE_MODEL", "gemma3:1b").strip()

# Модель для персонального ассистента /me
# По умолчанию используем OPENROUTER_MODEL, если ME_MODEL не указан
ME_MODEL = os.getenv("ME_MODEL", "").strip()
if not ME_MODEL:
    ME_MODEL = OPENROUTER_MODEL  # Fallback на основную модель

# Модель для голосового ассистента /voice (для ответа на вопросы)
# По умолчанию используем OPENROUTER_MODEL, если VOICE_MODEL не указан
VOICE_MODEL = os.getenv("VOICE_MODEL", "").strip()
if not VOICE_MODEL:
    VOICE_MODEL = OPENROUTER_MODEL  # Fallback на основную модель

# Модель для распознавания речи через OpenRouter
# По умолчанию используем VOICE_MODEL, если VOICE_WHISPER_MODEL не указан
VOICE_WHISPER_MODEL = os.getenv("VOICE_WHISPER_MODEL", "").strip()
if not VOICE_WHISPER_MODEL:
    VOICE_WHISPER_MODEL = VOICE_MODEL  # Fallback на VOICE_MODEL

# Системный промпт для голосового ассистента
VOICE_SYSTEM_PROMPT = os.getenv("VOICE_SYSTEM_PROMPT", "Ты голосовой ассистент. Пользователь задал вопрос голосом. Ответь на вопрос кратко и точно на русском языке. Не повторяй вопрос пользователя в ответе.").strip()

# Путь к файлу профиля пользователя
USER_PROFILE_PATH = PROJECT_ROOT / "config" / "user_profile.json"

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is missing in .env")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is missing in .env")
