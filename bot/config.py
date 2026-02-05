import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()

# RAG настройки
RAG_SIM_THRESHOLD = float(os.getenv("RAG_SIM_THRESHOLD", "0.5"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is missing in .env")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is missing in .env")
