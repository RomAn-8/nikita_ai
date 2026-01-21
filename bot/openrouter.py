import requests
from .config import OPENROUTER_API_KEY, OPENROUTER_MODEL

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }


def chat_completion(messages, timeout: int = 60, temperature: float = 0.7, model: str | None = None) -> str:
    payload = {
        "model": model or OPENROUTER_MODEL,
        "messages": messages,
        "temperature": float(temperature),
    }

    r = requests.post(OPENROUTER_CHAT_URL, headers=_headers(), json=payload, timeout=timeout)
    r.raise_for_status()

    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return ""
