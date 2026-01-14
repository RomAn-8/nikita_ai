import requests
from .config import OPENROUTER_API_KEY, OPENROUTER_MODEL

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def chat_completion(messages, timeout: int = 60) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0.7,
    }

    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()

    data = r.json()
    return data["choices"][0]["message"]["content"]
