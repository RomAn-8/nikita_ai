import requests
import logging
from .config import OPENROUTER_API_KEY, OPENROUTER_MODEL

logger = logging.getLogger(__name__)

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }


def chat_completion_raw(
    messages,
    timeout: int = 60,
    temperature: float = 0.7,
    model: str | None = None,
) -> dict:
    payload = {
        "model": model or OPENROUTER_MODEL,
        "messages": messages,
        "temperature": float(temperature),
    }

    try:
        r = requests.post(OPENROUTER_CHAT_URL, headers=_headers(), json=payload, timeout=timeout)
        
        # Логируем детали ошибки перед raise_for_status
        if r.status_code != 200:
            error_detail = ""
            try:
                error_detail = r.json()
            except:
                error_detail = r.text[:500]
            
            logger.error(f"OpenRouter API error {r.status_code}: {error_detail}")
            logger.error(f"Request payload: model={payload.get('model')}, messages_count={len(payload.get('messages', []))}")
            
            # Пытаемся извлечь более понятное сообщение об ошибке
            if isinstance(error_detail, dict):
                error_msg = error_detail.get("error", {}).get("message", "") if isinstance(error_detail.get("error"), dict) else str(error_detail)
                if error_msg:
                    raise requests.exceptions.HTTPError(f"OpenRouter API error: {error_msg}")
        
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        # Пробрасываем HTTPError дальше с улучшенным сообщением
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in chat_completion_raw: {e}")
        raise


def chat_completion(
    messages,
    timeout: int = 60,
    temperature: float = 0.7,
    model: str | None = None,
) -> str:
    data = chat_completion_raw(messages, timeout=timeout, temperature=temperature, model=model)
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return ""
