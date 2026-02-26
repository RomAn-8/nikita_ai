"""LLM service wrapper for OpenRouter and Ollama."""

import requests
import logging
from typing import Any
from ..openrouter import chat_completion, chat_completion_raw
from ..config import (
    OPENROUTER_MODEL, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT,
    OLLAMA_TEMPERATURE, OLLAMA_NUM_CTX, OLLAMA_NUM_PREDICT, OLLAMA_SYSTEM_PROMPT,
    ANALYZE_MODEL
)

logger = logging.getLogger(__name__)


def call_llm(
    messages: list[dict[str, Any]],
    temperature: float = 0.7,
    model: str | None = None,
    timeout: int = 120,
) -> str | None:
    """
    Call LLM via OpenRouter.
    
    Args:
        messages: List of messages (with role and content)
        temperature: Temperature setting
        model: Model name (default: OPENROUTER_MODEL)
        timeout: Request timeout in seconds
        
    Returns:
        LLM response text or None on error
    """
    if model is None:
        model = OPENROUTER_MODEL
    
    return chat_completion(messages, temperature=temperature, model=model, timeout=timeout)


def call_llm_raw(
    messages: list[dict[str, Any]],
    temperature: float = 0.7,
    model: str | None = None,
    timeout: int = 120,
) -> dict[str, Any] | None:
    """
    Call LLM via OpenRouter and return raw response.
    
    Args:
        messages: List of messages (with role and content)
        temperature: Temperature setting
        model: Model name (default: OPENROUTER_MODEL)
        timeout: Request timeout in seconds
        
    Returns:
        Raw LLM response dict or None on error
    """
    if model is None:
        model = OPENROUTER_MODEL
    
    return chat_completion_raw(messages, temperature=temperature, model=model, timeout=timeout)


async def send_to_ollama(question: str, user_data: dict = None) -> str:
    """Отправляет запрос в Ollama API и возвращает ответ модели."""
    try:
        temperature = float(user_data.get("ollama_temperature", OLLAMA_TEMPERATURE)) if user_data else OLLAMA_TEMPERATURE
        num_ctx = int(user_data.get("ollama_num_ctx", OLLAMA_NUM_CTX)) if user_data else OLLAMA_NUM_CTX
        num_predict = int(user_data.get("ollama_num_predict", OLLAMA_NUM_PREDICT)) if user_data else OLLAMA_NUM_PREDICT
        system_prompt = user_data.get("ollama_system_prompt", OLLAMA_SYSTEM_PROMPT) if user_data else OLLAMA_SYSTEM_PROMPT
        
        if not (0.0 <= temperature <= 2.0):
            raise ValueError(f"Температура должна быть в диапазоне от 0.0 до 2.0, получено: {temperature}")
        if num_ctx <= 0 or num_ctx > 32768:
            raise ValueError(f"Контекстное окно должно быть от 1 до 32768, получено: {num_ctx}")
        if num_predict <= 0 or num_predict > 8192:
            raise ValueError(f"Максимальная длина ответа должна быть от 1 до 8192, получено: {num_predict}")
        
        api_url = f"{OLLAMA_BASE_URL}/api/chat"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        enhanced_question = question
        if any(phrase in question.lower() for phrase in ["что такое", "объясни", "расскажи", "парадокс", "гипотеза"]):
            enhanced_question = f"{question}\n\nВажно: отвечай точно, основываясь на реальных фактах. Если не уверен, скажи об этом."
        messages.append({"role": "user", "content": enhanced_question})
        
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": num_predict
            }
        }
        
        logger.info(f"Sending request to Ollama: {api_url}, model: {OLLAMA_MODEL}, temperature: {temperature}, num_ctx: {num_ctx}, num_predict: {num_predict}")
        logger.debug(f"Ollama payload: {payload}")
        
        response = requests.post(api_url, json=payload, timeout=OLLAMA_TIMEOUT)
        
        logger.debug(f"Ollama response status: {response.status_code}")
        response.raise_for_status()
        
        data = response.json()
        
        if "error" in data:
            error_msg = data.get("error", "Неизвестная ошибка")
            logger.error(f"Ollama API error: {error_msg}, full response: {data}")
            raise ValueError(f"Ошибка модели: {error_msg}")
        
        if "message" in data and "content" in data["message"]:
            answer = data["message"]["content"].strip()
            if answer:
                logger.info(f"Ollama response received, length: {len(answer)}")
                return answer
            else:
                logger.warning(f"Ollama returned empty content, full response: {data}")
                raise ValueError("Модель вернула пустой ответ")
        else:
            logger.warning(f"Unexpected Ollama response structure: {data}")
            raise ValueError("Неожиданный формат ответа от модели")
            
    except requests.exceptions.Timeout:
        logger.exception("Ollama request timeout")
        raise ConnectionError("Локальная модель недоступна (таймаут)")
    except requests.exceptions.ConnectionError:
        logger.exception("Ollama connection error")
        raise ConnectionError("Локальная модель недоступна (ошибка подключения)")
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if hasattr(e, 'response') and e.response else 'unknown'
        error_body = ""
        if hasattr(e, 'response') and e.response:
            try:
                error_body = e.response.text
                logger.error(f"Ollama HTTP error {status_code}: {error_body}")
            except:
                pass
        logger.exception(f"Ollama HTTP error: {status_code}")
        raise ConnectionError(f"Ошибка при обращении к локальной модели (HTTP {status_code})")
    except ValueError as e:
        logger.error(f"Ollama model error: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in send_to_ollama: {type(e).__name__}: {str(e)}")
        raise ConnectionError(f"Неожиданная ошибка при обращении к локальной модели: {str(e)}")


async def send_to_ollama_analyze(json_content: str, question: str) -> str:
    """Отправляет запрос в Ollama API для анализа JSON данных и возвращает ответ модели."""
    try:
        api_url = f"{OLLAMA_BASE_URL}/api/chat"
        
        system_prompt = "Ты — ассистент для анализа логов. Анализируй предоставленные JSON данные и отвечай на вопросы пользователя. Отвечай точно, кратко и только на русском языке."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"JSON данные:\n{json_content}\n\nВопрос: {question}"}
        ]
        
        payload = {
            "model": ANALYZE_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": OLLAMA_TEMPERATURE,
                "num_ctx": OLLAMA_NUM_CTX,
                "num_predict": OLLAMA_NUM_PREDICT
            }
        }
        
        logger.info(f"Sending analyze request to Ollama: {api_url}, model: {ANALYZE_MODEL}")
        logger.debug(f"Ollama analyze payload: {payload}")
        
        response = requests.post(api_url, json=payload, timeout=OLLAMA_TIMEOUT)
        
        logger.debug(f"Ollama analyze response status: {response.status_code}")
        response.raise_for_status()
        
        data = response.json()
        
        if "error" in data:
            error_msg = data.get("error", "Неизвестная ошибка")
            logger.error(f"Ollama API error: {error_msg}, full response: {data}")
            raise ValueError(f"Ошибка модели: {error_msg}")
        
        if "message" in data and "content" in data["message"]:
            answer = data["message"]["content"].strip()
            if answer:
                logger.info(f"Ollama analyze response received, length: {len(answer)}")
                return answer
            else:
                logger.warning(f"Ollama returned empty content, full response: {data}")
                raise ValueError("Модель вернула пустой ответ")
        else:
            logger.warning(f"Unexpected Ollama response structure: {data}")
            raise ValueError("Неожиданный формат ответа от модели")
            
    except requests.exceptions.Timeout:
        logger.exception("Ollama analyze request timeout")
        raise ConnectionError("Локальная модель недоступна (таймаут)")
    except requests.exceptions.ConnectionError:
        logger.exception("Ollama analyze connection error")
        raise ConnectionError("Локальная модель недоступна (ошибка подключения)")
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if hasattr(e, 'response') and e.response else 'unknown'
        error_body = ""
        if hasattr(e, 'response') and e.response:
            try:
                error_body = e.response.text
                logger.error(f"Ollama HTTP error {status_code}: {error_body}")
            except:
                pass
        logger.exception(f"Ollama HTTP error: {status_code}")
        raise ConnectionError(f"Ошибка при обращении к локальной модели (HTTP {status_code})")
    except ValueError as e:
        logger.error(f"Ollama model error: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in send_to_ollama_analyze: {type(e).__name__}: {str(e)}")
        raise ConnectionError(f"Неожиданная ошибка при обращении к локальной модели: {str(e)}")


def get_ollama_settings_display(user_data: dict = None) -> str:
    """Формирует строку с текущими настройками модели."""
    temperature = float(user_data.get("ollama_temperature", OLLAMA_TEMPERATURE)) if user_data else OLLAMA_TEMPERATURE
    num_ctx = int(user_data.get("ollama_num_ctx", OLLAMA_NUM_CTX)) if user_data else OLLAMA_NUM_CTX
    num_predict = int(user_data.get("ollama_num_predict", OLLAMA_NUM_PREDICT)) if user_data else OLLAMA_NUM_PREDICT
    system_prompt = user_data.get("ollama_system_prompt", OLLAMA_SYSTEM_PROMPT) if user_data else OLLAMA_SYSTEM_PROMPT
    
    return (
        f"📊 Текущие настройки модели:\n"
        f"• Температура: {temperature}\n"
        f"• Контекстное окно: {num_ctx}\n"
        f"• Максимальная длина ответа: {num_predict}\n"
        f"• Системный промпт: {system_prompt[:50]}{'...' if len(system_prompt) > 50 else ''}"
    )
