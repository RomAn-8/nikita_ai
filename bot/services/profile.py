"""User profile service."""

import json
import logging
from pathlib import Path

from ..config import USER_PROFILE_PATH, ME_MODEL
from ..openrouter import chat_completion

logger = logging.getLogger(__name__)


def load_user_profile() -> dict:
    """Load user profile from JSON file. Create default profile if file doesn't exist."""
    try:
        if not USER_PROFILE_PATH.exists():
            # Create default profile
            default_profile = {
                "name": "",
                "interests": [],
                "communication_style": "",
                "habits": [],
                "preferences": {}
            }
            save_user_profile(default_profile)
            return default_profile
        
        with open(USER_PROFILE_PATH, "r", encoding="utf-8") as f:
            profile = json.load(f)
            # Ensure all required fields are present
            default_profile = {
                "name": "",
                "interests": [],
                "communication_style": "",
                "habits": [],
                "preferences": {}
            }
            for key in default_profile:
                if key not in profile:
                    profile[key] = default_profile[key]
            return profile
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing user profile JSON: {e}")
        raise ValueError("Профиль пользователя содержит невалидный JSON. Попробуйте восстановить файл.")
    except Exception as e:
        logger.error(f"Error loading user profile: {e}")
        raise ValueError(f"Ошибка при загрузке профиля: {e}")


def save_user_profile(profile: dict) -> None:
    """Save user profile to JSON file."""
    try:
        # Create directory if it doesn't exist
        USER_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        with open(USER_PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving user profile: {e}")
        raise ValueError(f"Ошибка при сохранении профиля: {e}")


def build_me_system_prompt(profile: dict) -> str:
    """Build system prompt for personal assistant based on user profile."""
    profile_text = json.dumps(profile, ensure_ascii=False, indent=2)
    return f"""Ты — персональный агент пользователя. Вот что ты о нем знаешь:

{profile_text}

Твоя задача — помогать ему, исходя из его привычек и интересов. Отвечай в его любимом стиле общения."""


def update_profile_from_text(text: str) -> dict:
    """Update user profile by extracting new facts from text via LLM."""
    try:
        # Load current profile
        current_profile = load_user_profile()
        
        # Create prompt for LLM
        profile_structure = json.dumps({
            "name": "",
            "interests": [],
            "communication_style": "",
            "habits": [],
            "preferences": {}
        }, ensure_ascii=False, indent=2)
        
        update_prompt = f"""Извлеки из этого сообщения новые факты о пользователе и верни обновленный JSON-профиль.

Текущий профиль:
{json.dumps(current_profile, ensure_ascii=False, indent=2)}

Сообщение пользователя:
{text}

ВАЖНО:
1. Сохрани все существующие данные из текущего профиля
2. Добавь новые факты из сообщения
3. Обнови существующие поля, если в сообщении есть более актуальная информация
4. Верни ТОЛЬКО валидный JSON без дополнительных объяснений
5. Структура должна соответствовать этой схеме:
{profile_structure}

Верни только JSON объект."""
        
        messages = [
            {"role": "user", "content": update_prompt}
        ]
        
        # Send request to LLM via OpenRouter
        response = chat_completion(messages, temperature=0.3, model=ME_MODEL)
        
        if not response:
            raise ValueError("Модель не вернула ответ при обновлении профиля")
        
        # Try to extract JSON from response (may be wrapped in markdown code blocks)
        response_clean = response.strip()
        
        # Remove markdown code blocks if present
        if response_clean.startswith("```json"):
            response_clean = response_clean[7:]
        elif response_clean.startswith("```"):
            response_clean = response_clean[3:]
        
        if response_clean.endswith("```"):
            response_clean = response_clean[:-3]
        
        response_clean = response_clean.strip()
        
        # Parse JSON
        try:
            updated_profile = json.loads(response_clean)
            
            # Validate structure
            required_keys = {"name", "interests", "communication_style", "habits", "preferences"}
            if not all(key in updated_profile for key in required_keys):
                raise ValueError("Профиль не содержит все необходимые поля")
            
            # Ensure interests and habits are lists
            if not isinstance(updated_profile.get("interests"), list):
                updated_profile["interests"] = []
            if not isinstance(updated_profile.get("habits"), list):
                updated_profile["habits"] = []
            if not isinstance(updated_profile.get("preferences"), dict):
                updated_profile["preferences"] = {}
            
            return updated_profile
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from LLM response: {e}")
            logger.error(f"Response was: {response_clean[:500]}")
            raise ValueError("Модель вернула невалидный JSON. Попробуйте еще раз или обновите профиль вручную.")
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error updating profile from text: {e}")
        raise ValueError(f"Ошибка при обновлении профиля: {e}")
