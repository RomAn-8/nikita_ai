"""Context management functions for getting/setting bot state."""

from telegram.ext import ContextTypes

from ..config import OPENROUTER_MODEL
from .database import (
    db_get_temperature,
    db_get_memory_enabled,
    db_get_model,
    DEFAULT_TEMPERATURE,
    DEFAULT_MEMORY_ENABLED,
)

# Constants
TEMPERATURE_MIN = 0.0
TEMPERATURE_MAX = 2.0


def get_mode(context: ContextTypes.DEFAULT_TYPE) -> str:
    """Get current mode from context."""
    return context.user_data.get("mode", "text")


def get_temperature(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> float:
    """Get temperature setting from context or database."""
    t = context.user_data.get("temperature", None)
    if isinstance(t, (int, float)):
        return float(t)

    db_t = db_get_temperature(chat_id)
    if isinstance(db_t, (int, float)):
        context.user_data["temperature"] = float(db_t)
        return float(db_t)

    context.user_data["temperature"] = float(DEFAULT_TEMPERATURE)
    return float(DEFAULT_TEMPERATURE)


def get_memory_enabled(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> bool:
    """Get memory enabled setting from context or database."""
    v = context.user_data.get("memory_enabled", None)
    if isinstance(v, bool):
        return v

    db_v = db_get_memory_enabled(chat_id)
    if isinstance(db_v, bool):
        context.user_data["memory_enabled"] = bool(db_v)
        return bool(db_v)

    context.user_data["memory_enabled"] = bool(DEFAULT_MEMORY_ENABLED)
    return bool(DEFAULT_MEMORY_ENABLED)


def get_model(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> str:
    """Get model setting from context or database."""
    v = context.user_data.get("model", None)
    if isinstance(v, str) and v.strip():
        return v.strip()

    db_v = db_get_model(chat_id)
    if isinstance(db_v, str) and db_v.strip():
        context.user_data["model"] = db_v.strip()
        return db_v.strip()

    # пустая строка => openrouter.py возьмёт OPENROUTER_MODEL из config
    return ""


def get_effective_model(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> str:
    """Get effective model (selected or default)."""
    selected = get_model(context, chat_id)
    return selected if selected else OPENROUTER_MODEL


def clamp_temperature(value: float) -> float:
    """Clamp temperature value to valid range."""
    if value < TEMPERATURE_MIN:
        return TEMPERATURE_MIN
    if value > TEMPERATURE_MAX:
        return TEMPERATURE_MAX
    return value
