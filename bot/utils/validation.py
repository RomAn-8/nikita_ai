"""Validation utilities."""

from ..services.context_manager import TEMPERATURE_MIN, TEMPERATURE_MAX, clamp_temperature


def validate_temperature(value: float) -> float:
    """Validate and clamp temperature value."""
    return clamp_temperature(value)


def validate_chat_id(chat_id: Any) -> int:
    """Validate chat ID."""
    try:
        return int(chat_id)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid chat_id: {chat_id}")
