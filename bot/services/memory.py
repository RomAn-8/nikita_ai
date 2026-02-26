"""Memory service for chat history management."""

from typing import Any
from ..services.database import (
    db_add_message,
    db_get_messages,
    db_clear_messages,
    MEMORY_LIMIT_MESSAGES,
    MEMORY_CHAT_MODES,
)


def add_message(chat_id: int, mode: str, role: str, content: str) -> None:
    """Add message to memory."""
    db_add_message(chat_id, mode, role, content)


def get_messages(chat_id: int, mode: str, limit: int = MEMORY_LIMIT_MESSAGES) -> list[dict[str, str]]:
    """Get messages from memory."""
    return db_get_messages(chat_id, mode, limit)


def clear_messages(chat_id: int, mode: str | None = None) -> None:
    """Clear messages from memory."""
    db_clear_messages(chat_id, mode)


def build_messages_with_memory(system_prompt: str, chat_id: int, modes: tuple[str, ...] | None = None) -> list[dict[str, str]]:
    """
    Build messages list with system prompt and memory.
    
    Args:
        system_prompt: System prompt text
        chat_id: Chat ID
        modes: Modes to include in memory (default: MEMORY_CHAT_MODES)
        
    Returns:
        List of messages with system prompt and history
    """
    if modes is None:
        modes = MEMORY_CHAT_MODES
    
    # Get history from all specified modes
    history: list[dict[str, str]] = []
    for mode in modes:
        history.extend(db_get_messages(chat_id, mode, MEMORY_LIMIT_MESSAGES))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_history: list[dict[str, str]] = []
    for msg in history:
        key = (msg.get("role"), msg.get("content"))
        if key not in seen:
            seen.add(key)
            unique_history.append(msg)
    
    # Sort by insertion order (assuming messages are in chronological order)
    return [{"role": "system", "content": system_prompt}] + unique_history
