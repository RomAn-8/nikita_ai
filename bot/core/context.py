"""AgentContext for managing bot state."""

from dataclasses import dataclass
from typing import Any
from telegram.ext import ContextTypes


@dataclass
class AgentContext:
    """Unified context object for agent operations."""
    
    chat_id: int
    user_id: int
    mode: str
    temperature: float
    memory_enabled: bool
    model: str | None
    user_data: dict[str, Any]
    context: ContextTypes.DEFAULT_TYPE
    
    @classmethod
    def from_telegram_context(
        cls,
        update: Any,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int | None = None,
        user_id: int | None = None,
    ) -> "AgentContext":
        """Create AgentContext from Telegram update and context."""
        if chat_id is None:
            chat_id = int(update.effective_chat.id) if update.effective_chat else 0
        if user_id is None:
            user_id = int(update.effective_user.id) if update.effective_user else 0
        
        # Import here to avoid circular imports
        from ..services.context_manager import (
            get_mode,
            get_temperature,
            get_memory_enabled,
            get_model,
        )
        
        return cls(
            chat_id=chat_id,
            user_id=user_id,
            mode=get_mode(context),
            temperature=get_temperature(context, chat_id),
            memory_enabled=get_memory_enabled(context, chat_id),
            model=get_model(context, chat_id) or None,
            user_data=context.user_data,
            context=context,
        )
    
    def update_mode(self, mode: str) -> None:
        """Update the current mode."""
        self.user_data["mode"] = mode
        self.mode = mode
    
    def update_temperature(self, temperature: float) -> None:
        """Update the temperature setting."""
        self.user_data["temperature"] = temperature
        self.temperature = temperature
    
    def update_memory_enabled(self, enabled: bool) -> None:
        """Update the memory enabled setting."""
        self.user_data["memory_enabled"] = enabled
        self.memory_enabled = enabled
    
    def update_model(self, model: str | None) -> None:
        """Update the model setting."""
        if model:
            self.user_data["model"] = model
            self.model = model
        else:
            self.user_data.pop("model", None)
            self.model = None
