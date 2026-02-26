"""Base Handler class for command processing."""

from abc import ABC, abstractmethod
from typing import Any
from telegram import Update
from telegram.ext import ContextTypes

from ..core.context import AgentContext


class Handler(ABC):
    """Base class for all command handlers."""
    
    @abstractmethod
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle a command or message.
        
        Args:
            update: Telegram update object
            context: Telegram context object
        """
        pass
    
    def get_agent_context(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> AgentContext:
        """Create AgentContext from update and context."""
        return AgentContext.from_telegram_context(update, context)
