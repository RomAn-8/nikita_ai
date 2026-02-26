"""Message router for handling different modes."""

from typing import Any, Callable, Awaitable
from telegram import Update
from telegram.ext import ContextTypes

from .context import AgentContext


class MessageRouter:
    """Router for routing messages to appropriate handlers based on mode."""
    
    def __init__(self):
        self._handlers: dict[str, Callable[[Update, ContextTypes.DEFAULT_TYPE, AgentContext, str], Awaitable[None]]] = {}
    
    def register(self, mode: str, handler: Callable[[Update, ContextTypes.DEFAULT_TYPE, AgentContext, str], Awaitable[None]]) -> None:
        """Register a handler for a specific mode."""
        self._handlers[mode] = handler
    
    async def route(self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> bool:
        """
        Route a message to the appropriate handler.
        
        Args:
            update: Telegram update
            context: Telegram context
            text: Message text
            
        Returns:
            True if message was handled, False otherwise
        """
        agent_context = AgentContext.from_telegram_context(update, context)
        mode = agent_context.mode
        
        handler = self._handlers.get(mode)
        if handler:
            await handler(update, context, agent_context, text)
            return True
        
        return False


# Global router instance
_router = MessageRouter()


def get_router() -> MessageRouter:
    """Get the global message router."""
    return _router


def register_mode_handler(mode: str, handler: Callable[[Update, ContextTypes.DEFAULT_TYPE, AgentContext, str], Awaitable[None]]) -> None:
    """Register a mode handler in the global router."""
    _router.register(mode, handler)
