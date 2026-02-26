"""Settings command handlers."""

from telegram import Update
from telegram.ext import ContextTypes

from ..core.errors import safe_reply_text
from ..core.context import AgentContext
from ..handlers.base import Handler
from ..services.database import db_set_temperature, db_set_memory_enabled, db_set_model
from ..services.context_manager import get_effective_model, clamp_temperature
from ..config import OPENROUTER_MODEL


class ChTemperatureHandler(Handler):
    """Handler for /ch_temperature command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /ch_temperature command."""
        agent_context = self.get_agent_context(update, context)
        chat_id = agent_context.chat_id
        
        if not context.args:
            # Show current temperature
            temp = agent_context.temperature
            await safe_reply_text(update, f"Текущая температура: {temp}")
            return
        
        try:
            new_temp = float(context.args[0])
            clamped_temp = clamp_temperature(new_temp)
            agent_context.update_temperature(clamped_temp)
            db_set_temperature(chat_id, clamped_temp)
            await safe_reply_text(update, f"Температура установлена: {clamped_temp}")
        except ValueError:
            await safe_reply_text(update, "❌ Неверный формат температуры. Используйте число, например: /ch_temperature 0.7")


class ChMemoryHandler(Handler):
    """Handler for /ch_memory command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /ch_memory command."""
        agent_context = self.get_agent_context(update, context)
        chat_id = agent_context.chat_id
        
        if not context.args:
            # Show current memory status
            mem = agent_context.memory_enabled
            status = "ВКЛ" if mem else "ВЫКЛ"
            await safe_reply_text(update, f"Память: {status}")
            return
        
        arg = context.args[0].lower().strip()
        if arg in ("on", "1", "true", "вкл", "да"):
            agent_context.update_memory_enabled(True)
            db_set_memory_enabled(chat_id, True)
            await safe_reply_text(update, "✅ Память включена")
        elif arg in ("off", "0", "false", "выкл", "нет"):
            agent_context.update_memory_enabled(False)
            db_set_memory_enabled(chat_id, False)
            await safe_reply_text(update, "❌ Память выключена")
        else:
            await safe_reply_text(update, "❌ Используйте: /ch_memory on или /ch_memory off")


class ClearMemoryHandler(Handler):
    """Handler for /clear_memory command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /clear_memory command."""
        agent_context = self.get_agent_context(update, context)
        chat_id = agent_context.chat_id
        
        from ..services.database import db_clear_messages
        db_clear_messages(chat_id)
        await safe_reply_text(update, "✅ Память очищена")


# Command functions for backward compatibility
async def ch_temperature_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /ch_temperature."""
    handler = ChTemperatureHandler()
    await handler.handle(update, context)


async def ch_memory_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /ch_memory."""
    handler = ChMemoryHandler()
    await handler.handle(update, context)


async def clear_memory_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /clear_memory."""
    handler = ClearMemoryHandler()
    await handler.handle(update, context)
