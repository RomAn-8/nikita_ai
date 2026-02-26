"""Voice assistant handler."""

from telegram import Update
from telegram.ext import ContextTypes

from ..core.errors import safe_reply_text
from ..handlers.base import Handler
from ..config import VOICE_MODEL
from ..utils.helpers import reset_tz, reset_forest


class VoiceHandler(Handler):
    """Handler for /voice command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /voice command."""
        if not update.message:
            return
        
        agent_context = self.get_agent_context(update, context)
        agent_context.update_mode("voice")
        reset_tz(context)
        reset_forest(context)
        
        await safe_reply_text(
            update,
            f"✅ Режим голосового ассистента включён 🎤\n"
            f"Модель: {VOICE_MODEL}\n\n"
            f"Отправь голосовое сообщение"
        )


async def voice_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /voice."""
    handler = VoiceHandler()
    await handler.handle(update, context)
