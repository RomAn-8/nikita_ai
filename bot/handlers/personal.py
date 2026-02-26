"""Personal assistant handler."""

from telegram import Update
from telegram.ext import ContextTypes

from ..core.errors import safe_reply_text
from ..handlers.base import Handler
from ..services.profile import load_user_profile
from ..config import ME_MODEL
from ..utils.helpers import reset_tz, reset_forest
import logging

logger = logging.getLogger(__name__)


class MeHandler(Handler):
    """Handler for /me command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /me command."""
        if not update.message:
            return
        
        agent_context = self.get_agent_context(update, context)
        agent_context.update_mode("me")
        reset_tz(context)
        reset_forest(context)
        
        try:
            profile = load_user_profile()
            profile_info = ""
            if profile.get("name"):
                profile_info = f"\n👤 Имя: {profile['name']}"
            if profile.get("interests"):
                profile_info += f"\n🎯 Интересы: {', '.join(profile['interests'][:3])}"
                if len(profile['interests']) > 3:
                    profile_info += "..."
        except Exception as e:
            logger.warning(f"Error loading profile in me_cmd: {e}")
            profile_info = "\n⚠️ Профиль не загружен. Используйте команду 'Обновить профиль' для создания профиля."
        
        await safe_reply_text(
            update,
            f"✅ Режим персонального ассистента активирован.\n"
            f"Модель: {ME_MODEL}\n"
            f"{profile_info}\n\n"
            f"Теперь все ваши сообщения будут обрабатываться через персонального ассистента.\n"
            f"Для выхода из режима используйте /mode_text или другой режим.\n\n"
        )


async def me_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /me."""
    handler = MeHandler()
    await handler.handle(update, context)
