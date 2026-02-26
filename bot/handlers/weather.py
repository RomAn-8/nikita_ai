"""Weather command handlers."""

from telegram import Update
from telegram.ext import ContextTypes

from ..core.errors import safe_reply_text
from ..handlers.base import Handler
from ..weather_subscription import start_weather_subscription, stop_weather_subscription
from ..services.database import db_add_message
import logging

logger = logging.getLogger(__name__)


class WeatherSubHandler(Handler):
    """Handler for /weather_sub command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /weather_sub command."""
        agent_context = self.get_agent_context(update, context)
        chat_id = agent_context.chat_id
        
        if not context.args or len(context.args) < 2:
            await safe_reply_text(
                update,
                "Использование: /weather_sub <Город> <время_в_секундах>\n"
                "Пример: /weather_sub Москва 30\n"
                "Подписка будет собирать погоду каждые 10 секунд и отправлять summary каждые указанные секунды.",
            )
            return
        
        city = context.args[0].strip()
        try:
            summary_interval = int(context.args[1])
            if summary_interval < 10:
                await safe_reply_text(update, "Интервал summary должен быть не менее 10 секунд.")
                return
        except ValueError:
            await safe_reply_text(update, "Время должно быть числом (в секундах).")
            return
        
        try:
            start_weather_subscription(
                chat_id=chat_id,
                city=city,
                summary_interval=summary_interval,
                bot=context.bot,
                context=context,
                db_add_message=db_add_message,
            )
        except Exception as e:
            logger.exception(f"Failed to start weather subscription: {e}")
            await safe_reply_text(update, f"Ошибка при запуске подписки: {e}")


class WeatherSubStopHandler(Handler):
    """Handler for /weather_sub_stop command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /weather_sub_stop command."""
        agent_context = self.get_agent_context(update, context)
        chat_id = agent_context.chat_id
        
        if not context.args or len(context.args) < 1:
            await safe_reply_text(update, "Использование: /weather_sub_stop <Город>\nПример: /weather_sub_stop Москва")
            return
        
        city = context.args[0].strip()
        stopped = stop_weather_subscription(chat_id=chat_id, city=city, context=context)
        
        if stopped:
            await safe_reply_text(update, f"✅ Подписка на погоду для {city} остановлена.")
        else:
            await safe_reply_text(update, f"❌ Подписка на погоду для {city} не найдена.")


async def weather_sub_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /weather_sub."""
    handler = WeatherSubHandler()
    await handler.handle(update, context)


async def weather_sub_stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /weather_sub_stop."""
    handler = WeatherSubStopHandler()
    await handler.handle(update, context)
