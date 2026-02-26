"""Digest command handler."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from telegram import Update
from telegram.ext import ContextTypes

from ..core.errors import safe_reply_text
from ..handlers.base import Handler
from ..services.context_manager import get_temperature, get_model
from ..services.llm import call_llm
from ..mcp_weather import get_weather_via_mcp
from ..mcp_news import get_news_via_mcp
from ..summarizer import MODE_SUMMARY
from ..utils.helpers import _city_prepositional_case
import logging

logger = logging.getLogger(__name__)


class DigestHandler(Handler):
    """Handler for /digest command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /digest command."""
        if not update.message:
            return
        
        agent_context = self.get_agent_context(update, context)
        chat_id = agent_context.chat_id
        
        if not context.args:
            await safe_reply_text(
                update,
                "Использование: /digest <город погоды>, <тема новостей>\n"
                "Пример: /digest Москва, технологии\n"
                "Пример: /digest Самара, спорт"
            )
            return
        
        full_text = " ".join(context.args)
        parts = [p.strip() for p in full_text.split(",", 1)]
        
        if len(parts) < 2:
            await safe_reply_text(
                update,
                "Неверный формат. Используйте: /digest <город>, <тема>\n"
                "Пример: /digest Москва, технологии"
            )
            return
        
        city = parts[0]
        news_topic = parts[1]
        
        if not city or not news_topic:
            await safe_reply_text(update, "Город и тема новостей должны быть указаны.")
            return
        
        await update.message.chat.send_action("typing")
        
        city_prep = _city_prepositional_case(city)
        
        weather_text = await get_weather_via_mcp(city)
        news_text = await get_news_via_mcp(news_topic, count=5)
        
        SAMARA_OFFSET = timedelta(hours=4)
        SAMARA_TIMEZONE = timezone(SAMARA_OFFSET)
        now = datetime.now(SAMARA_TIMEZONE)
        date_str = now.strftime("%d.%m.%Y %H:%M")
        
        markdown_content = f"""# Сводка погоды в {city_prep} и новости по теме {news_topic}
**Дата:** {date_str}

## Погода: {city}

{weather_text}

## Новости: {news_topic}

{news_text}

---
*Сгенерировано автоматически*
"""
        
        digest_dir = Path(__file__).resolve().parent.parent / "digests"
        digest_dir.mkdir(exist_ok=True)
        filename = f"digest_{chat_id}_{now.strftime('%Y%m%d_%H%M%S')}.md"
        filepath = digest_dir / filename
        
        try:
            filepath.write_text(markdown_content, encoding="utf-8")
        except Exception as e:
            logger.exception(f"Failed to save digest file: {e}")
            await safe_reply_text(update, f"Ошибка при сохранении файла: {e}")
            return
        
        mode = MODE_SUMMARY
        temperature = get_temperature(context, chat_id)
        model = get_model(context, chat_id) or None
        
        system_prompt = """Ты помощник, который формирует сводку на основе данных о погоде и новостях.
Сделай сводку краткой, информативной и приятной для чтения.
Используй данные о погоде и новостях, которые тебе предоставлены."""
        
        user_prompt = f"""Создай сводку на основе следующих данных:

Погода в {city}:
{weather_text}

Новости по теме {news_topic}:
{news_text}

Создай краткую, информативную сводку."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            summary = call_llm(messages, temperature=temperature, model=model)
            if summary:
                await safe_reply_text(update, summary)
            else:
                await safe_reply_text(update, "Не удалось создать сводку.")
        except Exception as e:
            logger.exception(f"Error creating digest: {e}")
            await safe_reply_text(update, f"Ошибка при создании сводки: {e}")


async def digest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /digest."""
    handler = DigestHandler()
    await handler.handle(update, context)
