"""Special mode handlers (tz, forest)."""

from telegram import Update
from telegram.ext import ContextTypes

from ..core.errors import safe_reply_text
from ..handlers.base import Handler
from ..services.context_manager import get_temperature, get_model
from ..services.llm import call_llm
from ..utils.text import looks_like_json
from ..core.prompts import SYSTEM_PROMPT_TZ, SYSTEM_PROMPT_FOREST
from ..utils.helpers import reset_forest, reset_tz
from ..utils.tz_helpers import send_final_tz_json


class TzCreationSiteHandler(Handler):
    """Handler for /tz_creation_site command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /tz_creation_site command."""
        agent_context = self.get_agent_context(update, context)
        agent_context.update_mode("tz")
        context.user_data["tz_history"] = []
        context.user_data["tz_questions"] = 0
        context.user_data["tz_done"] = False
        reset_forest(context)
        
        chat_id = agent_context.chat_id
        temperature = agent_context.temperature
        model = agent_context.model
        
        first = (call_llm(
            [
                {"role": "system", "content": SYSTEM_PROMPT_TZ},
                {"role": "user", "content": "Начни. Задай первый вопрос, чтобы собрать требования для ТЗ на создание сайта."},
            ],
            temperature=temperature,
            model=model,
        ) or "").strip()
        
        if looks_like_json(first):
            await send_final_tz_json(update, context, first, temperature=temperature, model=model)
            return
        
        context.user_data["tz_questions"] = 1
        context.user_data["tz_history"].append({"role": "assistant", "content": first})
        await safe_reply_text(update, first)


class ForestSplitHandler(Handler):
    """Handler for /forest_split command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /forest_split command."""
        agent_context = self.get_agent_context(update, context)
        agent_context.update_mode("forest")
        context.user_data["forest_history"] = []
        context.user_data["forest_questions"] = 0
        context.user_data["forest_done"] = False
        context.user_data.pop("forest_result", None)
        reset_tz(context)
        
        chat_id = agent_context.chat_id
        temperature = agent_context.temperature
        model = agent_context.model
        
        first = (call_llm(
            [
                {"role": "system", "content": SYSTEM_PROMPT_FOREST},
                {"role": "user", "content": "Начни. Задай первый вопрос для расчёта кто кому сколько должен."},
            ],
            temperature=temperature,
            model=model,
        ) or "").strip()
        
        context.user_data["forest_questions"] = 1
        context.user_data["forest_history"].append({"role": "assistant", "content": first})
        await safe_reply_text(update, first)


async def tz_creation_site_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /tz_creation_site."""
    handler = TzCreationSiteHandler()
    await handler.handle(update, context)


async def forest_split_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /forest_split."""
    handler = ForestSplitHandler()
    await handler.handle(update, context)
