"""Mode command handlers."""

import json
from telegram import Update
from telegram.ext import ContextTypes

from ..core.errors import safe_reply_text
from ..core.context import AgentContext
from ..handlers.base import Handler
from ..services.database import db_set_model, db_set_memory_enabled, utc_now_iso
from ..services.context_manager import get_effective_model
from ..config import OPENROUTER_MODEL
from ..summarizer import MODE_SUMMARY


def reset_tz(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reset TZ mode state."""
    context.user_data.pop("tz_history", None)
    context.user_data.pop("tz_questions", None)
    context.user_data.pop("tz_done", None)


def reset_forest(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reset forest mode state."""
    context.user_data.pop("forest_history", None)
    context.user_data.pop("forest_questions", None)
    context.user_data.pop("forest_done", None)
    context.user_data.pop("forest_result", None)


class ModeTextHandler(Handler):
    """Handler for /mode_text command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /mode_text command."""
        agent_context = self.get_agent_context(update, context)
        chat_id = agent_context.chat_id
        
        agent_context.update_mode("text")
        reset_tz(context)
        reset_forest(context)
        
        # Reset to default model
        agent_context.update_model(None)
        db_set_model(chat_id, "")
        
        model = get_effective_model(context, chat_id)
        await safe_reply_text(update, f"Ок. Режим: text. Модель: {model}")


class ModeJsonHandler(Handler):
    """Handler for /mode_json command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /mode_json command."""
        agent_context = self.get_agent_context(update, context)
        
        agent_context.update_mode("json")
        reset_tz(context)
        reset_forest(context)
        
        payload = {
            "title": "Режим установлен",
            "time": utc_now_iso(),
            "tag": "system",
            "answer": "Ок. Режим установлен: json",
            "steps": [],
            "warnings": [],
            "need_clarification": False,
            "clarifying_question": "",
        }
        context.user_data["last_payload"] = payload
        await safe_reply_text(update, json.dumps(payload, ensure_ascii=False, indent=2))


class ModeSummaryHandler(Handler):
    """Handler for /mode_summary command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /mode_summary command."""
        agent_context = self.get_agent_context(update, context)
        chat_id = agent_context.chat_id
        
        agent_context.update_mode(MODE_SUMMARY)
        reset_tz(context)
        reset_forest(context)
        
        # Memory is always needed in summary mode
        agent_context.update_memory_enabled(True)
        db_set_memory_enabled(chat_id, True)
        
        await safe_reply_text(update, "Ок. Режим: summary (сжатие истории: summary вместо полной истории).")


class ThinkingModelHandler(Handler):
    """Handler for /thinking_model command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /thinking_model command."""
        agent_context = self.get_agent_context(update, context)
        
        agent_context.update_mode("thinking")
        reset_tz(context)
        reset_forest(context)
        
        await safe_reply_text(update, "Ок. Режим: thinking (решаю пошагово).")


class ExpertGroupModelHandler(Handler):
    """Handler for /expert_group_model command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /expert_group_model command."""
        agent_context = self.get_agent_context(update, context)
        
        agent_context.update_mode("experts")
        reset_tz(context)
        reset_forest(context)
        
        await safe_reply_text(update, "Ок. Режим: experts (группа экспертов).")


# Command functions for backward compatibility
async def mode_text_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /mode_text."""
    handler = ModeTextHandler()
    await handler.handle(update, context)


async def mode_json_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /mode_json."""
    handler = ModeJsonHandler()
    await handler.handle(update, context)


async def mode_summary_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /mode_summary."""
    handler = ModeSummaryHandler()
    await handler.handle(update, context)


async def thinking_model_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /thinking_model."""
    handler = ThinkingModelHandler()
    await handler.handle(update, context)


async def expert_group_model_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /expert_group_model."""
    handler = ExpertGroupModelHandler()
    await handler.handle(update, context)
