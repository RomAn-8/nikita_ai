"""Model selection handlers."""

from telegram import Update
from telegram.ext import ContextTypes

from ..core.errors import safe_reply_text
from ..handlers.base import Handler
from ..services.database import db_set_model
from ..config import MODEL_GLM, MODEL_GEMMA


class ModelGlmHandler(Handler):
    """Handler for /model_glm command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /model_glm command."""
        if not MODEL_GLM:
            await safe_reply_text(update, "Модель OPENROUTER_MODEL_GLM не задана в .env")
            return
        
        agent_context = self.get_agent_context(update, context)
        chat_id = agent_context.chat_id
        
        agent_context.update_model(MODEL_GLM)
        db_set_model(chat_id, MODEL_GLM)
        await safe_reply_text(update, f"Ок. Модель установлена: {MODEL_GLM}")


class ModelGemmaHandler(Handler):
    """Handler for /model_gemma command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /model_gemma command."""
        if not MODEL_GEMMA:
            await safe_reply_text(update, "Модель OPENROUTER_MODEL_GEMMA не задана в .env")
            return
        
        agent_context = self.get_agent_context(update, context)
        chat_id = agent_context.chat_id
        
        agent_context.update_model(MODEL_GEMMA)
        db_set_model(chat_id, MODEL_GEMMA)
        await safe_reply_text(update, f"Ок. Модель установлена: {MODEL_GEMMA}")


async def model_glm_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /model_glm."""
    handler = ModelGlmHandler()
    await handler.handle(update, context)


async def model_gemma_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /model_gemma."""
    handler = ModelGemmaHandler()
    await handler.handle(update, context)
