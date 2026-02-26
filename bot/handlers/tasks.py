"""Task list handler."""

from telegram import Update
from telegram.ext import ContextTypes

from ..core.errors import safe_reply_text
from ..handlers.base import Handler
from ..utils.helpers import reset_tz, reset_forest


class TaskListHandler(Handler):
    """Handler for /task_list command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /task_list command."""
        if not update.message:
            return
        
        agent_context = self.get_agent_context(update, context)
        agent_context.update_mode("task_list")
        reset_tz(context)
        reset_forest(context)
        
        welcome_text = """✅ Режим работы с задачами активирован!

Теперь вы можете отправлять словесные команды для работы с задачами:

📝 Примеры команд:
• "Создай задачу на 15-02-2026 в 10:00 с приоритетом high: Подготовить презентацию"
• "Покажи задачи с приоритетом high"
• "Покажи невыполненные задачи"
• "Удали задачу в строке 5"
• "Покажи задачи с приоритетом high и предложи, что делать первым"

Для выхода из режима используйте команду /cancel или переключитесь на другой режим."""
        
        await safe_reply_text(update, welcome_text)


async def task_list_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /task_list."""
    handler = TaskListHandler()
    await handler.handle(update, context)
