"""Review PR handler."""

import os
from telegram import Update
from telegram.ext import ContextTypes

from ..core.errors import safe_reply_text
from ..handlers.base import Handler
from ..mcp_client import get_pr_info, get_pr_files, get_pr_diff
from ..services.llm import call_llm
from ..config import OPENROUTER_MODEL
from ..config import PR_REVIEW_AVAILABLE
from scripts.review_pr import get_rag_context as get_rag_context_for_pr, create_review_prompt
import logging

logger = logging.getLogger(__name__)


class ReviewPrHandler(Handler):
    """Handler for /review_pr command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /review_pr command."""
        if not PR_REVIEW_AVAILABLE:
            await safe_reply_text(
                update,
                "❌ Функция анализа PR недоступна. Убедитесь, что скрипт review_pr.py существует."
            )
            return
        
        if not update.message:
            return
        
        if not context.args or len(context.args) != 1:
            await safe_reply_text(
                update,
                "Использование: /review_pr <номер_pr>\n"
                "Пример: /review_pr 123\n\n"
                "Убедитесь, что:\n"
                "1. MCP сервер python-sdk запущен (http://127.0.0.1:8000/mcp)\n"
                "2. В переменных окружения установлен GB_TOKEN (или добавьте в .env)"
            )
            return
        
        try:
            pr_number = int(context.args[0])
        except ValueError:
            await safe_reply_text(update, f"❌ Номер PR должен быть числом, получено: {context.args[0]}")
            return
        
        await update.message.chat.send_action("typing")
        
        github_token = os.getenv("GB_TOKEN", "").strip() or os.getenv("GITHUB_TOKEN", "").strip()
        if not github_token:
            await safe_reply_text(
                update,
                "❌ GitHub token не найден в переменных окружения.\n"
                "Добавьте GB_TOKEN или GITHUB_TOKEN в .env файл или установите как переменную окружения."
            )
            return
        
        owner = "RomAn-8"
        repo = "nikita_ai"
        
        try:
            await safe_reply_text(update, f"📥 Получаю данные PR #{pr_number}...")
            try:
                pr_info = await get_pr_info(owner, repo, pr_number, github_token)
            except ValueError as e:
                error_msg = str(e)
                if "404" in error_msg or "не найден" in error_msg.lower():
                    await safe_reply_text(update, f"❌ PR #{pr_number} не найден в репозитории {owner}/{repo}.\nПроверьте номер PR.")
                elif "401" in error_msg or "Unauthorized" in error_msg:
                    await safe_reply_text(update, f"❌ Ошибка авторизации GitHub.\nПроверьте правильность GB_TOKEN в .env файле.")
                else:
                    await safe_reply_text(update, f"❌ Ошибка при получении информации о PR:\n{error_msg}\n\nПроверьте:\n1. MCP сервер запущен (http://127.0.0.1:8000/mcp)\n2. Правильность GB_TOKEN")
                return
            
            try:
                pr_files = await get_pr_files(owner, repo, pr_number, github_token)
            except ValueError as e:
                error_msg = str(e)
                await safe_reply_text(update, f"❌ Ошибка при получении файлов PR:\n{error_msg}\n\nПроверьте:\n1. MCP сервер запущен\n2. Правильность GB_TOKEN\n3. Доступ к репозиторию")
                return
            
            try:
                pr_diff = await get_pr_diff(owner, repo, pr_number, github_token)
            except ValueError as e:
                error_msg = str(e)
                await safe_reply_text(update, f"❌ Ошибка при получении diff PR:\n{error_msg}\n\nПроверьте:\n1. MCP сервер запущен\n2. Правильность GB_TOKEN")
                return
            
            pr_title = pr_info.get("title", "N/A")
            await safe_reply_text(update, f"✅ Получены данные PR: {pr_title}\n📁 Файлов изменено: {len(pr_files)}\n🔍 Ищу релевантную документацию...")
            
            rag_context = await get_rag_context_for_pr(pr_info, pr_files, pr_diff)
            if rag_context:
                await safe_reply_text(update, "✅ Найдена релевантная документация\n🤖 Генерирую ревью...")
            else:
                await safe_reply_text(update, "⚠️ Релевантная документация не найдена\n🤖 Генерирую ревью...")
            
            messages = create_review_prompt(pr_info, pr_files, pr_diff, rag_context)
            review_text = call_llm(messages, temperature=0.3, model=OPENROUTER_MODEL)
            
            if not review_text or not review_text.strip():
                await safe_reply_text(update, "❌ LLM вернул пустое ревью.")
                return
            
            max_length = 4000
            if len(review_text) <= max_length:
                await safe_reply_text(update, f"📝 **Ревью PR #{pr_number}:**\n\n{review_text}", parse_mode="Markdown")
            else:
                await safe_reply_text(update, f"📝 **Ревью PR #{pr_number}:**\n\n{review_text[:max_length]}...", parse_mode="Markdown")
                remaining = review_text[max_length:]
                while remaining:
                    chunk = remaining[:max_length]
                    remaining = remaining[max_length:]
                    await safe_reply_text(update, chunk, parse_mode="Markdown")
            
            await safe_reply_text(update, "✅ Анализ завершен!")
            
        except Exception as e:
            logger.exception(f"Error reviewing PR #{pr_number}: {e}")
            await safe_reply_text(update, f"❌ Ошибка при анализе PR: {e}")


async def review_pr_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /review_pr."""
    handler = ReviewPrHandler()
    await handler.handle(update, context)
