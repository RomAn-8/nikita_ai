"""RAG command handlers."""

from telegram import Update
from telegram.ext import ContextTypes

from ..core.errors import safe_reply_text
from ..handlers.base import Handler
from ..embeddings import process_readme_file, process_docs_folder, clear_all_embeddings
from ..utils.helpers import reset_tz, reset_forest
import logging

logger = logging.getLogger(__name__)


class EmbedCreateHandler(Handler):
    """Handler for /embed_create command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /embed_create command."""
        if not update.message:
            return
        
        context.user_data["waiting_for_readme"] = True
        
        await safe_reply_text(
            update,
            "✅ Ожидаю .md файл.\n"
            "Пожалуйста, отправьте любой .md файл в чат (как документ)."
        )


class EmbedDocsHandler(Handler):
    """Handler for /embed_docs command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /embed_docs command."""
        if not update.message:
            return
        
        await update.message.chat.send_action("typing")
        
        try:
            result = process_docs_folder(replace_existing=True)
            
            if not result["success"]:
                error_msg = result.get("error", "Неизвестная ошибка")
                await safe_reply_text(
                    update,
                    f"❌ Ошибка при индексации папки docs/: {error_msg}\n"
                    f"Обработано файлов: {result.get('files_processed', 0)}/{result.get('total_files', 0)}"
                )
                return
            
            stats = []
            stats.append(f"✅ Эмбеддинги успешно созданы для папки docs/!")
            stats.append(f"📁 Обработано файлов: {result['files_processed']}/{result['total_files']}")
            stats.append(f"📦 Всего чанков: {result['total_chunks']}")
            stats.append("")
            
            if result.get("results"):
                stats.append("📄 Обработанные файлы:")
                for file_result in result["results"]:
                    if file_result.get("status") == "success":
                        stats.append(f"  ✅ {file_result['file']} ({file_result['chunks']} чанков)")
                    else:
                        stats.append(f"  ❌ {file_result['file']}: {file_result.get('error', 'Ошибка')}")
            
            if result.get("errors"):
                stats.append("")
                stats.append("⚠️ Ошибки:")
                for error in result["errors"][:5]:
                    stats.append(f"  - {error}")
                if len(result["errors"]) > 5:
                    stats.append(f"  ... и еще {len(result['errors']) - 5} ошибок")
            
            response_text = "\n".join(stats)
            await safe_reply_text(update, response_text)
            
        except Exception as e:
            logger.exception(f"Error in embed_docs_cmd: {e}")
            await safe_reply_text(update, f"❌ Ошибка при индексации папки docs/: {e}")


class RagModelHandler(Handler):
    """Handler for /rag_model command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /rag_model command."""
        if not update.message:
            return
        
        agent_context = self.get_agent_context(update, context)
        agent_context.update_mode("rag")
        context.user_data["rag_submode"] = "rag_filter"
        reset_tz(context)
        reset_forest(context)
        
        await safe_reply_text(
            update,
            "✅ Режим RAG активирован. Доступны 3 режима:\n"
            "- \"RAG+фильтр\" или \"RAG+фильтр <вопрос>\" - поиск с порогом похожести (по умолчанию)\n"
            "- \"RAG без фильтра\" или \"RAG без фильтра <вопрос>\" - поиск без порога\n"
            "- \"Без RAG\" или \"Без RAG <вопрос>\" - обычный ответ без поиска\n\n"
            "После выбора режима можно просто задавать вопросы - режим сохраняется."
        )


class ClearEmbeddingsHandler(Handler):
    """Handler for /clear_embeddings command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /clear_embeddings command."""
        try:
            deleted_count = clear_all_embeddings()
            if deleted_count > 0:
                logger.info(f"Cleared {deleted_count} embedding chunks from database")
                await safe_reply_text(update, f"✅ Удалено {deleted_count} эмбеддингов из базы данных.")
            else:
                await safe_reply_text(update, "ℹ️ Эмбеддинги не найдены в базе данных.")
        except Exception as e:
            logger.exception(f"Error clearing embeddings: {e}")
            await safe_reply_text(update, f"❌ Ошибка при удалении эмбеддингов: {e}")


async def embed_create_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /embed_create."""
    handler = EmbedCreateHandler()
    await handler.handle(update, context)


async def embed_docs_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /embed_docs."""
    handler = EmbedDocsHandler()
    await handler.handle(update, context)


async def rag_model_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /rag_model."""
    handler = RagModelHandler()
    await handler.handle(update, context)


async def clear_embeddings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /clear_embeddings."""
    handler = ClearEmbeddingsHandler()
    await handler.handle(update, context)
