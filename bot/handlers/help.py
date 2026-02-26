"""Help command handler."""

from telegram import Update
from telegram.ext import ContextTypes

from ..core.errors import safe_reply_text
from ..handlers.base import Handler
from ..services.context_manager import get_temperature, get_memory_enabled, get_model
from ..services.memory import add_message
from ..services.database import db_add_message
from ..services.llm import call_llm
from ..config import OPENROUTER_MODEL, RAG_SIM_THRESHOLD, RAG_TOP_K, EMBEDDING_MODEL
from ..embeddings import search_relevant_chunks, has_embeddings, list_indexed_documents
from ..mcp_client import get_git_branch
from ..utils.text import _short_model_name
from ..core.prompts import SYSTEM_PROMPT_TEXT
from ..services.database import build_messages_with_db_memory
from ..config import MODEL_GLM, MODEL_GEMMA, PR_REVIEW_AVAILABLE


class HelpHandler(Handler):
    """Handler for /help command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        if not update.message:
            return
        
        # If no args - show command list
        if not context.args:
            lines = [
                "📋 Доступные команды:",
                "",
                "🔧 Основные режимы:",
                f"/mode_text — режим text + {_short_model_name(OPENROUTER_MODEL)}",
                "/mode_json — JSON на каждое сообщение",
                f"/mode_summary — режим summary + {_short_model_name(OPENROUTER_MODEL)} (сжатие истории)",
                "/summary_debug — показать текущее summary (режим summary)",
            ]
            
            if MODEL_GLM:
                lines.append(f"/model_glm — модель {_short_model_name(MODEL_GLM)}")
            if MODEL_GEMMA:
                lines.append(f"/model_gemma — модель {_short_model_name(MODEL_GEMMA)}")
            
            lines.extend([
                "",
                "🤖 Специальные режимы:",
                "/tz_creation_site — собрать ТЗ на сайт (итог JSON)",
                "/forest_split — кто кому должен (итог текст)",
                "/thinking_model — решать пошагово",
                "/expert_group_model — группа экспертов",
                "",
                "⚙️ Настройки:",
                "/ch_temperature — показать/изменить температуру (пример: /ch_temperature 0.7)",
                "/ch_memory — память ВКЛ/ВЫКЛ (пример: /ch_memory off)",
                "/clear_memory — очистить память чата",
                "/clear_embeddings — удалить все эмбеддинги",
                "",
                "🧪 Тестирование:",
                "/tokens_test — тест токенов (включить режим)",
                "/tokens_next — тест токенов: следующий этап",
                "/tokens_stop — тест токенов: сводка и выход",
                "",
                "📚 RAG и эмбеддинги:",
                "/embed_create — создать эмбеддинги из .md файла (сначала отправьте файл)",
                "/embed_docs — создать эмбеддинги из всех файлов в папке docs/",
                "/rag_model — режим RAG",
                "",
                "💬 Словесные команды (в режиме RAG):",
                "• \"RAG+фильтр\" или \"RAG+фильтр <вопрос>\" — поиск с порогом похожести",
                "• \"RAG без фильтра\" или \"RAG без фильтра <вопрос>\" — поиск без порога",
                "• \"Без RAG\" или \"Без RAG <вопрос>\" — обычный ответ без поиска",
                "",
                "🌤️ Погода:",
                "/weather_sub — подписка на погоду (пример: /weather_sub Москва 30)",
                "/weather_sub_stop — остановить подписку (пример: /weather_sub_stop Москва)",
                "/digest — утренняя сводка: погода + новости (пример: /digest Москва, технологии)",
                "",
                "👤 Регистрация и записи:",
                "/register — регистрация (пример: /register Иванов Иван Иванович +79991234567)",
                "/unregister — удалить свою регистрацию",
                "/train_signup — запись на тренировку (пример: /train_signup 15-02-2026 18:00 [примечание])",
                "/train_move — перенос записи (пример: /train_move 1 16-02-2026 19:00)",
                "/train_cancel — отмена записи (пример: /train_cancel 1)",
                "/support — поддержка с RAG (пример: /support можно перенести запись?)",
                "/task_list — режим работы с задачами (словесные команды для создания, просмотра, удаления задач)",
                "",
                "🎤 Голосовой ассистент:",
                "/voice — голосовой ассистент (отправьте голосовое сообщение для распознавания и ответа, для выхода: /stop или /cancel)",
                "",
                "🤖 Локальные модели:",
                "/local_model — режим локальной модели Ollama (переключение режима, затем просто пишите сообщения)",
                "/analyze — анализ JSON файлов с логами через Ollama (отправьте JSON файл, затем задайте вопрос)",
                "/me — персональный ассистент (использует профиль пользователя, команды: 'Обновить профиль', 'Кто я?')",
                "",
                "🚀 Деплой:",
                "/deploy_bot — деплой бота на сервер (требует настройки переменных окружения)",
                "/stop_bot — остановить бота на сервере (опции: -v удалить данные, -i удалить образы)",
            ])
            
            if PR_REVIEW_AVAILABLE:
                lines.append("/review_pr — анализ Pull Request (пример: /review_pr 123)")
            
            lines.extend([
                "",
                "📖 Справка:",
                "/help — показать список команд или ответить на вопрос о проекте",
            ])
            
            await safe_reply_text(update, "\n".join(lines))
            return
        
        # If args provided - use RAG to answer question
        question_text = " ".join(context.args).strip()
        if not question_text:
            await safe_reply_text(update, "Пожалуйста, задайте вопрос о проекте. Пример: /help Как работает RAG система?")
            return
        
        await update.message.chat.send_action("typing")
        
        agent_context = self.get_agent_context(update, context)
        chat_id = agent_context.chat_id
        temperature = agent_context.temperature
        memory_enabled = agent_context.memory_enabled
        model = agent_context.model
        
        # Check for embeddings
        if not has_embeddings(EMBEDDING_MODEL):
            await safe_reply_text(
                update,
                "⚠️ Эмбеддинги не найдены в базе данных.\n"
                "Сначала создайте эмбеддинги с помощью команды /embed_create.\n"
                "Отправьте README.md и файлы из папки docs/ для индексации документации."
            )
            return
        
        # Check if question is about git branch
        question_lower = question_text.lower()
        is_git_branch_question = any(keyword in question_lower for keyword in [
            "ветка", "ветку", "ветки", "branch", "git branch", "текущая ветка",
            "какая ветка", "какую ветку", "какие ветки"
        ])
        
        # Get git branch via MCP (optional)
        git_branch_name = None
        try:
            git_branch_name = await get_git_branch()
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Не удалось получить git ветку через MCP: {e}")
        
        # If question about git branch and we got info - answer directly
        if is_git_branch_question and git_branch_name:
            await safe_reply_text(update, f"🌿 Текущая ветка git: `{git_branch_name}`")
            return
        elif is_git_branch_question and not git_branch_name:
            await safe_reply_text(
                update,
                "⚠️ Не удалось получить текущую ветку git.\n"
                "Убедитесь, что:\n"
                "- MCP сервер запущен (http://127.0.0.1:8000/mcp)\n"
                "- Текущая директория MCP сервера является git репозиторием"
            )
            return
        
        # Search relevant chunks
        filtered_chunks = []
        try:
            relevant_chunks = search_relevant_chunks(
                question_text,
                model=EMBEDDING_MODEL,
                top_k=RAG_TOP_K,
                min_similarity=RAG_SIM_THRESHOLD,
                apply_threshold=True
            )
            filtered_chunks = [chunk for chunk in relevant_chunks if chunk["similarity"] >= RAG_SIM_THRESHOLD]
            
            if not filtered_chunks:
                relevant_chunks_no_threshold = search_relevant_chunks(
                    question_text,
                    model=EMBEDDING_MODEL,
                    top_k=RAG_TOP_K * 2,
                    min_similarity=0.0,
                    apply_threshold=False
                )
                filtered_chunks = [chunk for chunk in relevant_chunks_no_threshold if chunk["similarity"] > 0.3]
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception(f"Error searching relevant chunks: {e}")
            await safe_reply_text(update, f"Ошибка при поиске релевантных фрагментов: {e}")
            return
        
        if not filtered_chunks:
            indexed_docs = list_indexed_documents(EMBEDDING_MODEL)
            error_msg = "⚠️ Не нашла релевантных фрагментов в документации для ответа на ваш вопрос."
            if indexed_docs:
                error_msg += f"\n\nПроиндексированные документы: {', '.join(indexed_docs[:5])}"
                if len(indexed_docs) > 5:
                    error_msg += f" и еще {len(indexed_docs) - 5}"
            else:
                error_msg += "\n\nДокументация не проиндексирована. Используйте:\n"
                error_msg += "- `/embed_create` для индексации README.md\n"
                error_msg += "- `/embed_docs` для индексации папки docs/"
            error_msg += "\n\nПопробуйте переформулировать вопрос или проиндексировать документацию."
            await safe_reply_text(update, error_msg)
            return
        
        # Build context for LLM
        context_parts = ["Релевантная информация из документации проекта:\n"]
        for i, chunk in enumerate(filtered_chunks, 1):
            context_parts.append(f"[Фрагмент {i} (doc_name={chunk['doc_name']}, chunk_index={chunk['chunk_index']}, score={chunk['similarity']:.4f})]:")
            context_parts.append(chunk["text"])
            context_parts.append("")
        
        context_parts.append(f"Вопрос пользователя о проекте: {question_text}")
        if git_branch_name:
            context_parts.append(f"\nТекущая ветка git: {git_branch_name}")
        context_parts.append("\nОтветь на вопрос пользователя, используя информацию из документации выше.")
        context_parts.append("Если информация недостаточна, укажи это в ответе.")
        
        user_content = "\n".join(context_parts)
        
        # Build messages for LLM
        system_prompt = SYSTEM_PROMPT_TEXT
        if memory_enabled:
            messages = build_messages_with_db_memory(system_prompt, chat_id=chat_id)
        else:
            messages = [{"role": "system", "content": system_prompt}]
        
        messages.append({"role": "user", "content": user_content})
        
        # Call LLM
        try:
            answer = call_llm(messages, temperature=temperature, model=model)
            answer = (answer or "").strip() or "Пустой ответ от модели."
        except Exception as e:
            await safe_reply_text(update, f"Ошибка запроса к LLM: {e}")
            return
        
        # Save to DB
        mode = "text"
        db_add_message(chat_id, mode, "user", f"/help {question_text}")
        db_add_message(chat_id, mode, "assistant", answer)
        
        await safe_reply_text(update, answer)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /help."""
    handler = HelpHandler()
    await handler.handle(update, context)
