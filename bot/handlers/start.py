"""Start command handler."""

from telegram import Update
from telegram.ext import ContextTypes

from ..core.errors import safe_reply_text
from ..handlers.base import Handler
from ..services.context_manager import get_mode, get_temperature, get_memory_enabled, get_effective_model
from ..utils.text import _short_model_name
from ..config import OPENROUTER_MODEL
from ..config import MODEL_GLM, MODEL_GEMMA, PR_REVIEW_AVAILABLE


class StartHandler(Handler):
    """Handler for /start command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        agent_context = self.get_agent_context(update, context)
        mode = agent_context.mode
        chat_id = agent_context.chat_id
        t = agent_context.temperature
        mem = agent_context.memory_enabled
        current_model = get_effective_model(context, chat_id)
        
        lines = [
            "Привет! 👋",
            "",
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
            "",
            "🎮 Игры:",
            "/tictactoe — крестики-нолики против ИИ (игра через inline-кнопки)",
        ])

        if PR_REVIEW_AVAILABLE:
            lines.append("/review_pr — анализ Pull Request (пример: /review_pr 123)")
        
        lines.extend([
            "",
            "📖 Справка:",
            "/help — показать список команд или ответить на вопрос о проекте",
        ])
        
        lines.extend([
            "",
            f"Текущий режим: {mode}",
            f"Температура: {t}",
            f"Память: {'ВКЛ' if mem else 'ВЫКЛ'}",
            f"Модель: {current_model}",
        ])
        
        await safe_reply_text(update, "\n".join(lines))


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /start."""
    handler = StartHandler()
    await handler.handle(update, context)
