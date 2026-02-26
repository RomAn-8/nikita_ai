"""Local model and analyze handlers."""

import re
from telegram import Update
from telegram.ext import ContextTypes

from ..core.errors import safe_reply_text
from ..handlers.base import Handler
from ..config import OLLAMA_MODEL, OLLAMA_TEMPERATURE, OLLAMA_NUM_CTX, OLLAMA_NUM_PREDICT, OLLAMA_SYSTEM_PROMPT
from ..utils.helpers import reset_tz, reset_forest
from ..services.llm import send_to_ollama, get_ollama_settings_display


class LocalModelHandler(Handler):
    """Handler for /local_model command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /local_model command."""
        if not update.message:
            return
        
        if not context.args:
            agent_context = self.get_agent_context(update, context)
            agent_context.update_mode("local_model")
            reset_tz(context)
            reset_forest(context)
            
            settings_text = _get_ollama_settings_display(context.user_data)
            
            await safe_reply_text(
                update,
                f"✅ Режим локальной модели Ollama активирован.\n"
                f"Модель: {OLLAMA_MODEL}\n\n"
                f"{settings_text}\n\n"
                f"Теперь все ваши сообщения будут обрабатываться через локальную модель.\n"
                f"Для выхода из режима используйте /mode_text или другой режим.\n\n"
                f"💡 Команды для изменения настроек:\n"
                f"• \"изменить температуру 0.7\"\n"
                f"• \"изменить контекстное окно 4096\"\n"
                f"• \"изменить максимальную длину ответа 512\"\n"
                f"• \"показать текущие настройки модели\"\n"
                f"• \"сбросить настройки модели\""
            )
            return
        
        text = " ".join(context.args).strip().lower()
        
        temp_match = re.search(r'изменить\s+температуру\s+([\d.]+)', text)
        if temp_match:
            try:
                new_temp = float(temp_match.group(1))
                if 0.0 <= new_temp <= 2.0:
                    context.user_data["ollama_temperature"] = new_temp
                    await safe_reply_text(update, f"✅ Температура изменена на {new_temp}")
                else:
                    await safe_reply_text(update, "❌ Температура должна быть в диапазоне от 0.0 до 2.0")
                return
            except ValueError:
                await safe_reply_text(update, "❌ Неверный формат температуры")
                return
        
        ctx_match = re.search(r'изменить\s+контекстное\s+окно\s+(\d+)', text)
        if ctx_match:
            try:
                new_ctx = int(ctx_match.group(1))
                if new_ctx > 0:
                    context.user_data["ollama_num_ctx"] = new_ctx
                    await safe_reply_text(update, f"✅ Контекстное окно изменено на {new_ctx}")
                else:
                    await safe_reply_text(update, "❌ Контекстное окно должно быть больше 0")
                return
            except ValueError:
                await safe_reply_text(update, "❌ Неверный формат контекстного окна")
                return
        
        predict_match = re.search(r'изменить\s+максимальную\s+длину\s+ответа\s+(\d+)', text)
        if predict_match:
            try:
                new_predict = int(predict_match.group(1))
                if new_predict > 0:
                    context.user_data["ollama_num_predict"] = new_predict
                    await safe_reply_text(update, f"✅ Максимальная длина ответа изменена на {new_predict}")
                else:
                    await safe_reply_text(update, "❌ Максимальная длина ответа должна быть больше 0")
                return
            except ValueError:
                await safe_reply_text(update, "❌ Неверный формат максимальной длины ответа")
                return
        
        if "показать текущие настройки модели" in text or "показать настройки" in text:
            settings_text = get_ollama_settings_display(context.user_data)
            await safe_reply_text(update, settings_text)
            return
        
        if "сбросить настройки модели" in text or "сбросить настройки" in text:
            context.user_data.pop("ollama_temperature", None)
            context.user_data.pop("ollama_num_ctx", None)
            context.user_data.pop("ollama_num_predict", None)
            context.user_data.pop("ollama_system_prompt", None)
            settings_text = get_ollama_settings_display(context.user_data)
            await safe_reply_text(update, f"✅ Настройки сброшены к значениям по умолчанию:\n\n{settings_text}")
            return
        
        question = " ".join(context.args)
        
        try:
            answer = await send_to_ollama(question, context.user_data)
            await safe_reply_text(update, answer)
        except ValueError as e:
            await safe_reply_text(update, f"❌ {str(e)}\n\n💡 Попробуйте сбросить настройки командой: сбросить настройки модели")
        except ConnectionError as e:
            await safe_reply_text(update, f"❌ {str(e)}")
        except Exception as e:
            await safe_reply_text(update, f"❌ Ошибка при обработке запроса: {str(e)}")


class AnalyzeHandler(Handler):
    """Handler for /analyze command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /analyze command."""
        if not update.message:
            return
        
        agent_context = self.get_agent_context(update, context)
        agent_context.update_mode("analyze")
        context.user_data.pop("analyze_json_content", None)
        
        await safe_reply_text(update, "Отправь JSON файл с логами для анализа")


async def local_model_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /local_model."""
    handler = LocalModelHandler()
    await handler.handle(update, context)


async def analyze_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /analyze."""
    handler = AnalyzeHandler()
    await handler.handle(update, context)
