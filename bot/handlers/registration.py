"""Registration and training handlers."""

from telegram import Update
from telegram.ext import ContextTypes

from ..core.errors import safe_reply_text
from ..handlers.base import Handler
from ..mcp_client import user_register, user_delete, reg_create, reg_find_by_user, reg_reschedule, reg_cancel, user_get
from ..embeddings import search_relevant_chunks, has_embeddings
from ..services.context_manager import get_temperature, get_model
from ..services.llm import call_llm
from ..config import EMBEDDING_MODEL, RAG_SIM_THRESHOLD, RAG_TOP_K
from ..core.prompts import SYSTEM_PROMPT_TEXT
import logging

logger = logging.getLogger(__name__)


class RegisterHandler(Handler):
    """Handler for /register command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /register command."""
        if not update.message:
            return
        
        if not context.args or len(context.args) < 2:
            await safe_reply_text(update, "Использование: /register <ФИО> <телефон>\nПример: /register Иванов Иван Иванович +79991234567")
            return
        
        username = update.effective_user.username
        if not username:
            await safe_reply_text(update, "❌ Ошибка: у вас не установлен username в Telegram. Пожалуйста, установите username в настройках Telegram и попробуйте снова.")
            return
        
        if len(context.args) > 2:
            fio = " ".join(context.args[:-1])
            phone = context.args[-1]
        else:
            fio = context.args[0]
            phone = context.args[1]
        
        try:
            result = await user_register(username, fio, phone)
            if result and result.get("status") == "registered":
                await safe_reply_text(update, "✅ Вы зарегистрированы")
            elif result and result.get("status") == "updated":
                await safe_reply_text(update, "✅ Данные обновлены")
            else:
                await safe_reply_text(update, "❌ Ошибка при регистрации")
        except ValueError as e:
            await safe_reply_text(update, f"❌ {e}")
        except Exception as e:
            logger.exception(f"Error in register_cmd: {e}")
            await safe_reply_text(update, f"❌ Неизвестная ошибка: {e}")


class UnregisterHandler(Handler):
    """Handler for /unregister command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /unregister command."""
        if not update.message:
            return
        
        username = update.effective_user.username
        if not username:
            await safe_reply_text(update, "❌ Ошибка: у вас не установлен username в Telegram. Пожалуйста, установите username в настройках Telegram.")
            return
        
        try:
            active_regs = []
            try:
                active_regs = await reg_find_by_user(username) or []
            except ValueError:
                pass
            
            if active_regs:
                await safe_reply_text(
                    update,
                    f"⚠️ У вас есть {len(active_regs)} активных записей. Сначала отмените их командой /train_cancel <reg_id>"
                )
                return
            
            result = await user_delete(username)
            if result:
                await safe_reply_text(update, "✅ Ваша регистрация удалена")
            else:
                await safe_reply_text(update, "❌ Ошибка при удалении регистрации")
        except ValueError as e:
            await safe_reply_text(update, f"❌ {e}")
        except Exception as e:
            logger.exception(f"Error in unregister_cmd: {e}")
            await safe_reply_text(update, f"❌ Неизвестная ошибка: {e}")


class TrainSignupHandler(Handler):
    """Handler for /train_signup command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /train_signup command."""
        if not update.message:
            return
        
        if not context.args or len(context.args) < 2:
            await safe_reply_text(update, "Использование: /train_signup <дата DD-MM-YYYY> <время HH:MM> [примечание]\nПример: /train_signup 15-02-2026 18:00\nПример с примечанием: /train_signup 15-02-2026 10:00 Уличная тренировка кроссфит гиря 16 кг")
            return
        
        username = update.effective_user.username
        if not username:
            await safe_reply_text(update, "❌ Ошибка: у вас не установлен username в Telegram. Пожалуйста, установите username в настройках Telegram.")
            return
        
        date = context.args[0]
        time = context.args[1]
        note = " ".join(context.args[2:]) if len(context.args) > 2 else ""
        
        try:
            result = await reg_create(username, date, time, note)
            if result:
                reg_id = result.get("reg_id")
                row_url = result.get("row_url", "")
                response_text = f"✅ Вы записаны на {date} в {time}\nID записи: {reg_id}"
                if note:
                    response_text += f"\nПримечание: {note}"
                response_text += f"\nСсылка: {row_url}"
                await safe_reply_text(update, response_text)
            else:
                await safe_reply_text(update, "❌ Ошибка при создании записи")
        except ValueError as e:
            await safe_reply_text(update, f"❌ {e}")
        except Exception as e:
            logger.exception(f"Error in train_signup_cmd: {e}")
            await safe_reply_text(update, f"❌ Неизвестная ошибка: {e}")


class TrainMoveHandler(Handler):
    """Handler for /train_move command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /train_move command."""
        if not update.message:
            return
        
        if not context.args or len(context.args) < 3:
            await safe_reply_text(update, "Использование: /train_move <reg_id> <дата DD-MM-YYYY> <время HH:MM>\nПример: /train_move 1 16-02-2026 19:00")
            return
        
        try:
            reg_id = int(context.args[0])
            new_date = context.args[1]
            new_time = context.args[2]
            
            result = await reg_reschedule(reg_id, new_date, new_time)
            if result:
                row_url = result.get("row_url", "")
                await safe_reply_text(
                    update,
                    f"✅ Запись {reg_id} перенесена на {new_date} {new_time}\nСсылка: {row_url}"
                )
            else:
                await safe_reply_text(update, "❌ Ошибка при переносе записи")
        except ValueError as e:
            await safe_reply_text(update, f"❌ {e}")
        except Exception as e:
            logger.exception(f"Error in train_move_cmd: {e}")
            await safe_reply_text(update, f"❌ Неизвестная ошибка: {e}")


class TrainCancelHandler(Handler):
    """Handler for /train_cancel command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /train_cancel command."""
        if not update.message:
            return
        
        if not context.args or len(context.args) < 1:
            await safe_reply_text(update, "Использование: /train_cancel <reg_id>\nПример: /train_cancel 1")
            return
        
        try:
            reg_id = int(context.args[0])
            result = await reg_cancel(reg_id)
            if result:
                await safe_reply_text(update, f"✅ Запись {reg_id} отменена и удалена из системы")
            else:
                await safe_reply_text(update, "❌ Ошибка при отмене записи")
        except ValueError as e:
            await safe_reply_text(update, f"❌ {e}")
        except Exception as e:
            logger.exception(f"Error in train_cancel_cmd: {e}")
            await safe_reply_text(update, f"❌ Неизвестная ошибка: {e}")


class SupportHandler(Handler):
    """Handler for /support command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /support command."""
        if not update.message:
            return
        
        if not context.args:
            await safe_reply_text(update, "Использование: /support <вопрос>\nПример: /support можно перенести запись?")
            return
        
        question = " ".join(context.args)
        username = update.effective_user.username
        if not username:
            await safe_reply_text(update, "❌ Ошибка: у вас не установлен username в Telegram. Пожалуйста, установите username в настройках Telegram.")
            return
        
        agent_context = self.get_agent_context(update, context)
        chat_id = agent_context.chat_id
        temperature = agent_context.temperature
        model = agent_context.model
        
        await update.message.chat.send_action("typing")
        
        try:
            user_data = None
            try:
                user_data = await user_get(username)
            except ValueError as e:
                logger.warning(f"Could not get user data: {e}")
            
            active_regs = []
            try:
                active_regs = await reg_find_by_user(username) or []
            except ValueError as e:
                logger.warning(f"Could not get user registrations: {e}")
            
            context_parts = []
            if user_data:
                context_parts.append(f"Данные пользователя: {user_data}")
            if active_regs:
                context_parts.append(f"Активные записи пользователя: {active_regs}")
            
            rag_context = ""
            if has_embeddings(EMBEDDING_MODEL):
                try:
                    chunks = search_relevant_chunks(
                        question,
                        model=EMBEDDING_MODEL,
                        top_k=RAG_TOP_K,
                        min_similarity=RAG_SIM_THRESHOLD,
                        apply_threshold=True
                    )
                    filtered_chunks = [chunk for chunk in chunks if chunk["similarity"] >= RAG_SIM_THRESHOLD]
                    if filtered_chunks:
                        rag_context = "\n\nРелевантная информация из документации:\n"
                        for chunk in filtered_chunks:
                            rag_context += f"- {chunk['text']}\n"
                except Exception as e:
                    logger.warning(f"Error searching chunks: {e}")
            
            user_context = "\n".join(context_parts) if context_parts else "Нет данных о пользователе"
            
            system_prompt = f"""Ты помощник поддержки. Отвечай на вопросы пользователя, используя информацию из документации и данные пользователя.
            
{user_context}
{rag_context}

Отвечай кратко и по делу."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            
            answer = call_llm(messages, temperature=temperature, model=model)
            if answer:
                await safe_reply_text(update, answer)
            else:
                await safe_reply_text(update, "Не удалось получить ответ.")
        except Exception as e:
            logger.exception(f"Error in support_cmd: {e}")
            await safe_reply_text(update, f"❌ Ошибка: {e}")


async def register_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /register."""
    handler = RegisterHandler()
    await handler.handle(update, context)


async def unregister_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /unregister."""
    handler = UnregisterHandler()
    await handler.handle(update, context)


async def train_signup_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /train_signup."""
    handler = TrainSignupHandler()
    await handler.handle(update, context)


async def train_move_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /train_move."""
    handler = TrainMoveHandler()
    await handler.handle(update, context)


async def train_cancel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /train_cancel."""
    handler = TrainCancelHandler()
    await handler.handle(update, context)


async def support_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /support."""
    handler = SupportHandler()
    await handler.handle(update, context)
