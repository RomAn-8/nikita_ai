"""Unified error handling for God Agent."""

import logging
from typing import Any
from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


class AgentError(Exception):
    """Base exception for agent errors."""
    pass


class ToolExecutionError(AgentError):
    """Error during tool execution."""
    pass


class ConfigurationError(AgentError):
    """Configuration error."""
    pass


async def safe_reply_text(update: Update, text: str, parse_mode: str | None = None) -> None:
    """
    Safely reply to a message, handling errors gracefully.
    
    Args:
        update: Telegram update object
        text: Text to send
        parse_mode: Parse mode for message
    """
    if not update.message:
        return
    
    from ..utils.text import split_telegram_text
    from telegram.error import TimedOut, BadRequest
    
    chunks = split_telegram_text(text)
    for ch in chunks:
        try:
            await update.message.reply_text(ch, parse_mode=parse_mode)
        except TimedOut:
            return
        except BadRequest as e:
            msg = str(e).lower()
            if "message is too long" in msg and len(ch) > 500:
                for sub in split_telegram_text(ch, limit=2000):
                    try:
                        await update.message.reply_text(sub, parse_mode=parse_mode)
                    except Exception:
                        return
            else:
                logger.error(f"BadRequest sending message: {e}")
                return
        except Exception as e:
            logger.error(f"Error sending message: {e}", exc_info=True)
            return


async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Unified error handler.
    
    Args:
        update: Telegram update object (may be None)
        context: Telegram context object
    """
    error = context.error
    if error is None:
        return
    
    error_type = type(error).__name__
    error_msg = str(error)
    
    # Ignore connection errors - they are usually temporary
    if "ConnectError" in error_type or "getaddrinfo failed" in error_msg or "11001" in error_msg:
        logger.warning(f"Connection error (likely temporary): {error_type}: {error_msg}")
        # Don't show to user - this is a temporary network issue
        return
    
    logger.error(f"Unhandled error: {error}", exc_info=True)
    
    if isinstance(update, Update) and update.message:
        try:
            await update.message.reply_text(
                f"❌ Произошла ошибка при обработке запроса. Попробуйте ещё раз."
            )
        except Exception as e:
            logger.error(f"Error sending error message: {e}", exc_info=True)
