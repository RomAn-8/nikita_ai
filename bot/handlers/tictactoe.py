"""Tic-Tac-Toe game command handler."""

import logging

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from ..core.errors import safe_reply_text
from ..services.tictactoe import TicTacToe, start_game, get_game, end_game

logger = logging.getLogger(__name__)


def _build_keyboard(game: TicTacToe) -> InlineKeyboardMarkup:
    """Build inline keyboard for the current game state."""
    keyboard = []
    for row in range(3):
        row_buttons = []
        for col in range(3):
            pos = row * 3 + col
            cell_value = game.board[pos]
            if cell_value == game.EMPTY:
                button_text = str(pos + 1)
            elif cell_value == game.HUMAN:
                button_text = "❌"
            else:
                button_text = "⭕"
            row_buttons.append(
                InlineKeyboardButton(button_text, callback_data=f"tictactoe_{pos}")
            )
        keyboard.append(row_buttons)

    if game.game_over:
        keyboard.append([
            InlineKeyboardButton("🔄 Новая игра", callback_data="tictactoe_reset"),
            InlineKeyboardButton("❌ Выход", callback_data="tictactoe_exit"),
        ])
    else:
        keyboard.append([
            InlineKeyboardButton("❌ Выход", callback_data="tictactoe_exit"),
        ])

    return InlineKeyboardMarkup(keyboard)


async def tictactoe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start a new Tic-Tac-Toe game."""
    if not update.message:
        return

    chat_id = update.effective_chat.id

    try:
        game = start_game(chat_id)
        reply_markup = _build_keyboard(game)
        await update.message.reply_text(
            f"🎮 Крестики-нолики\n\n{game.get_board_display()}\n\n{game.message}",
            reply_markup=reply_markup,
        )
    except Exception as e:
        logger.exception("Error in tictactoe_cmd: %s", e)
        await safe_reply_text(update, "Ошибка при запуске игры. Попробуйте позже.")


async def tictactoe_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle Tic-Tac-Toe button clicks."""
    if not update.callback_query:
        return

    query = update.callback_query
    if not query.message:
        await query.answer()
        return

    data = query.data
    chat_id = query.message.chat_id

    try:
        if data == "tictactoe_reset":
            await query.answer()
            start_game(chat_id)
        elif data == "tictactoe_exit":
            await query.answer()
            end_game(chat_id)
            await query.edit_message_text("Игра завершена. 👋")
            return
        elif data.startswith("tictactoe_"):
            try:
                position = int(data.split("_")[1])
            except (ValueError, IndexError):
                await query.answer("Неверный ход", show_alert=True)
                return

            game = get_game(chat_id)
            if not game:
                await query.answer(
                    "Игра не найдена. Начните новую: /tictactoe", show_alert=True
                )
                return

            if not game.make_move(position):
                await query.answer("Недействительный ход", show_alert=True)
                return

            await query.answer()
        else:
            return

        game = get_game(chat_id)
        if not game:
            return

        reply_markup = _build_keyboard(game)
        await query.edit_message_text(
            f"🎮 Крестики-нолики\n\n{game.get_board_display()}\n\n{game.message}",
            reply_markup=reply_markup,
        )
    except Exception as e:
        logger.exception("Error in tictactoe_callback: %s", e)
        try:
            await query.answer("Ошибка при обработке хода", show_alert=True)
        except Exception:
            pass
