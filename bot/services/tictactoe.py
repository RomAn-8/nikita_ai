"""Tic-Tac-Toe game service with AI opponent."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Game state storage: chat_id -> TicTacToe instance
GAMES: dict[int, "TicTacToe"] = {}


class TicTacToe:
    """Tic-Tac-Toe game logic with AI opponent."""

    EMPTY = 0
    HUMAN = 1
    AI = 2

    def __init__(self):
        """Initialize a new game."""
        self.board = [self.EMPTY] * 9  # 0-8: positions
        self.human_symbol = "❌"
        self.ai_symbol = "⭕"
        self.empty_symbol = "⬜"
        self.game_over = False
        self.winner: Optional[int] = None
        self.message = "Ваш ход"

    def make_move(self, position: int) -> bool:
        """Make human move. Returns True if valid."""
        if position < 0 or position > 8:
            return False
        if self.board[position] != self.EMPTY:
            return False
        if self.game_over:
            return False

        self.board[position] = self.HUMAN

        # Check if human won
        winner = self._check_winner()
        if winner == self.HUMAN:
            self.game_over = True
            self.winner = self.HUMAN
            self.message = "Вы выиграли! 🎉"
            return True

        # Check if board is full (draw)
        if self._is_full():
            self.game_over = True
            self.winner = None
            self.message = "Ничья! 🤝"
            return True

        # AI move
        ai_pos = self._find_best_move()
        if self.board[ai_pos] != self.EMPTY:
            raise RuntimeError(f"AI move validation failed: position {ai_pos} is not empty")
        self.board[ai_pos] = self.AI

        # Check if AI won
        winner = self._check_winner()
        if winner == self.AI:
            self.game_over = True
            self.winner = self.AI
            self.message = "ИИ выиграл! 🤖"
            return True

        # Check if board is full (draw)
        if self._is_full():
            self.game_over = True
            self.winner = None
            self.message = "Ничья! 🤝"
            return True

        self.message = "Ваш ход"
        return True

    def get_board_display(self) -> str:
        """Get formatted board display."""
        symbols = []
        for cell in self.board:
            if cell == self.EMPTY:
                symbols.append(self.empty_symbol)
            elif cell == self.HUMAN:
                symbols.append(self.human_symbol)
            else:
                symbols.append(self.ai_symbol)

        board_str = (
            f"{symbols[0]} {symbols[1]} {symbols[2]}\n"
            f"{symbols[3]} {symbols[4]} {symbols[5]}\n"
            f"{symbols[6]} {symbols[7]} {symbols[8]}"
        )
        return board_str

    def _check_winner(self) -> Optional[int]:
        """Check if there is a winner. Returns HUMAN, AI, or None."""
        winning_combos = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]              # diagonals
        ]

        for combo in winning_combos:
            values = [self.board[i] for i in combo]
            if values[0] != self.EMPTY and values[0] == values[1] == values[2]:
                return values[0]

        return None

    def _is_full(self) -> bool:
        """Check if board is full."""
        return self.EMPTY not in self.board

    def _find_best_move(self) -> int:
        """Find best move for AI using minimax."""
        best_score = float('-inf')
        best_move = None

        for i in range(9):
            if self.board[i] == self.EMPTY:
                self.board[i] = self.AI
                score = self._minimax(0, False)
                self.board[i] = self.EMPTY

                if score > best_score:
                    best_score = score
                    best_move = i

        if best_move is None:
            raise RuntimeError("No empty cells available for AI move")
        return best_move

    def _minimax(self, depth: int, is_maximizing: bool) -> int:
        """Minimax algorithm for AI decision."""
        winner = self._check_winner()

        if winner == self.AI:
            return 10 - depth
        elif winner == self.HUMAN:
            return depth - 10
        elif self._is_full():
            return 0

        if is_maximizing:
            best_score = float('-inf')
            for i in range(9):
                if self.board[i] == self.EMPTY:
                    self.board[i] = self.AI
                    score = self._minimax(depth + 1, False)
                    self.board[i] = self.EMPTY
                    best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for i in range(9):
                if self.board[i] == self.EMPTY:
                    self.board[i] = self.HUMAN
                    score = self._minimax(depth + 1, True)
                    self.board[i] = self.EMPTY
                    best_score = min(score, best_score)
            return best_score


def start_game(chat_id: int) -> TicTacToe:
    """Start a new game for chat."""
    game = TicTacToe()
    GAMES[chat_id] = game
    return game


def get_game(chat_id: int) -> Optional[TicTacToe]:
    """Get active game for chat."""
    return GAMES.get(chat_id)


def end_game(chat_id: int) -> None:
    """End game for chat."""
    GAMES.pop(chat_id, None)
