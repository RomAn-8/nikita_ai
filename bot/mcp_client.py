"""
Клиент для вызова MCP-инструмента git_branch.

Ожидается, что где‑то запущен MCP‑сервер со строковым инструментом:
    @mcp.tool()
    async def git_branch(repo_path: str | None = None) -> str: ...

Этот модуль подключается к такому серверу по Streamable HTTP и вызывает tool `git_branch`.
"""

import logging
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent

logger = logging.getLogger(__name__)

MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"  # Адрес MCP-сервера


async def get_git_branch(repo_path: str | None = None) -> str | None:
    """
    Асинхронный вызов MCP-инструмента `git_branch`.

    Args:
        repo_path: Путь к репозиторию (опционально). Если не указан, используется
                   путь к nikita_ai репозиторию (где находится этот файл).

    Returns:
        Название текущей ветки git или None в случае ошибки
    """
    try:
        # Если путь не указан, используем путь к nikita_ai репозиторию
        if repo_path is None:
            # Определяем корень nikita_ai (где находится этот файл: bot/mcp_client.py)
            nikita_ai_root = Path(__file__).resolve().parent.parent
            repo_path = str(nikita_ai_root)
            logger.debug(f"Автоматически определен путь к nikita_ai: {repo_path}")
        
        logger.debug(f"Вызываем git_branch для репозитория: {repo_path}")
        
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                # Инициализируем MCP-сессию
                await session.initialize()

                # Вызываем инструмент git_branch с путем к репозиторию
                result = await session.call_tool(
                    "git_branch",
                    arguments={"repo_path": repo_path},
                )

        # Собираем текстовый контент из результата
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)

        if not parts:
            return None

        branch_name = " ".join(p.strip() for p in parts if p.strip())
        # Если это сообщение об ошибке, возвращаем None
        if "Ошибка" in branch_name or "error" in branch_name.lower():
            return None
        
        return branch_name

    except Exception:
        # В случае любой ошибки возвращаем None
        return None
