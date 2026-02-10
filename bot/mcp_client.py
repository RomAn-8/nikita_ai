"""
Клиент для вызова MCP-инструмента git_branch.

Ожидается, что где‑то запущен MCP‑сервер со строковым инструментом:
    @mcp.tool()
    async def git_branch() -> str: ...

Этот модуль подключается к такому серверу по Streamable HTTP и вызывает tool `git_branch`.
"""

from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent

MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"  # Адрес MCP-сервера


async def get_git_branch() -> str | None:
    """
    Асинхронный вызов MCP-инструмента `git_branch`.

    Returns:
        Название текущей ветки git или None в случае ошибки
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                # Инициализируем MCP-сессию
                await session.initialize()

                # Вызываем инструмент git_branch
                result = await session.call_tool(
                    "git_branch",
                    arguments={},
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
