"""
Клиент для вызова MCP-инструмента новостей.

Ожидается, что где‑то запущен MCP‑сервер со строковым инструментом:
    @mcp.tool()
    def get_news(topic: str, count: int = 5) -> str: ...

Этот модуль подключается к такому серверу по Streamable HTTP и вызывает tool `get_news`.
"""

from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent

MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"  # Адрес MCP-сервера


async def get_news_via_mcp(topic: str, count: int = 5) -> str:
    """
    Асинхронный вызов MCP-инструмента `get_news`.

    Возвращает человекочитаемую строку с новостями или текст ошибки.
    """
    topic = (topic or "").strip()
    if not topic:
        return "Тема новостей не указана."

    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                # Инициализируем MCP-сессию
                await session.initialize()

                # Вызываем инструмент новостей
                result = await session.call_tool(
                    "get_news",
                    arguments={"topic": topic, "count": count},
                )

        # Собираем текстовый контент из результата
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)

        if not parts:
            return f"Не удалось разобрать ответ MCP-инструмента для темы '{topic}'."

        return " ".join(p.strip() for p in parts if p.strip())

    except Exception as e:
        return f"Ошибка при вызове MCP-инструмента новостей: {e}"
