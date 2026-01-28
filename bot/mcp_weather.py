"""
Клиент для вызова MCP-инструмента погоды.

Ожидается, что где‑то запущен MCP‑сервер со строковым инструментом:
    @mcp.tool()
    def get_weather(city: str, unit: str = "celsius") -> str: ...

Этот модуль подключается к такому серверу по Streamable HTTP и вызывает tool `get_weather`.
"""

from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent

MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"  # Адрес MCP-сервера погоды


async def get_weather_via_mcp(city: str, unit: str = "celsius") -> str:
    """
    Асинхронный вызов MCP-инструмента `get_weather`.

    Возвращает человекочитаемую строку с погодой или текст ошибки.
    """
    city = (city or "").strip()
    if not city:
        return "Город не указан."

    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                # Инициализируем MCP-сессию
                await session.initialize()

                # Вызываем инструмент погоды
                result = await session.call_tool(
                    "get_weather",
                    arguments={"city": city, "unit": unit},
                )

        # Собираем текстовый контент из результата
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)

        if not parts:
            return f"Не удалось разобрать ответ MCP-инструмента для города {city}."

        return " ".join(p.strip() for p in parts if p.strip())

    except Exception as e:
        return f"Ошибка при вызове MCP-инструмента погоды: {e}"