"""
Клиент для вызова MCP-инструментов управления Docker контейнерами.

Ожидается, что где‑то запущен MCP‑сервер со следующими инструментами:
    @mcp.tool()
    def site_up() -> str: ...
    
    @mcp.tool()
    def site_screenshot() -> str: ...
    
    @mcp.tool()
    def site_down() -> str: ...

Этот модуль подключается к такому серверу по Streamable HTTP и вызывает эти tools.
"""

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent

MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"  # Адрес MCP-сервера


async def site_up_via_mcp() -> str:
    """
    Асинхронный вызов MCP-инструмента `site_up`.
    
    Возвращает строку с результатом операции и URL сайта.
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                # Инициализируем MCP-сессию
                await session.initialize()
                
                # Вызываем инструмент
                result = await session.call_tool(
                    "site_up",
                    arguments={},
                )
        
        # Собираем текстовый контент из результата
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)
        
        if not parts:
            return "Не удалось разобрать ответ MCP-инструмента site_up."
        
        return " ".join(p.strip() for p in parts if p.strip())
    
    except Exception as e:
        return f"Ошибка при вызове MCP-инструмента site_up: {e}"


async def site_screenshot_via_mcp() -> str:
    """
    Асинхронный вызов MCP-инструмента `site_screenshot`.
    
    Возвращает путь к сохранённому PNG файлу.
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                # Инициализируем MCP-сессию
                await session.initialize()
                
                # Вызываем инструмент
                result = await session.call_tool(
                    "site_screenshot",
                    arguments={},
                )
        
        # Собираем текстовый контент из результата
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)
        
        if not parts:
            return "Не удалось разобрать ответ MCP-инструмента site_screenshot."
        
        return " ".join(p.strip() for p in parts if p.strip())
    
    except Exception as e:
        return f"Ошибка при вызове MCP-инструмента site_screenshot: {e}"


async def site_down_via_mcp() -> str:
    """
    Асинхронный вызов MCP-инструмента `site_down`.
    
    Возвращает строку с результатом операции.
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                # Инициализируем MCP-сессию
                await session.initialize()
                
                # Вызываем инструмент
                result = await session.call_tool(
                    "site_down",
                    arguments={},
                )
        
        # Собираем текстовый контент из результата
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)
        
        if not parts:
            return "Не удалось разобрать ответ MCP-инструмента site_down."
        
        return " ".join(p.strip() for p in parts if p.strip())
    
    except Exception as e:
        return f"Ошибка при вызове MCP-инструмента site_down: {e}"
