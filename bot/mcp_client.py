"""
Клиент для вызова MCP-инструментов.

Ожидается, что где‑то запущен MCP‑сервер с инструментами:
    - git_branch
    - get_pr_diff
    - get_pr_files
    - get_pr_info

Этот модуль подключается к такому серверу по Streamable HTTP и вызывает инструменты.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent

logger = logging.getLogger(__name__)

# Адрес MCP-сервера (можно переопределить через переменную окружения)
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")


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


async def get_pr_diff(owner: str, repo: str, pr_number: int, github_token: str) -> str | None:
    """
    Асинхронный вызов MCP-инструмента `get_pr_diff`.

    Args:
        owner: Владелец репозитория
        repo: Название репозитория
        pr_number: Номер PR
        github_token: GitHub token

    Returns:
        Diff строка или None в случае ошибки
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                result = await session.call_tool(
                    "get_pr_diff",
                    arguments={
                        "owner": owner,
                        "repo": repo,
                        "pr_number": pr_number,
                        "github_token": github_token,
                    },
                )

        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)

        if not parts:
            return None

        diff_text = " ".join(p.strip() for p in parts if p.strip())
        if "Ошибка" in diff_text or "error" in diff_text.lower():
            logger.error(f"Error getting PR diff: {diff_text}")
            return None

        return diff_text

    except Exception as e:
        logger.exception(f"Exception getting PR diff: {e}")
        return None


async def get_pr_files(owner: str, repo: str, pr_number: int, github_token: str) -> list[dict[str, Any]] | None:
    """
    Асинхронный вызов MCP-инструмента `get_pr_files`.

    Args:
        owner: Владелец репозитория
        repo: Название репозитория
        pr_number: Номер PR
        github_token: GitHub token

    Returns:
        Список словарей с информацией о файлах или None в случае ошибки
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                result = await session.call_tool(
                    "get_pr_files",
                    arguments={
                        "owner": owner,
                        "repo": repo,
                        "pr_number": pr_number,
                        "github_token": github_token,
                    },
                )

        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)

        if not parts:
            return None

        files_json = " ".join(p.strip() for p in parts if p.strip())
        if "Ошибка" in files_json or "error" in files_json.lower():
            logger.error(f"Error getting PR files: {files_json}")
            return None

        try:
            return json.loads(files_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse PR files JSON: {e}")
            return None

    except Exception as e:
        logger.exception(f"Exception getting PR files: {e}")
        return None


async def get_pr_info(owner: str, repo: str, pr_number: int, github_token: str) -> dict[str, Any] | None:
    """
    Асинхронный вызов MCP-инструмента `get_pr_info`.

    Args:
        owner: Владелец репозитория
        repo: Название репозитория
        pr_number: Номер PR
        github_token: GitHub token

    Returns:
        Словарь с информацией о PR или None в случае ошибки
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                result = await session.call_tool(
                    "get_pr_info",
                    arguments={
                        "owner": owner,
                        "repo": repo,
                        "pr_number": pr_number,
                        "github_token": github_token,
                    },
                )

        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)

        if not parts:
            return None

        info_json = " ".join(p.strip() for p in parts if p.strip())
        if "Ошибка" in info_json or "error" in info_json.lower():
            logger.error(f"Error getting PR info: {info_json}")
            return None

        try:
            return json.loads(info_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse PR info JSON: {e}")
            return None

    except Exception as e:
        logger.exception(f"Exception getting PR info: {e}")
        return None
