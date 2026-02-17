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
        # Проверяем, что это не сообщение об ошибке от MCP сервера
        # Ошибки обычно начинаются с "Ошибка:" или содержат "error:" в начале
        if diff_text.startswith("Ошибка") or diff_text.lower().startswith("error:"):
            logger.error(f"Error getting PR diff: {diff_text}")
            raise ValueError(diff_text)

        return diff_text

    except ValueError:
        raise
    except ConnectionError as e:
        logger.exception(f"Connection error to MCP server: {e}")
        raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}. Убедитесь, что сервер запущен.")
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg.lower() or "timeout" in error_msg.lower():
            raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}. Убедитесь, что сервер запущен.")
        logger.exception(f"Exception getting PR diff: {e}")
        raise ValueError(f"Ошибка при получении diff PR через MCP: {e}")


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
            logger.error("MCP tool get_pr_files returned empty response")
            return None

        files_json = " ".join(p.strip() for p in parts if p.strip())
        # Проверяем, что это не сообщение об ошибке от MCP сервера
        # Ошибки обычно начинаются с "Ошибка:" или содержат "error:" в начале
        if files_json.startswith("Ошибка") or files_json.lower().startswith("error:"):
            logger.error(f"Error getting PR files: {files_json}")
            raise ValueError(files_json)

        try:
            return json.loads(files_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse PR files JSON: {e}. Response: {files_json[:200]}")
            raise ValueError(f"Не удалось разобрать ответ от MCP сервера: {e}")

    except ValueError:
        raise
    except ConnectionError as e:
        logger.exception(f"Connection error to MCP server: {e}")
        raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}. Убедитесь, что сервер запущен.")
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg.lower() or "timeout" in error_msg.lower():
            raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}. Убедитесь, что сервер запущен.")
        logger.exception(f"Exception getting PR files: {e}")
        raise ValueError(f"Ошибка при получении файлов PR через MCP: {e}")


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
        # Проверяем, что это не сообщение об ошибке от MCP сервера
        # Ошибки обычно начинаются с "Ошибка:" или содержат "error:" в начале
        if info_json.startswith("Ошибка") or info_json.lower().startswith("error:"):
            logger.error(f"Error getting PR info: {info_json}")
            raise ValueError(info_json)

        try:
            return json.loads(info_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse PR info JSON: {e}. Response: {info_json[:200]}")
            raise ValueError(f"Не удалось разобрать ответ от MCP сервера: {e}")

    except ValueError:
        raise
    except ConnectionError as e:
        logger.exception(f"Connection error to MCP server: {e}")
        raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}. Убедитесь, что сервер запущен.")
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg.lower() or "timeout" in error_msg.lower():
            raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}. Убедитесь, что сервер запущен.")
        logger.exception(f"Exception getting PR info: {e}")
        raise ValueError(f"Ошибка при получении информации о PR через MCP: {e}")


# ==================== Google Sheets MCP Client Functions ====================

async def user_get(username: str) -> dict[str, Any] | None:
    """
    Получить данные пользователя по username.
    
    Args:
        username: Username пользователя из Telegram
        
    Returns:
        Словарь с данными пользователя или None в случае ошибки
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    "user_get",
                    arguments={"username": username},
                )
        
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)
        
        if not parts:
            return None
        
        response_text = " ".join(p.strip() for p in parts if p.strip())
        if response_text.startswith("Ошибка") or response_text.lower().startswith("error:"):
            logger.error(f"Error getting user: {response_text}")
            raise ValueError(response_text)
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse user data JSON: {e}")
            raise ValueError(f"Не удалось разобрать ответ от MCP сервера: {e}")
    
    except ValueError:
        raise
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg.lower():
            raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}")
        logger.exception(f"Exception getting user: {e}")
        raise ValueError(f"Ошибка при получении данных пользователя: {e}")


async def user_register(username: str, fio: str, phone: str) -> dict[str, Any] | None:
    """
    Зарегистрировать или обновить данные пользователя.
    
    Args:
        username: Username из Telegram
        fio: ФИО пользователя
        phone: Телефон пользователя
        
    Returns:
        Словарь со статусом операции или None в случае ошибки
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    "user_register",
                    arguments={
                        "username": username,
                        "fio": fio,
                        "phone": phone,
                    },
                )
        
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)
        
        if not parts:
            return None
        
        response_text = " ".join(p.strip() for p in parts if p.strip())
        if response_text.startswith("Ошибка") or response_text.lower().startswith("error:"):
            logger.error(f"Error registering user: {response_text}")
            raise ValueError(response_text)
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse registration response JSON: {e}")
            raise ValueError(f"Не удалось разобрать ответ от MCP сервера: {e}")
    
    except ValueError:
        raise
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg.lower():
            raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}")
        logger.exception(f"Exception registering user: {e}")
        raise ValueError(f"Ошибка при регистрации пользователя: {e}")


async def user_block(username: str) -> bool:
    """
    Заблокировать пользователя.
    
    Args:
        username: Username пользователя из Telegram
        
    Returns:
        True если успешно, иначе выбрасывает ValueError
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    "user_block",
                    arguments={"username": username},
                )
        
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)
        
        if not parts:
            return False
        
        response_text = " ".join(p.strip() for p in parts if p.strip())
        if response_text.startswith("Ошибка") or response_text.lower().startswith("error:"):
            logger.error(f"Error blocking user: {response_text}")
            raise ValueError(response_text)
        
        return True
    
    except ValueError:
        raise
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg.lower():
            raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}")
        logger.exception(f"Exception blocking user: {e}")
        raise ValueError(f"Ошибка при блокировке пользователя: {e}")


async def user_unblock(username: str) -> bool:
    """
    Разблокировать пользователя.
    
    Args:
        username: Username пользователя из Telegram
        
    Returns:
        True если успешно, иначе выбрасывает ValueError
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    "user_unblock",
                    arguments={"username": username},
                )
        
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)
        
        if not parts:
            return False
        
        response_text = " ".join(p.strip() for p in parts if p.strip())
        if response_text.startswith("Ошибка") or response_text.lower().startswith("error:"):
            logger.error(f"Error unblocking user: {response_text}")
            raise ValueError(response_text)
        
        return True
    
    except ValueError:
        raise
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg.lower():
            raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}")
        logger.exception(f"Exception unblocking user: {e}")
        raise ValueError(f"Ошибка при разблокировке пользователя: {e}")


async def user_delete(username: str) -> bool:
    """
    Удалить регистрацию пользователя из Google Sheets.
    
    Args:
        username: Username пользователя из Telegram
        
    Returns:
        True если успешно, иначе False
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    "user_delete",
                    arguments={"username": username},
                )
        
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)
        
        if not parts:
            return False
        
        response_text = " ".join(p.strip() for p in parts if p.strip())
        if response_text.startswith("Ошибка") or response_text.lower().startswith("error:"):
            logger.error(f"Error deleting user: {response_text}")
            raise ValueError(response_text)
        
        try:
            response_data = json.loads(response_text)
            return response_data.get("status") == "deleted"
        except json.JSONDecodeError:
            # Если не JSON, проверяем текстовый ответ
            return "удален" in response_text.lower() or "deleted" in response_text.lower()
    
    except ValueError:
        raise
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg.lower():
            raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}")
        logger.exception(f"Exception deleting user: {e}")
        raise ValueError(f"Ошибка при удалении пользователя: {e}")


async def reg_create(username: str, date: str, time: str, note: str = "") -> dict[str, Any] | None:
    """
    Создать запись на тренировку.
    
    Args:
        username: Username пользователя из Telegram
        date: Дата в формате DD-MM-YYYY
        time: Время в формате HH:MM
        note: Примечание к записи (опционально)
        
    Returns:
        Словарь с данными созданной записи или None в случае ошибки
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                arguments = {
                    "username": username,
                    "date": date,
                    "time": time,
                }
                if note:
                    arguments["note"] = note
                
                result = await session.call_tool(
                    "reg_create",
                    arguments=arguments,
                )
        
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)
        
        if not parts:
            return None
        
        response_text = " ".join(p.strip() for p in parts if p.strip())
        if response_text.startswith("Ошибка") or response_text.lower().startswith("error:"):
            logger.error(f"Error creating registration: {response_text}")
            raise ValueError(response_text)
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse registration creation response JSON: {e}")
            raise ValueError(f"Не удалось разобрать ответ от MCP сервера: {e}")
    
    except ValueError:
        raise
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg.lower():
            raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}")
        logger.exception(f"Exception creating registration: {e}")
        raise ValueError(f"Ошибка при создании записи: {e}")


async def reg_find_by_user(username: str) -> list[dict[str, Any]] | None:
    """
    Найти все активные записи пользователя.
    
    Args:
        username: Username пользователя из Telegram
        
    Returns:
        Список словарей с данными записей или None в случае ошибки
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    "reg_find_by_user",
                    arguments={"username": username},
                )
        
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)
        
        if not parts:
            return []
        
        response_text = " ".join(p.strip() for p in parts if p.strip())
        if response_text.startswith("Ошибка") or response_text.lower().startswith("error:"):
            logger.error(f"Error finding registrations: {response_text}")
            raise ValueError(response_text)
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse registrations JSON: {e}")
            raise ValueError(f"Не удалось разобрать ответ от MCP сервера: {e}")
    
    except ValueError:
        raise
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg.lower():
            raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}")
        logger.exception(f"Exception finding registrations: {e}")
        raise ValueError(f"Ошибка при поиске записей: {e}")


async def reg_reschedule(reg_id: int, new_date: str, new_time: str) -> dict[str, Any] | None:
    """
    Перенести запись на другое время.
    
    Args:
        reg_id: ID записи
        new_date: Новая дата в формате DD-MM-YYYY
        new_time: Новое время в формате HH:MM
        
    Returns:
        Словарь с обновленными данными записи или None в случае ошибки
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    "reg_reschedule",
                    arguments={
                        "reg_id": reg_id,
                        "new_date": new_date,
                        "new_time": new_time,
                    },
                )
        
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)
        
        if not parts:
            return None
        
        response_text = " ".join(p.strip() for p in parts if p.strip())
        if response_text.startswith("Ошибка") or response_text.lower().startswith("error:"):
            logger.error(f"Error rescheduling registration: {response_text}")
            raise ValueError(response_text)
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse reschedule response JSON: {e}")
            raise ValueError(f"Не удалось разобрать ответ от MCP сервера: {e}")
    
    except ValueError:
        raise
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg.lower():
            raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}")
        logger.exception(f"Exception rescheduling registration: {e}")
        raise ValueError(f"Ошибка при переносе записи: {e}")


async def reg_cancel(reg_id: int) -> bool:
    """
    Отменить запись.
    
    Args:
        reg_id: ID записи
        
    Returns:
        True если успешно, иначе выбрасывает ValueError
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    "reg_cancel",
                    arguments={"reg_id": reg_id},
                )
        
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)
        
        if not parts:
            return False
        
        response_text = " ".join(p.strip() for p in parts if p.strip())
        if response_text.startswith("Ошибка") or response_text.lower().startswith("error:"):
            logger.error(f"Error canceling registration: {response_text}")
            raise ValueError(response_text)
        
        return True
    
    except ValueError:
        raise
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg.lower():
            raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}")
        logger.exception(f"Exception canceling registration: {e}")
        raise ValueError(f"Ошибка при отмене записи: {e}")


# ==================== Task Management MCP Client Functions ====================

async def task_create(date: str, time: str, task: str, priority: str) -> dict[str, Any] | None:
    """
    Создать задачу в Google Sheets.
    
    Args:
        date: Дата в формате DD-MM-YYYY
        time: Время в формате HH:MM
        task: Описание задачи
        priority: Приоритет задачи (high/middle/low)
        
    Returns:
        Словарь с данными созданной задачи или None в случае ошибки
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    "task_create",
                    arguments={
                        "date": date,
                        "time": time,
                        "task": task,
                        "priority": priority,
                    },
                )
        
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)
        
        if not parts:
            return None
        
        response_text = " ".join(p.strip() for p in parts if p.strip())
        if response_text.startswith("Ошибка") or response_text.lower().startswith("error:"):
            logger.error(f"Error creating task: {response_text}")
            raise ValueError(response_text)
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse task creation response JSON: {e}")
            raise ValueError(f"Не удалось разобрать ответ от MCP сервера: {e}")
    
    except ValueError:
        raise
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg.lower():
            raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}")
        logger.exception(f"Exception creating task: {e}")
        raise ValueError(f"Ошибка при создании задачи: {e}")


async def task_list(priority: str | None = None, completed: bool | None = None, date_from: str | None = None, date_to: str | None = None) -> list[dict[str, Any]] | None:
    """
    Получить список задач с фильтрацией.
    
    Args:
        priority: Фильтр по приоритету (high/middle/low, опционально)
        completed: Фильтр по статусу выполнения (true/false, опционально)
        date_from: Начальная дата для фильтрации (DD-MM-YYYY, опционально)
        date_to: Конечная дата для фильтрации (DD-MM-YYYY, опционально)
        
    Returns:
        Список словарей с данными задач или None в случае ошибки
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                arguments = {}
                if priority is not None:
                    arguments["priority"] = priority
                if completed is not None:
                    arguments["completed"] = completed
                if date_from is not None:
                    arguments["date_from"] = date_from
                if date_to is not None:
                    arguments["date_to"] = date_to
                
                result = await session.call_tool(
                    "task_list",
                    arguments=arguments,
                )
        
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)
        
        if not parts:
            return []
        
        response_text = " ".join(p.strip() for p in parts if p.strip())
        if response_text.startswith("Ошибка") or response_text.lower().startswith("error:"):
            logger.error(f"Error getting task list: {response_text}")
            raise ValueError(response_text)
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse task list JSON: {e}")
            raise ValueError(f"Не удалось разобрать ответ от MCP сервера: {e}")
    
    except ValueError:
        raise
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg.lower():
            raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}")
        logger.exception(f"Exception getting task list: {e}")
        raise ValueError(f"Ошибка при получении списка задач: {e}")


async def task_delete(row_number: int) -> dict | None:
    """
    Удалить задачу из Google Sheets.
    
    Args:
        row_number: Номер строки в Google Sheets (начиная с 2)
        
    Returns:
        dict с полями status ("deleted" или "cleared") и message (опционально), или None при ошибке
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    "task_delete",
                    arguments={"row_number": row_number},
                )
        
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)
        
        if not parts:
            return None
        
        response_text = " ".join(p.strip() for p in parts if p.strip())
        if response_text.startswith("Ошибка") or response_text.lower().startswith("error:"):
            logger.error(f"Error deleting task: {response_text}")
            raise ValueError(response_text)
        
        try:
            response_data = json.loads(response_text)
            # Возвращаем dict с информацией о статусе
            status = response_data.get("status")
            if status in ["deleted", "cleared"]:
                return {
                    "status": status,
                    "row_number": response_data.get("row_number", row_number),
                    "message": response_data.get("message", "")
                }
            return None
        except json.JSONDecodeError:
            # Если не JSON, проверяем текстовый ответ
            if "удален" in response_text.lower() or "deleted" in response_text.lower():
                return {"status": "deleted", "row_number": row_number, "message": ""}
            if "очищен" in response_text.lower() or "cleared" in response_text.lower():
                return {"status": "cleared", "row_number": row_number, "message": ""}
            return None
    
    except ValueError:
        raise
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg.lower():
            raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}")
        logger.exception(f"Exception deleting task: {e}")
        raise ValueError(f"Ошибка при удалении задачи: {e}")


# ==================== DEPLOY FUNCTIONS ====================

async def deploy_check_docker(host: str, port: int, username: str, password: str) -> dict | None:
    """
    Проверяет наличие Docker на сервере и устанавливает его, если отсутствует.
    
    Args:
        host: SSH host сервера
        port: SSH port
        username: SSH username
        password: SSH password
    
    Returns:
        dict с результатом проверки/установки Docker или None при ошибке
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    "deploy_check_docker",
                    arguments={
                        "host": host,
                        "port": port,
                        "username": username,
                        "password": password,
                    },
                )
        
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)
        
        if not parts:
            return None
        
        response_text = " ".join(p.strip() for p in parts if p.strip())
        if response_text.startswith("Ошибка") or response_text.lower().startswith("error:"):
            logger.error(f"Error checking Docker: {response_text}")
            raise ValueError(response_text)
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return None
    
    except ValueError:
        raise
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg.lower():
            raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}")
        logger.exception(f"Exception checking Docker: {e}")
        raise ValueError(f"Ошибка при проверке Docker: {e}")


async def deploy_upload_image(host: str, port: int, username: str, password: str, image_tar_path: str, remote_path: str) -> dict | None:
    """
    Загружает Docker image (.tar файл) на сервер через SCP.
    
    Args:
        host: SSH host сервера
        port: SSH port
        username: SSH username
        password: SSH password
        image_tar_path: Локальный путь к Docker image .tar файлу
        remote_path: Путь на сервере для сохранения файла
    
    Returns:
        dict с результатом загрузки или None при ошибке
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    "deploy_upload_image",
                    arguments={
                        "host": host,
                        "port": port,
                        "username": username,
                        "password": password,
                        "image_tar_path": image_tar_path,
                        "remote_path": remote_path,
                    },
                )
        
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)
        
        if not parts:
            return None
        
        response_text = " ".join(p.strip() for p in parts if p.strip())
        if response_text.startswith("Ошибка") or response_text.lower().startswith("error:"):
            logger.error(f"Error uploading image: {response_text}")
            raise ValueError(response_text)
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return None
    
    except ValueError:
        raise
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg.lower():
            raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}")
        logger.exception(f"Exception uploading image: {e}")
        raise ValueError(f"Ошибка при загрузке образа: {e}")


async def deploy_load_image(host: str, port: int, username: str, password: str, image_tar_path: str) -> dict | None:
    """
    Загружает Docker image в Docker на сервере.
    
    Args:
        host: SSH host сервера
        port: SSH port
        username: SSH username
        password: SSH password
        image_tar_path: Путь к .tar файлу на сервере
    
    Returns:
        dict с результатом загрузки образа или None при ошибке
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    "deploy_load_image",
                    arguments={
                        "host": host,
                        "port": port,
                        "username": username,
                        "password": password,
                        "image_tar_path": image_tar_path,
                    },
                )
        
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)
        
        if not parts:
            return None
        
        response_text = " ".join(p.strip() for p in parts if p.strip())
        if response_text.startswith("Ошибка") or response_text.lower().startswith("error:"):
            logger.error(f"Error loading image: {response_text}")
            raise ValueError(response_text)
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return None
    
    except ValueError:
        raise
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg.lower():
            raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}")
        logger.exception(f"Exception loading image: {e}")
        raise ValueError(f"Ошибка при загрузке образа в Docker: {e}")


async def deploy_create_compose(host: str, port: int, username: str, password: str, compose_content: str, remote_path: str = "/opt/nikita_ai/docker-compose.yml") -> dict | None:
    """
    Создает или обновляет docker-compose.yml на сервере.
    
    Args:
        host: SSH host сервера
        port: SSH port
        username: SSH username
        password: SSH password
        compose_content: YAML содержимое docker-compose.yml
        remote_path: Путь на сервере для сохранения файла
    
    Returns:
        dict с результатом создания/обновления файла или None при ошибке
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    "deploy_create_compose",
                    arguments={
                        "host": host,
                        "port": port,
                        "username": username,
                        "password": password,
                        "compose_content": compose_content,
                        "remote_path": remote_path,
                    },
                )
        
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)
        
        if not parts:
            return None
        
        response_text = " ".join(p.strip() for p in parts if p.strip())
        if response_text.startswith("Ошибка") or response_text.lower().startswith("error:"):
            logger.error(f"Error creating compose: {response_text}")
            raise ValueError(response_text)
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return None
    
    except ValueError:
        raise
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg.lower():
            raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}")
        logger.exception(f"Exception creating compose: {e}")
        raise ValueError(f"Ошибка при создании docker-compose.yml: {e}")


async def deploy_create_env(host: str, port: int, username: str, password: str, env_content: str, remote_path: str = "/opt/nikita_ai/.env") -> dict | None:
    """
    Создает или обновляет .env файл на сервере.
    
    Args:
        host: SSH host сервера
        port: SSH port
        username: SSH username
        password: SSH password
        env_content: Содержимое .env файла
        remote_path: Путь на сервере для сохранения файла
    
    Returns:
        dict с результатом создания/обновления файла или None при ошибке
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    "deploy_create_env",
                    arguments={
                        "host": host,
                        "port": port,
                        "username": username,
                        "password": password,
                        "env_content": env_content,
                        "remote_path": remote_path,
                    },
                )
        
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)
        
        if not parts:
            return None
        
        response_text = " ".join(p.strip() for p in parts if p.strip())
        if response_text.startswith("Ошибка") or response_text.lower().startswith("error:"):
            logger.error(f"Error creating env: {response_text}")
            raise ValueError(response_text)
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return None
    
    except ValueError:
        raise
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg.lower():
            raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}")
        logger.exception(f"Exception creating env: {e}")
        raise ValueError(f"Ошибка при создании .env файла: {e}")


async def deploy_start_bot(host: str, port: int, username: str, password: str, compose_path: str = "/opt/nikita_ai/docker-compose.yml") -> dict | None:
    """
    Запускает бота через docker-compose на сервере.
    
    Args:
        host: SSH host сервера
        port: SSH port
        username: SSH username
        password: SSH password
        compose_path: Путь к docker-compose.yml на сервере
    
    Returns:
        dict с результатом запуска или None при ошибке
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    "deploy_start_bot",
                    arguments={
                        "host": host,
                        "port": port,
                        "username": username,
                        "password": password,
                        "compose_path": compose_path,
                    },
                )
        
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)
        
        if not parts:
            return None
        
        response_text = " ".join(p.strip() for p in parts if p.strip())
        if response_text.startswith("Ошибка") or response_text.lower().startswith("error:"):
            logger.error(f"Error starting bot: {response_text}")
            raise ValueError(response_text)
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return None
    
    except ValueError:
        raise
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg.lower():
            raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}")
        logger.exception(f"Exception starting bot: {e}")
        raise ValueError(f"Ошибка при запуске бота: {e}")


async def deploy_check_container(host: str, port: int, username: str, password: str, container_name: str = "nikita_ai_bot") -> dict | None:
    """
    Проверяет статус контейнера и возвращает его логи.
    
    Args:
        host: SSH host сервера
        port: SSH port
        username: SSH username
        password: SSH password
        container_name: Имя контейнера
    
    Returns:
        dict с результатом проверки и логами или None при ошибке
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    "deploy_check_container",
                    arguments={
                        "host": host,
                        "port": port,
                        "username": username,
                        "password": password,
                        "container_name": container_name,
                    },
                )
        
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)
        
        if not parts:
            return None
        
        response_text = " ".join(p.strip() for p in parts if p.strip())
        if response_text.startswith("Ошибка") or response_text.lower().startswith("error:"):
            logger.error(f"Error checking container: {response_text}")
            raise ValueError(response_text)
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return None
    
    except ValueError:
        raise
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg.lower():
            raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}")
        logger.exception(f"Exception checking container: {e}")
        raise ValueError(f"Ошибка при проверке контейнера: {e}")


async def deploy_read_env(host: str, port: int, username: str, password: str, env_path: str = "/opt/nikita_ai/.env") -> dict | None:
    """
    Читает содержимое .env файла на сервере (токены скрыты).
    
    Args:
        host: SSH host сервера
        port: SSH port
        username: SSH username
        password: SSH password
        env_path: Путь к .env файлу на сервере
    
    Returns:
        dict с содержимым .env файла или None при ошибке
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    "deploy_read_env",
                    arguments={
                        "host": host,
                        "port": port,
                        "username": username,
                        "password": password,
                        "env_path": env_path,
                    },
                )
        
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)
        
        if not parts:
            return None
        
        response_text = " ".join(p.strip() for p in parts if p.strip())
        if response_text.startswith("Ошибка") or response_text.lower().startswith("error:"):
            logger.error(f"Error reading env: {response_text}")
            raise ValueError(response_text)
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return None
    
    except ValueError:
        raise
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg.lower():
            raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}")
        logger.exception(f"Exception reading env: {e}")
        raise ValueError(f"Ошибка при чтении .env файла: {e}")


async def deploy_stop_bot(host: str, port: int, username: str, password: str, compose_path: str = "/opt/nikita_ai/docker-compose.yml", remove_volumes: bool = False, remove_images: bool = False) -> dict | None:
    """
    Останавливает и удаляет бота с сервера.
    
    Args:
        host: SSH host сервера
        port: SSH port
        username: SSH username
        password: SSH password
        compose_path: Путь к docker-compose.yml на сервере
        remove_volumes: Удалять ли volumes (данные)
        remove_images: Удалять ли Docker образы
    
    Returns:
        dict с результатом остановки или None при ошибке
    """
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    "deploy_stop_bot",
                    arguments={
                        "host": host,
                        "port": port,
                        "username": username,
                        "password": password,
                        "compose_path": compose_path,
                        "remove_volumes": remove_volumes,
                        "remove_images": remove_images,
                    },
                )
        
        parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                parts.append(item.text)
        
        if not parts:
            return None
        
        response_text = " ".join(p.strip() for p in parts if p.strip())
        if response_text.startswith("Ошибка") or response_text.lower().startswith("error:"):
            logger.error(f"Error stopping bot: {response_text}")
            raise ValueError(response_text)
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return None
    
    except ValueError:
        raise
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg.lower():
            raise ValueError(f"Не удалось подключиться к MCP серверу по адресу {MCP_SERVER_URL}")
        logger.exception(f"Exception stopping bot: {e}")
        raise ValueError(f"Ошибка при остановке бота: {e}")
