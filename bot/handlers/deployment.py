"""Deployment handlers."""

import os
import asyncio
from pathlib import Path
from telegram import Update
from telegram.ext import ContextTypes

from ..core.errors import safe_reply_text
from ..handlers.base import Handler
from ..mcp_client import (
    deploy_check_docker, deploy_upload_image, deploy_load_image,
    deploy_create_compose, deploy_create_env, deploy_start_bot,
    deploy_check_container, deploy_stop_bot
)
from ..config import (
    OPENROUTER_API_KEY, OPENROUTER_MODEL, EMBEDDING_MODEL,
    RAG_SIM_THRESHOLD, RAG_TOP_K, OLLAMA_MODEL, OLLAMA_TIMEOUT,
    OLLAMA_TEMPERATURE, OLLAMA_NUM_CTX, OLLAMA_NUM_PREDICT, OLLAMA_SYSTEM_PROMPT
)
import logging

logger = logging.getLogger(__name__)


class DeployBotHandler(Handler):
    """Handler for /deploy_bot command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /deploy_bot command."""
        if not update.message:
            return
        
        try:
            deploy_ssh_host = os.getenv("DEPLOY_SSH_HOST", "").strip()
            deploy_ssh_port = int(os.getenv("DEPLOY_SSH_PORT", "22"))
            deploy_ssh_username = os.getenv("DEPLOY_SSH_USERNAME", "").strip()
            deploy_ssh_password = os.getenv("DEPLOY_SSH_PASSWORD", "").strip()
            deploy_image_tar_path = os.getenv("DEPLOY_IMAGE_TAR_PATH", "").strip()
            deploy_remote_path = os.getenv("DEPLOY_REMOTE_PATH", "/opt/nikita_ai").strip()
            deploy_bot_token = os.getenv("DEPLOY_BOT_TOKEN", "").strip()
            
            deploy_openrouter_api_key = OPENROUTER_API_KEY
            deploy_openrouter_model = OPENROUTER_MODEL
            deploy_embedding_model = EMBEDDING_MODEL
            deploy_rag_sim_threshold = str(RAG_SIM_THRESHOLD)
            deploy_rag_top_k = str(RAG_TOP_K)
            deploy_ollama_base_url = "http://127.0.0.1:11434"
            deploy_ollama_model = OLLAMA_MODEL
            deploy_ollama_timeout = str(OLLAMA_TIMEOUT)
            deploy_ollama_temperature = str(OLLAMA_TEMPERATURE)
            deploy_ollama_num_ctx = str(OLLAMA_NUM_CTX)
            deploy_ollama_num_predict = str(OLLAMA_NUM_PREDICT)
            deploy_ollama_system_prompt = OLLAMA_SYSTEM_PROMPT
            
            missing_vars = []
            if not deploy_ssh_host:
                missing_vars.append("DEPLOY_SSH_HOST")
            if not deploy_ssh_username:
                missing_vars.append("DEPLOY_SSH_USERNAME")
            if not deploy_ssh_password:
                missing_vars.append("DEPLOY_SSH_PASSWORD")
            if not deploy_image_tar_path:
                missing_vars.append("DEPLOY_IMAGE_TAR_PATH")
            if not deploy_bot_token:
                missing_vars.append("DEPLOY_BOT_TOKEN")
            
            if missing_vars:
                await safe_reply_text(
                    update,
                    f"❌ Отсутствуют обязательные переменные окружения:\n" + "\n".join(f"• {var}" for var in missing_vars)
                )
                return
            
            image_path = Path(deploy_image_tar_path)
            if not image_path.exists():
                await safe_reply_text(update, f"❌ Файл образа не найден: {deploy_image_tar_path}")
                return
            
            image_name = "nikita_ai"
            image_tag = "latest"
            
            await safe_reply_text(update, "🚀 Начинаю деплой бота на сервер...")
            
            await safe_reply_text(update, "📦 Проверяю наличие Docker на сервере...")
            docker_result = await deploy_check_docker(deploy_ssh_host, deploy_ssh_port, deploy_ssh_username, deploy_ssh_password)
            if not docker_result or docker_result.get("status") != "installed":
                error_msg = docker_result.get("message", "Неизвестная ошибка") if docker_result else "Ошибка при проверке Docker"
                await safe_reply_text(update, f"❌ Ошибка при проверке Docker: {error_msg}")
                return
            await safe_reply_text(update, f"✅ {docker_result.get('message', 'Docker готов')}")
            
            remote_image_path = f"{deploy_remote_path}/{image_path.name}"
            await safe_reply_text(update, f"📤 Загружаю образ на сервер: {deploy_image_tar_path}...")
            upload_result = await deploy_upload_image(
                deploy_ssh_host, deploy_ssh_port, deploy_ssh_username, deploy_ssh_password,
                deploy_image_tar_path, remote_image_path
            )
            if not upload_result or upload_result.get("status") != "success":
                error_msg = upload_result.get("message", "Неизвестная ошибка") if upload_result else "Ошибка при загрузке образа"
                await safe_reply_text(update, f"❌ Ошибка при загрузке образа: {error_msg}")
                return
            await safe_reply_text(update, f"✅ {upload_result.get('message', 'Образ загружен')}")
            
            await safe_reply_text(update, "🐳 Загружаю образ в Docker...")
            load_result = await deploy_load_image(
                deploy_ssh_host, deploy_ssh_port, deploy_ssh_username, deploy_ssh_password,
                remote_image_path
            )
            if not load_result or load_result.get("status") != "success":
                error_msg = load_result.get("message", "Неизвестная ошибка") if load_result else "Ошибка при загрузке образа в Docker"
                await safe_reply_text(update, f"❌ Ошибка при загрузке образа в Docker: {error_msg}")
                return
            await safe_reply_text(update, f"✅ {load_result.get('message', 'Образ загружен в Docker')}")
            
            compose_path = f"{deploy_remote_path}/docker-compose.yml"
            compose_content = f"""services:
  bot:
    image: {image_name}:{image_tag}
    container_name: nikita_ai_bot
    restart: unless-stopped
    network_mode: host
    env_file:
      - .env
    environment:
      - DB_PATH=/app/data/bot_memory.sqlite3
    volumes:
      - ./data:/app/data
      - ./digests:/app/bot/digests
    user: "0:0"
"""
            await safe_reply_text(update, "📝 Создаю docker-compose.yml...")
            compose_result = await deploy_create_compose(
                deploy_ssh_host, deploy_ssh_port, deploy_ssh_username, deploy_ssh_password,
                compose_content, compose_path
            )
            if not compose_result or compose_result.get("status") != "success":
                error_msg = compose_result.get("message", "Неизвестная ошибка") if compose_result else "Ошибка при создании docker-compose.yml"
                await safe_reply_text(update, f"❌ Ошибка при создании docker-compose.yml: {error_msg}")
                return
            compose_msg = compose_result.get('message', 'docker-compose.yml создан')
            if compose_result.get('skipped'):
                await safe_reply_text(update, f"⏭️ {compose_msg}")
            else:
                await safe_reply_text(update, f"✅ {compose_msg}")
            
            env_path = f"{deploy_remote_path}/.env"
            env_content = f"""TELEGRAM_BOT_TOKEN={deploy_bot_token}
OPENROUTER_API_KEY={deploy_openrouter_api_key}
OPENROUTER_MODEL={deploy_openrouter_model}
EMBEDDING_MODEL={deploy_embedding_model}
RAG_SIM_THRESHOLD={deploy_rag_sim_threshold}
RAG_TOP_K={deploy_rag_top_k}
OLLAMA_BASE_URL={deploy_ollama_base_url}
OLLAMA_MODEL={deploy_ollama_model}
OLLAMA_TIMEOUT={deploy_ollama_timeout}
OLLAMA_TEMPERATURE={deploy_ollama_temperature}
OLLAMA_NUM_CTX={deploy_ollama_num_ctx}
OLLAMA_NUM_PREDICT={deploy_ollama_num_predict}
OLLAMA_SYSTEM_PROMPT={deploy_ollama_system_prompt}
"""
            await safe_reply_text(update, "📝 Проверяю .env файл...")
            env_result = await deploy_create_env(
                deploy_ssh_host, deploy_ssh_port, deploy_ssh_username, deploy_ssh_password,
                env_content, env_path
            )
            if not env_result or env_result.get("status") != "success":
                error_msg = env_result.get("message", "Неизвестная ошибка") if env_result else "Ошибка при создании .env файла"
                await safe_reply_text(update, f"❌ Ошибка при создании .env файла: {error_msg}")
                return
            env_msg = env_result.get('message', '.env файл создан')
            if env_result.get('skipped'):
                await safe_reply_text(update, f"⏭️ {env_msg}")
            else:
                await safe_reply_text(update, f"✅ {env_msg}")
            
            await safe_reply_text(update, "🚀 Запускаю бота...")
            start_result = await deploy_start_bot(
                deploy_ssh_host, deploy_ssh_port, deploy_ssh_username, deploy_ssh_password,
                compose_path
            )
            if not start_result or start_result.get("status") != "success":
                error_msg = start_result.get("message", "Неизвестная ошибка") if start_result else "Ошибка при запуске бота"
                await safe_reply_text(update, f"❌ Ошибка при запуске бота: {error_msg}")
                return
            
            await asyncio.sleep(3)
            
            await safe_reply_text(update, "🔍 Проверяю статус контейнера...")
            container_result = await deploy_check_container(
                deploy_ssh_host, deploy_ssh_port, deploy_ssh_username, deploy_ssh_password
            )
            
            if container_result:
                container_status = container_result.get("container_status", "неизвестно")
                container_list = container_result.get("container_list", "")
                container_id = container_result.get("container_id", "")
                logs = container_result.get("logs", "")
                logs_preview = logs[-1000:] if len(logs) > 1000 else logs
                
                status_msg = f"✅ Деплой завершен успешно!\n\n"
                status_msg += f"Бот запущен на сервере {deploy_ssh_host}\n"
                status_msg += f"Путь: {deploy_remote_path}\n"
                status_msg += f"Контейнер: nikita_ai_bot\n"
                status_msg += f"Статус: {container_status}\n"
                if container_id:
                    status_msg += f"ID: {container_id}\n"
                if container_list:
                    status_msg += f"\nВсе контейнеры:\n{container_list}\n"
                status_msg += f"\nПоследние логи:\n```\n{logs_preview}\n```"
                
                await safe_reply_text(update, status_msg)
            else:
                await safe_reply_text(
                    update,
                    f"✅ Деплой завершен успешно!\n\n"
                    f"Бот запущен на сервере {deploy_ssh_host}\n"
                    f"Путь: {deploy_remote_path}\n"
                    f"Контейнер: nikita_ai_bot\n\n"
                    f"⚠️ Не удалось получить логи контейнера. Проверьте вручную: docker logs nikita_ai_bot"
                )
            
        except Exception as e:
            logger.exception(f"Error in deploy_bot_cmd: {e}")
            await safe_reply_text(update, f"❌ Ошибка при деплое: {e}")


class StopBotHandler(Handler):
    """Handler for /stop_bot command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /stop_bot command."""
        if not update.message:
            return
        
        try:
            deploy_ssh_host = os.getenv("DEPLOY_SSH_HOST", "").strip()
            deploy_ssh_port = int(os.getenv("DEPLOY_SSH_PORT", "22"))
            deploy_ssh_username = os.getenv("DEPLOY_SSH_USERNAME", "").strip()
            deploy_ssh_password = os.getenv("DEPLOY_SSH_PASSWORD", "").strip()
            deploy_remote_path = os.getenv("DEPLOY_REMOTE_PATH", "/opt/nikita_ai").strip()
            
            if not deploy_ssh_host or not deploy_ssh_username or not deploy_ssh_password:
                await safe_reply_text(
                    update,
                    "❌ Ошибка: Не заданы переменные окружения для деплоя.\n\n"
                    "Необходимо задать:\n"
                    "- DEPLOY_SSH_HOST\n"
                    "- DEPLOY_SSH_USERNAME\n"
                    "- DEPLOY_SSH_PASSWORD"
                )
                return
            
            compose_path = f"{deploy_remote_path}/docker-compose.yml"
            
            args = context.args or []
            remove_volumes = "--remove-volumes" in args or "-v" in args
            remove_images = "--remove-images" in args or "-i" in args
            
            if not args:
                await safe_reply_text(
                    update,
                    f"⚠️ Остановка бота на сервере {deploy_ssh_host}\n\n"
                    f"Использование:\n"
                    f"/stop_bot - остановить контейнер\n"
                    f"/stop_bot -v - остановить и удалить данные\n"
                    f"/stop_bot -i - остановить и удалить образы\n"
                    f"/stop_bot -v -i - полное удаление"
                )
                return
            
            await safe_reply_text(update, "🛑 Останавливаю бота...")
            
            stop_result = await deploy_stop_bot(
                deploy_ssh_host, deploy_ssh_port, deploy_ssh_username, deploy_ssh_password,
                compose_path, remove_volumes=remove_volumes, remove_images=remove_images
            )
            
            if stop_result and stop_result.get("status") == "success":
                message = stop_result.get("message", "Бот остановлен")
                await safe_reply_text(update, f"✅ {message}")
            else:
                error_msg = stop_result.get("message", "Неизвестная ошибка") if stop_result else "Ошибка при остановке бота"
                await safe_reply_text(update, f"❌ {error_msg}")
            
        except Exception as e:
            logger.exception(f"Error in stop_bot_cmd: {e}")
            await safe_reply_text(update, f"❌ Ошибка при остановке бота: {e}")


async def deploy_bot_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /deploy_bot."""
    handler = DeployBotHandler()
    await handler.handle(update, context)


async def stop_bot_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command function for /stop_bot."""
    handler = StopBotHandler()
    await handler.handle(update, context)
