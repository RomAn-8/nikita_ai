"""
Модуль для подписки на погоду с периодическим сбором данных и отправкой summary.
"""

import asyncio
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Any

from telegram import Bot
from telegram.ext import ContextTypes

from .mcp_weather import get_weather_via_mcp

# Самарское время (UTC+4)
SAMARA_TIMEZONE = timezone(timedelta(hours=4))

logger = logging.getLogger(__name__)


def _format_weather_for_summary(city: str, weather_text: str) -> str:
    """
    Форматирует текст погоды в формат для summary.
    Пример: "Москва, −2°C, ясно, ветер 6 м/с"

    Args:
        city: Название города
        weather_text: Текст погоды от MCP (формат: "Погода в Москва: -8°С; Состояние: небольшой снег; Влажность: 80%; Ветер: 1 м/с")

    Returns:
        Отформатированная строка
    """
    # Парсим компоненты из weather_text
    # Формат: "Погода в Москва: -8°С; Состояние: небольшой снег; Влажность: 80%; Ветер: 1 м/с"
    temp_match = re.search(r"Погода в [^:]+:\s*([^;°]+[°СC])", weather_text)
    condition_match = re.search(r"Состояние:\s*([^;]+)", weather_text)
    wind_match = re.search(r"Ветер:\s*([^;]+)", weather_text)

    temp = temp_match.group(1).strip() if temp_match else "нет данных"
    condition = condition_match.group(1).strip() if condition_match else "нет данных"
    wind = wind_match.group(1).strip() if wind_match else "нет данных"

    # Форматируем: "Москва, −2°C, ясно, ветер 6 м/с"
    return f"{city}, {temp}, {condition}, ветер {wind}"

# Режим для хранения записей погоды в БД
MODE_WEATHER_SUB = "weather_sub"

# Интервал сбора погоды (10 секунд)
WEATHER_COLLECT_INTERVAL = 10


async def weather_subscription_task(
    chat_id: int,
    city: str,
    summary_interval: int,
    bot: Bot,
    context: ContextTypes.DEFAULT_TYPE,
    db_add_message: Any,  # функция для записи в БД
) -> None:
    """
    Фоновая задача для сбора погоды и отправки summary.

    Args:
        chat_id: ID чата
        city: Название города
        summary_interval: Интервал отправки summary в секундах
        context: Контекст Telegram бота
        db_add_message: Функция для записи в БД
        safe_reply_text: Функция для безопасной отправки сообщений
    """
    subscription_key = f"weather_sub_{chat_id}_{city}"
    logger.info(f"Starting weather subscription for chat_id={chat_id}, city={city}, interval={summary_interval}s")

    try:
        # Отправляем подтверждение начала подписки
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=f"✅ Подписка на погоду для {city} активирована.\n"
                f"Сбор данных каждые {WEATHER_COLLECT_INTERVAL} секунд.\n"
                f"Summary каждые {summary_interval} секунд.",
            )
        except Exception as e:
            logger.exception(f"Failed to send subscription start message: {e}")

        weather_records: list[tuple[str, str]] = []  # [(timestamp, weather_text), ...]

        try:
            # Ждём 10 секунд перед первым сбором
            try:
                await asyncio.sleep(WEATHER_COLLECT_INTERVAL)
            except asyncio.CancelledError:
                raise

            # Устанавливаем время начала отсчёта summary ПОСЛЕ задержки, перед первой записью
            # Это гарантирует, что summary будет отправляться каждые summary_interval секунд
            last_summary_time = datetime.now(SAMARA_TIMEZONE)

            while True:
                try:
                    # Получаем текущую погоду через MCP
                    weather_text = await get_weather_via_mcp(city)
                    current_time = datetime.now(SAMARA_TIMEZONE)  # Самарское время (UTC+4)
                    timestamp_str = current_time.strftime("%H:%M:%S")

                    # Парсим погоду и форматируем в нужный формат: "Москва, −2°C, ясно, ветер 6 м/с"
                    formatted_weather = _format_weather_for_summary(city, weather_text)

                    # Записываем в БД
                    db_add_message(chat_id, MODE_WEATHER_SUB, "assistant", f"{timestamp_str} — {formatted_weather}")

                    # Добавляем в список для summary
                    weather_records.append((timestamp_str, formatted_weather))

                    # Проверяем, нужно ли отправить summary (по времени, каждые summary_interval секунд)
                    elapsed = (current_time - last_summary_time).total_seconds()
                    if elapsed >= summary_interval:
                        # Формируем summary из всех накопленных записей
                        summary_lines = []
                        for ts, wt in weather_records:
                            summary_lines.append(f"{ts} — {wt}")

                        if summary_lines:
                            summary_text = "\n".join(summary_lines)
                            try:
                                await bot.send_message(chat_id=chat_id, text=summary_text)
                            except Exception as e:
                                logger.exception(f"Failed to send weather summary: {e}")

                        # Очищаем записи и обновляем время последнего summary
                        weather_records.clear()
                        last_summary_time = current_time

                    # Ждём 10 секунд до следующего сбора (с проверкой на отмену)
                    try:
                        await asyncio.sleep(WEATHER_COLLECT_INTERVAL)
                    except asyncio.CancelledError:
                        # Отмена во время ожидания - выходим из цикла
                        raise

                except asyncio.CancelledError:
                    # Отмена задачи - выходим из цикла
                    logger.info(f"Weather subscription cancelled for chat_id={chat_id}, city={city}")
                    raise
                except Exception as e:
                    logger.exception(f"Error in weather subscription task: {e}")
                    # Продолжаем работу даже при ошибке, но проверяем на отмену
                    try:
                        await asyncio.sleep(WEATHER_COLLECT_INTERVAL)
                    except asyncio.CancelledError:
                        raise
        except asyncio.CancelledError:
            # Обработка отмены на верхнем уровне
            logger.info(f"Weather subscription task cancelled for chat_id={chat_id}, city={city}")
            # Не отправляем сообщение об остановке здесь, т.к. это может вызвать проблемы
            # Сообщение об остановке отправляется из команды weather_sub_stop_cmd
            raise
    except Exception as e:
        logger.exception(f"Fatal error in weather subscription task: {e}")
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=f"❌ Ошибка в подписке на погоду для {city}: {e}",
            )
        except Exception:
            pass
    finally:
        # Удаляем подписку из активных
        try:
            if "weather_subscriptions" in context.bot_data:
                if subscription_key in context.bot_data["weather_subscriptions"]:
                    del context.bot_data["weather_subscriptions"][subscription_key]
        except Exception:
            pass


def start_weather_subscription(
    chat_id: int,
    city: str,
    summary_interval: int,
    bot: Bot,
    context: ContextTypes.DEFAULT_TYPE,
    db_add_message: Any,
) -> None:
    """
    Запускает фоновую задачу подписки на погоду.

    Args:
        chat_id: ID чата
        city: Название города
        summary_interval: Интервал отправки summary в секундах
        context: Контекст Telegram бота
        db_add_message: Функция для записи в БД
        safe_reply_text: Функция для безопасной отправки сообщений
    """
    subscription_key = f"weather_sub_{chat_id}_{city}"

    # Инициализируем словарь подписок, если его нет
    if "weather_subscriptions" not in context.bot_data:
        context.bot_data["weather_subscriptions"] = {}

    # Останавливаем предыдущую подписку, если она есть
    if subscription_key in context.bot_data["weather_subscriptions"]:
        task = context.bot_data["weather_subscriptions"][subscription_key]
        if not task.done():
            task.cancel()

    # Создаём новую задачу
    task = asyncio.create_task(
        weather_subscription_task(
            chat_id=chat_id,
            city=city,
            summary_interval=summary_interval,
            bot=bot,
            context=context,
            db_add_message=db_add_message,
        )
    )

    # Сохраняем задачу
    context.bot_data["weather_subscriptions"][subscription_key] = task
    logger.info(f"Started weather subscription task: {subscription_key}")


def stop_weather_subscription(chat_id: int, city: str, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """
    Останавливает подписку на погоду.

    Args:
        chat_id: ID чата
        city: Название города
        context: Контекст Telegram бота

    Returns:
        True если подписка была остановлена, False если её не было
    """
    subscription_key = f"weather_sub_{chat_id}_{city}"

    if "weather_subscriptions" not in context.bot_data:
        return False

    if subscription_key not in context.bot_data["weather_subscriptions"]:
        return False

    task = context.bot_data["weather_subscriptions"][subscription_key]
    if not task.done():
        # Отменяем задачу, но не ждём её завершения (чтобы не блокировать)
        task.cancel()
        # Удаляем из словаря сразу, чтобы избежать повторных попыток остановки
        del context.bot_data["weather_subscriptions"][subscription_key]
        logger.info(f"Stopped weather subscription: {subscription_key}")
        return True
    else:
        # Задача уже завершена, просто удаляем
        del context.bot_data["weather_subscriptions"][subscription_key]
        logger.info(f"Removed completed weather subscription: {subscription_key}")
        return True
