#!/usr/bin/env python3
"""
Скрипт для автоматического ревью Pull Request с использованием RAG и MCP.

Использование:
    python scripts/review_pr.py <owner> <repo> <pr_number> <github_token>

Пример:
    python scripts/review_pr.py RomAn-8 nikita_ai 123 $GITHUB_TOKEN
"""

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx

# Добавляем корень проекта в путь для импорта модулей bot
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Загружаем переменные окружения перед импортом config
# Это нужно для работы в GitHub Actions, где .env файла нет
# Устанавливаем фиктивный TELEGRAM_BOT_TOKEN, чтобы избежать ошибки при импорте config
if not os.getenv("TELEGRAM_BOT_TOKEN"):
    os.environ["TELEGRAM_BOT_TOKEN"] = "dummy_token_for_script"

from bot.config import OPENROUTER_API_KEY, OPENROUTER_MODEL, RAG_SIM_THRESHOLD, RAG_TOP_K, EMBEDDING_MODEL
from bot.embeddings import search_relevant_chunks, has_embeddings
from bot.mcp_client import get_pr_diff, get_pr_files, get_pr_info
from bot.openrouter import chat_completion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def extract_keywords_from_text(text: str) -> str:
    """Извлекает ключевые слова из текста для RAG поиска."""
    # Убираем markdown разметку
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]+`", "", text)
    text = re.sub(r"#+\s*", "", text)
    
    # Извлекаем имена функций, классов, переменных
    patterns = [
        r"def\s+(\w+)",  # функции
        r"class\s+(\w+)",  # классы
        r"(\w+)\s*=",  # переменные
        r"@(\w+)",  # декораторы
    ]
    
    keywords = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        keywords.update(matches)
    
    # Добавляем слова из текста (исключая служебные)
    words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", text)
    stop_words = {"the", "and", "or", "but", "for", "with", "from", "this", "that", "are", "was", "were", "been", "have", "has", "had", "will", "would", "should", "could", "may", "might", "must", "can"}
    keywords.update(w.lower() for w in words if w.lower() not in stop_words and len(w) > 3)
    
    return " ".join(list(keywords)[:20])  # Ограничиваем количество ключевых слов


async def get_rag_context(pr_info: dict[str, Any], pr_files: list[dict[str, Any]], pr_diff: str) -> str:
    """Использует RAG для поиска релевантной документации."""
    if not has_embeddings(EMBEDDING_MODEL):
        logger.warning("No embeddings found in database. Skipping RAG context.")
        return ""
    
    # Формируем запросы для RAG
    queries = []
    
    # 1. По названию и описанию PR
    if pr_info.get("title"):
        queries.append(pr_info["title"])
    if pr_info.get("body"):
        queries.append(pr_info["body"][:500])  # Ограничиваем длину
    
    # 2. По именам файлов
    file_names = [f.get("filename", "") for f in pr_files]
    if file_names:
        queries.append(" ".join(file_names))
    
    # 3. По ключевым словам из diff
    if pr_diff:
        keywords = extract_keywords_from_text(pr_diff[:2000])  # Ограничиваем длину
        if keywords:
            queries.append(keywords)
    
    # Ищем релевантные чанки для каждого запроса
    all_chunks = []
    seen_chunks = set()
    
    for query in queries:
        if not query.strip():
            continue
        
        try:
            chunks = search_relevant_chunks(
                query,
                model=EMBEDDING_MODEL,
                top_k=RAG_TOP_K,
                min_similarity=RAG_SIM_THRESHOLD,
                apply_threshold=True,
            )
            
            for chunk in chunks:
                chunk_key = (chunk["doc_name"], chunk["chunk_index"])
                if chunk_key not in seen_chunks:
                    seen_chunks.add(chunk_key)
                    all_chunks.append(chunk)
        except Exception as e:
            logger.warning(f"Error searching chunks for query '{query[:50]}...': {e}")
            continue
    
    # Форматируем контекст
    if not all_chunks:
        return ""
    
    context_parts = ["Релевантная документация из проекта:\n"]
    for i, chunk in enumerate(all_chunks[:5], 1):  # Ограничиваем до 5 чанков
        context_parts.append(f"[Документ {i}: {chunk['doc_name']}, фрагмент {chunk['chunk_index']}, релевантность {chunk['similarity']:.3f}]")
        context_parts.append(chunk["text"])
        context_parts.append("")
    
    return "\n".join(context_parts)


def format_pr_files(files: list[dict[str, Any]]) -> str:
    """Форматирует список файлов для промпта."""
    if not files:
        return "Нет измененных файлов."
    
    parts = ["Измененные файлы:"]
    for file_info in files:
        filename = file_info.get("filename", "unknown")
        status = file_info.get("status", "unknown")
        additions = file_info.get("additions", 0)
        deletions = file_info.get("deletions", 0)
        parts.append(f"- {filename} ({status}): +{additions}/-{deletions}")
    
    return "\n".join(parts)


def create_review_prompt(pr_info: dict[str, Any], pr_files: list[dict[str, Any]], pr_diff: str, rag_context: str) -> list[dict[str, str]]:
    """Создает промпт для LLM для генерации ревью."""
    
    system_prompt = """Ты - опытный code reviewer, который анализирует Pull Request и предоставляет конструктивную обратную связь.

Твоя задача:
1. Найти потенциальные баги и проблемы
2. Проверить соответствие кода стилю и архитектуре проекта
3. Предложить улучшения
4. Задать вопросы, если что-то непонятно

Формат ответа:
## 🔍 Найденные проблемы
- [Описание проблемы с указанием файла и строки, если возможно]

## 💡 Предложения по улучшению
- [Конкретные предложения]

## ❓ Вопросы
- [Вопросы к автору PR]

## ✅ Положительные моменты
- [Что сделано хорошо]

Будь конкретным, конструктивным и вежливым. Если проблем не найдено, так и напиши."""

    user_parts = []
    
    # Информация о PR
    user_parts.append(f"## Информация о PR")
    user_parts.append(f"Название: {pr_info.get('title', 'N/A')}")
    user_parts.append(f"Описание: {pr_info.get('body', 'N/A')[:500]}")
    user_parts.append(f"Автор: {pr_info.get('author', 'N/A')}")
    user_parts.append(f"Ветки: {pr_info.get('head_branch', 'N/A')} → {pr_info.get('base_branch', 'N/A')}")
    user_parts.append("")
    
    # Файлы
    user_parts.append(format_pr_files(pr_files))
    user_parts.append("")
    
    # RAG контекст
    if rag_context:
        user_parts.append(rag_context)
        user_parts.append("")
    
    # Diff
    user_parts.append("## Diff изменений")
    # Ограничиваем размер diff (GitHub API может вернуть очень большой diff)
    max_diff_length = 15000
    if len(pr_diff) > max_diff_length:
        user_parts.append(f"[Diff обрезан, показаны первые {max_diff_length} символов]")
        pr_diff = pr_diff[:max_diff_length] + "\n... [diff обрезан]"
    user_parts.append("```diff")
    user_parts.append(pr_diff)
    user_parts.append("```")
    
    user_content = "\n".join(user_parts)
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


async def post_review_comment(owner: str, repo: str, pr_number: int, github_token: str, review_text: str) -> bool:
    """Публикует комментарий с ревью в PR через GitHub API."""
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {github_token}",
        "User-Agent": "nikita_ai-review-bot/1.0",
    }
    
    body = {
        "body": review_text,
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            logger.info(f"Review comment posted successfully to PR #{pr_number}")
            return True
    except httpx.HTTPStatusError as e:
        logger.error(f"Failed to post comment: {e.response.status_code} - {e.response.text}")
        return False
    except Exception as e:
        logger.error(f"Exception posting comment: {e}")
        return False


async def review_pr(owner: str, repo: str, pr_number: int, github_token: str) -> int:
    """Основная функция для ревью PR."""
    logger.info(f"Starting review for PR #{pr_number} in {owner}/{repo}")
    
    # 1. Получаем данные PR через MCP
    logger.info("Fetching PR data via MCP...")
    pr_info = await get_pr_info(owner, repo, pr_number, github_token)
    if not pr_info:
        logger.error("Failed to get PR info via MCP")
        return 1
    
    pr_files = await get_pr_files(owner, repo, pr_number, github_token)
    if pr_files is None:
        logger.error("Failed to get PR files via MCP")
        return 1
    
    pr_diff = await get_pr_diff(owner, repo, pr_number, github_token)
    if not pr_diff:
        logger.error("Failed to get PR diff via MCP")
        return 1
    
    logger.info(f"PR: {pr_info.get('title', 'N/A')}")
    logger.info(f"Files changed: {len(pr_files)}")
    logger.info(f"Diff length: {len(pr_diff)} characters")
    
    # 2. Получаем RAG контекст
    logger.info("Searching for relevant documentation via RAG...")
    rag_context = await get_rag_context(pr_info, pr_files, pr_diff)
    if rag_context:
        logger.info("Found relevant documentation via RAG")
    else:
        logger.info("No relevant documentation found via RAG")
    
    # 3. Генерируем ревью через LLM
    logger.info("Generating review via LLM...")
    messages = create_review_prompt(pr_info, pr_files, pr_diff, rag_context)
    
    try:
        review_text = chat_completion(messages, temperature=0.3, model=OPENROUTER_MODEL)
        if not review_text or not review_text.strip():
            logger.error("LLM returned empty review")
            return 1
        
        logger.info(f"Review generated ({len(review_text)} characters)")
    except Exception as e:
        logger.error(f"Error generating review: {e}")
        return 1
    
    # 4. Публикуем комментарий в PR
    logger.info("Posting review comment to PR...")
    success = await post_review_comment(owner, repo, pr_number, github_token, review_text)
    
    if success:
        logger.info("Review completed successfully!")
        return 0
    else:
        logger.error("Failed to post review comment")
        return 1


def main():
    """Точка входа скрипта."""
    if len(sys.argv) != 5:
        print("Usage: python scripts/review_pr.py <owner> <repo> <pr_number> <github_token>")
        sys.exit(1)
    
    owner = sys.argv[1]
    repo = sys.argv[2]
    try:
        pr_number = int(sys.argv[3])
    except ValueError:
        print(f"Error: PR number must be an integer, got: {sys.argv[3]}")
        sys.exit(1)
    github_token = sys.argv[4]
    
    # Если токен не передан как аргумент, пробуем получить из переменных окружения
    if not github_token:
        github_token = os.getenv("GB_TOKEN", "").strip() or os.getenv("GITHUB_TOKEN", "").strip()
    
    if not github_token:
        print("Error: GitHub token is required. Set GB_TOKEN or GITHUB_TOKEN environment variable, or pass as 4th argument.")
        sys.exit(1)
    
    exit_code = asyncio.run(review_pr(owner, repo, pr_number, github_token))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
