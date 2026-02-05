"""
Модуль для работы с эмбеддингами через OpenRouter API.
"""
import json
import math
import re
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .config import OPENROUTER_API_KEY

logger = logging.getLogger(__name__)

# Константы
EMBEDDING_MODEL = "google/gemini-embedding-001"
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
CHUNK_SIZE = 1000  # символов
CHUNK_OVERLAP = 150  # символов

# Путь к БД
DB_PATH = Path(__file__).resolve().parent / "bot_memory.sqlite3"


def open_db() -> sqlite3.Connection:
    """Открывает соединение с БД."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def init_embeddings_table() -> None:
    """Создает таблицу doc_chunks если её нет."""
    with open_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS doc_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_name TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL,
                embedding_json TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL,
                model TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(doc_name, chunk_index, model)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc_name_model ON doc_chunks(doc_name, model)"
        )
        conn.commit()


def normalize_text(text: str) -> str:
    """Нормализует текст: убирает лишние пробелы и пустые строки."""
    # Заменяем множественные пробелы на один
    text = re.sub(r" +", " ", text)
    # Заменяем множественные переносы строк на максимум два
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Убираем пробелы в начале и конце
    text = text.strip()
    return text


def _find_word_boundary_backward(text: str, pos: int) -> int:
    """Находит ближайшую границу слова назад от позиции pos."""
    # Границы слов: пробелы, переносы строк, знаки препинания
    word_boundaries = set(" \n\t\r.,;:!?()[]{}\"'")
    
    # Ищем назад не более чем на 100 символов
    search_start = max(0, pos - 100)
    for i in range(pos, search_start - 1, -1):
        if i < len(text) and text[i] in word_boundaries:
            return i + 1  # Возвращаем позицию после границы
    return search_start


def _find_word_boundary_forward(text: str, pos: int) -> int:
    """Находит ближайшую границу слова вперед от позиции pos."""
    # Границы слов: пробелы, переносы строк, знаки препинания
    word_boundaries = set(" \n\t\r.,;:!?()[]{}\"'")
    
    # Ищем вперед не более чем на 100 символов
    search_end = min(len(text), pos + 100)
    for i in range(pos, search_end):
        if i < len(text) and text[i] in word_boundaries:
            return i + 1  # Возвращаем позицию после границы
    return search_end


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict[str, Any]]:
    """
    Разбивает текст на чанки с overlap, разбивая по границам слов.
    
    Returns:
        Список словарей с ключами: text, start_offset, end_offset, chunk_index
    """
    chunks = []
    text_len = len(text)
    
    # Защита от слишком больших файлов
    MAX_CHUNKS = 10000  # Максимальное количество чанков
    if text_len > chunk_size * MAX_CHUNKS:
        raise ValueError(f"Файл слишком большой: {text_len} символов. Максимум: {chunk_size * MAX_CHUNKS} символов")
    
    # Убеждаемся, что overlap меньше chunk_size
    if overlap >= chunk_size:
        overlap = chunk_size // 2
        logger.warning(f"Overlap ({overlap}) слишком большой, уменьшен до {overlap}")
    
    start = 0
    chunk_index = 0
    
    while start < text_len:
        # Предполагаемый конец чанка
        tentative_end = min(start + chunk_size, text_len)
        
        # Если это не последний чанк и не конец текста, ищем границу слова
        if tentative_end < text_len:
            # Проверяем, не заканчивается ли чанк в середине слова
            # Если символ на позиции tentative_end - это буква/цифра, ищем границу слова назад
            if tentative_end > 0 and text[tentative_end - 1].isalnum():
                end = _find_word_boundary_backward(text, tentative_end)
            else:
                # Уже на границе слова, но проверим, не начинается ли следующий чанк в середине слова
                end = tentative_end
        else:
            # Последний чанк - берем до конца
            end = text_len
        
        # Убеждаемся, что end не меньше start
        if end <= start:
            end = min(start + 1, text_len)
        
        # Создаем чанк
        chunk_text = text[start:end].strip()
        
        # Пропускаем пустые чанки
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "start_offset": start,
                "end_offset": end,
                "chunk_index": chunk_index,
            })
            chunk_index += 1
        
        # Защита от бесконечного цикла
        if chunk_index > MAX_CHUNKS:
            raise ValueError(f"Слишком много чанков: {chunk_index}. Возможно, проблема с overlap.")
        
        # Следующий чанк начинается с overlap назад от конца текущего
        if end >= text_len:
            break
        
        # Вычисляем начало следующего чанка с учетом overlap
        next_start = end - overlap
        if next_start < 0:
            next_start = 0
        
        # Если следующий чанк начинается в середине слова, ищем границу слова вперед
        if next_start < text_len and next_start > 0 and text[next_start - 1].isalnum():
            start = _find_word_boundary_forward(text, next_start)
        else:
            start = next_start
        
        # Защита от зацикливания
        if start >= end:
            start = end
        if start == end and end < text_len:
            start = end + 1
    
    return chunks


def generate_embeddings_batch(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """
    Генерирует эмбеддинги для списка текстов батчем через OpenRouter API.
    
    Args:
        texts: Список текстов для эмбеддинга
        model: Модель для эмбеддингов
        
    Returns:
        Список векторов эмбеддингов
    """
    if not texts:
        return []
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model,
        "input": texts,
    }
    
    try:
        response = requests.post(OPENROUTER_EMBEDDINGS_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        # Извлекаем эмбеддинги из ответа
        embeddings = []
        for item in data.get("data", []):
            embedding = item.get("embedding", [])
            if embedding:
                embeddings.append(embedding)
        
        return embeddings
    except Exception as e:
        logger.exception(f"Error generating embeddings: {e}")
        raise


def save_chunks_to_db(
    doc_name: str,
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    model: str = EMBEDDING_MODEL,
) -> tuple[int, int]:
    """
    Сохраняет чанки и эмбеддинги в БД.
    
    Args:
        doc_name: Имя документа
        chunks: Список чанков (с text, start_offset, end_offset, chunk_index)
        embeddings: Список векторов эмбеддингов
        model: Модель эмбеддингов
        
    Returns:
        Кортеж (количество сохраненных записей, размерность эмбеддинга)
    """
    if len(chunks) != len(embeddings):
        raise ValueError(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings")
    
    if not embeddings:
        return 0, 0
    
    embedding_dim = len(embeddings[0])
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    
    saved_count = 0
    with open_db() as conn:
        for chunk, embedding in zip(chunks, embeddings):
            if len(embedding) != embedding_dim:
                raise ValueError(f"Embedding dimension mismatch: expected {embedding_dim}, got {len(embedding)}")
            
            embedding_json = json.dumps(embedding)
            
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO doc_chunks
                    (doc_name, chunk_index, text, start_offset, end_offset, 
                     embedding_json, embedding_dim, model, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_name,
                        chunk["chunk_index"],
                        chunk["text"],
                        chunk["start_offset"],
                        chunk["end_offset"],
                        embedding_json,
                        embedding_dim,
                        model,
                        created_at,
                    ),
                )
                saved_count += 1
            except Exception as e:
                logger.exception(f"Error saving chunk {chunk['chunk_index']}: {e}")
                raise
        
        conn.commit()
    
    return saved_count, embedding_dim


def delete_doc_embeddings(doc_name: str, model: str = EMBEDDING_MODEL) -> int:
    """
    Удаляет все эмбеддинги для документа и модели.
    
    Returns:
        Количество удаленных записей
    """
    with open_db() as conn:
        cursor = conn.execute(
            "DELETE FROM doc_chunks WHERE doc_name = ? AND model = ?",
            (doc_name, model),
        )
        deleted_count = cursor.rowcount
        conn.commit()
    
    return deleted_count


def doc_exists(doc_name: str, model: str = EMBEDDING_MODEL) -> bool:
    """Проверяет, существуют ли эмбеддинги для документа."""
    with open_db() as conn:
        cursor = conn.execute(
            "SELECT COUNT(*) FROM doc_chunks WHERE doc_name = ? AND model = ?",
            (doc_name, model),
        )
        count = cursor.fetchone()[0]
        return count > 0


def clear_all_embeddings() -> int:
    """
    Удаляет все эмбеддинги из базы данных.
    
    Returns:
        Количество удаленных записей
    """
    with open_db() as conn:
        cursor = conn.execute("DELETE FROM doc_chunks")
        deleted_count = cursor.rowcount
        conn.commit()
    
    return deleted_count


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Вычисляет cosine similarity между двумя векторами."""
    if len(a) != len(b):
        raise ValueError(f"Vectors must have the same length: {len(a)} != {len(b)}")
    
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def search_relevant_chunks(
    query_text: str,
    model: str = EMBEDDING_MODEL,
    top_k: int = 3,
    min_similarity: float = 0.5,
    apply_threshold: bool = True,
) -> list[dict[str, Any]]:
    """
    Ищет релевантные чанки по запросу пользователя.
    
    Args:
        query_text: Текст запроса пользователя
        model: Модель эмбеддингов
        top_k: Количество возвращаемых чанков
        min_similarity: Минимальная cosine similarity (используется только если apply_threshold=True)
        apply_threshold: Если True, фильтровать чанки по min_similarity. Если False, возвращать все найденные чанки до top_k
        
    Returns:
        Список словарей с ключами: text, chunk_index, similarity, doc_name
    """
    # Генерируем эмбеддинг для запроса
    query_embeddings = generate_embeddings_batch([query_text], model=model)
    if not query_embeddings:
        return []
    
    query_embedding = query_embeddings[0]
    
    # Загружаем все чанки из БД
    with open_db() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT doc_name, chunk_index, text, embedding_json, embedding_dim
            FROM doc_chunks
            WHERE model = ?
            """,
            (model,),
        )
        rows = cursor.fetchall()
    
    # Вычисляем similarity для каждого чанка
    results = []
    for row in rows:
        try:
            chunk_embedding = json.loads(row["embedding_json"])
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            
            # Если apply_threshold=False, добавляем все чанки независимо от similarity
            # Если apply_threshold=True, фильтруем по min_similarity
            if not apply_threshold or similarity >= min_similarity:
                results.append({
                    "text": row["text"],
                    "chunk_index": row["chunk_index"],
                    "similarity": similarity,
                    "doc_name": row["doc_name"],
                })
        except Exception as e:
            logger.exception(f"Error processing chunk {row['chunk_index']}: {e}")
            continue
    
    # Сортируем по similarity (убывание) и берем top_k
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


def has_embeddings(model: str = EMBEDDING_MODEL) -> bool:
    """Проверяет, есть ли эмбеддинги в базе данных."""
    with open_db() as conn:
        cursor = conn.execute(
            "SELECT COUNT(*) FROM doc_chunks WHERE model = ?",
            (model,),
        )
        count = cursor.fetchone()[0]
        return count > 0


def process_readme_file(
    file_content: str,
    doc_name: str = "README.md",
    model: str = EMBEDDING_MODEL,
    replace_existing: bool = True,
) -> dict[str, Any]:
    """
    Обрабатывает README.md файл: нормализует, разбивает на чанки, генерирует эмбеддинги, сохраняет в БД.
    
    Args:
        file_content: Содержимое файла
        doc_name: Имя документа (по умолчанию "README.md")
        model: Модель эмбеддингов
        replace_existing: Если True, удаляет старые записи перед созданием новых
        
    Returns:
        Словарь со статистикой: {
            "success": bool,
            "doc_name": str,
            "text_length": int,
            "chunks_count": int,
            "embedding_dim": int,
            "model": str,
            "first_chunk_preview": str,
            "first_embedding_preview": list[float],
            "error": str | None,
        }
    """
    try:
        # Инициализируем таблицу
        init_embeddings_table()
        
        # Проверяем существование
        if doc_exists(doc_name, model):
            if replace_existing:
                deleted = delete_doc_embeddings(doc_name, model)
                logger.info(f"Deleted {deleted} existing embeddings for {doc_name}")
            else:
                return {
                    "success": False,
                    "error": f"Эмбеддинги для {doc_name} уже существуют. Используйте replace_existing=True для пересоздания.",
                }
        
        # Проверяем размер файла перед обработкой
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 МБ
        if len(file_content) > MAX_FILE_SIZE:
            return {
                "success": False,
                "error": f"Файл слишком большой: {len(file_content)} байт. Максимум: {MAX_FILE_SIZE} байт (10 МБ)",
            }
        
        # Нормализуем текст
        normalized_text = normalize_text(file_content)
        text_length = len(normalized_text)
        
        # Разбиваем на чанки
        try:
            chunks = split_into_chunks(normalized_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        except ValueError as e:
            return {
                "success": False,
                "error": str(e),
            }
        
        chunks_count = len(chunks)
        
        if chunks_count == 0:
            return {
                "success": False,
                "error": "Текст пуст после нормализации",
            }
        
        # Обрабатываем чанки батчами, чтобы не загружать всю память
        BATCH_SIZE = 50  # Обрабатываем по 50 чанков за раз
        all_embeddings = []
        
        for i in range(0, chunks_count, BATCH_SIZE):
            batch_chunks = chunks[i:i + BATCH_SIZE]
            chunk_texts = [chunk["text"] for chunk in batch_chunks]
            
            # Генерируем эмбеддинги для батча
            batch_embeddings = generate_embeddings_batch(chunk_texts, model=model)
            
            if len(batch_embeddings) != len(batch_chunks):
                return {
                    "success": False,
                    "error": f"Несоответствие количества эмбеддингов в батче {i//BATCH_SIZE + 1}: получено {len(batch_embeddings)}, ожидалось {len(batch_chunks)}",
                }
            
            # Сохраняем батч в БД
            saved_count, embedding_dim = save_chunks_to_db(doc_name, batch_chunks, batch_embeddings, model=model)
            
            if saved_count != len(batch_chunks):
                return {
                    "success": False,
                    "error": f"Ошибка сохранения батча {i//BATCH_SIZE + 1}: сохранено {saved_count} из {len(batch_chunks)}",
                }
            
            all_embeddings.extend(batch_embeddings)
        
        # Проверяем, что все эмбеддинги получены
        if len(all_embeddings) != chunks_count:
            return {
                "success": False,
                "error": f"Несоответствие общего количества эмбеддингов: получено {len(all_embeddings)}, ожидалось {chunks_count}",
            }
        
        saved_count = chunks_count
        
        # Превью первого чанка (первые 200 символов)
        first_chunk_preview = chunks[0]["text"][:200]
        if len(chunks[0]["text"]) > 200:
            first_chunk_preview += "..."
        
        # Превью первого эмбеддинга (первые 10 чисел)
        first_embedding_preview = all_embeddings[0][:10] if all_embeddings else []
        
        return {
            "success": True,
            "doc_name": doc_name,
            "text_length": text_length,
            "chunks_count": chunks_count,
            "embedding_dim": embedding_dim,
            "model": model,
            "first_chunk_preview": first_chunk_preview,
            "first_embedding_preview": first_embedding_preview,
            "error": None,
        }
    except Exception as e:
        logger.exception(f"Error processing README file: {e}")
        return {
            "success": False,
            "error": str(e),
        }
