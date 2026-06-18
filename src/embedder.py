"""Модуль для преобразования кодовой базы в эмбеддинги."""

import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

from src.config import MODEL_NAME


def chunk_to_text(chunk: dict) -> str:
    """
    Преобразует чанк кода в текст для подачи в модель.

    Args:
        chunk: Словарь с полями name, file_path, docstring, source_code.

    Returns:
        Строка, готовая для векторизации.
    """
    #ATTENTION: возможно, стоит передавать не только имя, но и путь к файлу, если система будет работать не точно.
    parts = [
        f"passage: Type: {chunk['type']}", #passage - префикс для улучшения качества поиска e5-small
        f"Name: {chunk['name']}",
        f"File: {chunk['file_path']}",
    ]

    if chunk.get("docstring"):
        parts.append(f"Description: {chunk['docstring']}")
    parts.append(f"Code:\n{chunk['source_code']}")
    
    return "\n".join(parts)


def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """
    Загружает модель sentence-transformers. Если она есть локально - берем без
    проверок онлайн. Если нету - загружаем.

    Args:
        model_name: Имя модели из HuggingFace.

    Returns:
        Загруженная модель.
    """
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        print(f"Модель {model_name} не найдена в кэше. Загружаю...")
        return SentenceTransformer(model_name)

def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    show_progress_bar: bool = True,
) -> np.ndarray:
    """
    Векторизует список текстов.

    Args:
        model: Загруженная модель SentenceTransformer.
        texts: Список строк для векторизации.
        show_progress_bar: Показывать прогресс.

    Returns:
        Матрица эмбеддингов формы (len(texts), embedding_dim).
    """
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )


def encode_chunks(
    model: SentenceTransformer,
    chunks: list[dict],
    show_progress_bar: bool = True,
) -> np.ndarray:
    """
    Векторизует список чанков кода.

    Обёртка, принимает сразу чанки, а не строки.

    Args:
        model: Загруженная модель.
        chunks: Список чанков из parser.py.
        show_progress_bar: Показывать прогресс.

    Returns:
        Матрица эмбеддингов формы (len(chunks), embedding_dim).
    """
    texts = [chunk_to_text(chunk) for chunk in chunks]
    return encode_texts(model, texts, show_progress_bar=show_progress_bar)


def encode_query(model: SentenceTransformer, query: str) -> np.ndarray:
    """
    Векторизует один поисковый запрос.

    Args:
        model: Загруженная модель.
        query: Текст запроса пользователя.

    Returns:
        Вектор запроса формы (embedding_dim,).
    """
    query_text = f"query: {query}" #query - префикс для улучшения качества поиска e5-small
    return encode_texts(model, [query_text], show_progress_bar=False)[0]