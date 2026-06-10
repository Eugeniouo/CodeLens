"""Модуль для преобразования кодовой базы в эмбеддинги."""

from sentence_transformers import SentenceTransformer
import numpy as np
from src.config import MODEL_NAME

def get_embeddings(texts: list[str], model_name: str = MODEL_NAME) -> np.ndarray:
    """
    Вычисляет эмбеддинги для списка текстов.

    Args:
        texts (list[str]): список текстов
        model_name (str): имя модели sentence-transformers

    Returns:
        np.ndarray: матрица эмбеддингов
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings