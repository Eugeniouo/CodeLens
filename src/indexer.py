"""Модуль для сохранения векторов в ChromaDB."""

import time
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from .embedder import chunk_to_text, encode_texts, load_model
from src.config import DB_PATH, COLLECTION_NAME

type CodeChunk = dict[str, Any]
type MetadataValue = str | int | float | bool
type IndexStats = dict[str, int | float]


def get_collection(
    db_path: str = DB_PATH,
    collection_name: str = COLLECTION_NAME,
    reset: bool = False,
):
    """
    Создаёт или открывает коллекцию ChromaDB.

    Используем cosine, потому что embedder.py нормализует эмбеддинги
    через normalize_embeddings=True.
    """
    client = chromadb.PersistentClient(path=db_path)

    if reset:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass

    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def get_chunk_id(chunk: CodeChunk) -> str:
    """
    Возвращает ID чанка.

    Основной источник ID - parser.py.
    Там ID уже содержит путь, имя элемента и номер строки.
    """
    chunk_id = chunk.get("id")

    if chunk_id:
        return str(chunk_id)

    file_path = str(chunk.get("file_path", "unknown"))
    name = str(chunk.get("name", "unknown"))
    start_line = str(chunk.get("start_line", "0"))

    return f"{file_path}:{name}:{start_line}"


def clean_metadata_value(value: Any) -> MetadataValue:
    """
    ChromaDB принимает в metadata только простые значения:
    str, int, float, bool.

    Если попадётся что-то сложное, например list или dict,
    приводим к строке.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return value

    if isinstance(value, str):
        return value

    return str(value)


def build_metadata(chunk: CodeChunk) -> dict[str, MetadataValue]:
    """
    Формирует metadata для ChromaDB.

    Поля name и path оставлены специально под текущий searcher.py:

        metadata.get("name", "unknown")
        metadata.get("path", "unknown")
    """
    name = str(chunk.get("name", "unknown"))
    file_path = str(chunk.get("file_path", "unknown"))
    chunk_type = str(chunk.get("type", "unknown"))

    metadata: dict[str, Any] = {
        "id": get_chunk_id(chunk),
        "name": name,
        "path": file_path,
        "file_path": file_path,
        "type": chunk_type,
        "element_path": f"File: {file_path}\nName: {name}",
    }

    if chunk.get("start_line") is not None:
        metadata["start_line"] = chunk["start_line"]

    if chunk.get("end_line") is not None:
        metadata["end_line"] = chunk["end_line"]

    if chunk.get("docstring"):
        metadata["docstring"] = chunk["docstring"]

    cleaned_metadata: dict[str, MetadataValue] = {}

    for key, value in metadata.items():
        if value is not None:
            cleaned_metadata[key] = clean_metadata_value(value)

    return cleaned_metadata


def get_document(chunk: CodeChunk) -> str:
    """
    Возвращает документ, который будет храниться в ChromaDB.

    Важно:
    эмбеддинг строится не по этому тексту напрямую, а по chunk_to_text(chunk).
    А document лучше хранить как исходный код, потому что searcher.py
    возвращает document как source_code.
    """
    source_code = chunk.get("source_code")

    if source_code:
        return str(source_code)

    return chunk_to_text(chunk)


def index_chunks(
    chunks: list[CodeChunk],
    texts: list[str] | None = None,
    db_path: str = DB_PATH,
    collection_name: str = COLLECTION_NAME,
    model: SentenceTransformer | None = None,
    batch_size: int = 128,
    reset: bool = False,
) -> IndexStats:
    """
    Сохраняет чанки и их эмбеддинги в ChromaDB.

    Args:
        chunks: Чанки из parser.py.
        texts: Тексты для эмбеддинга. Обычно это [chunk_to_text(chunk)].
        db_path: Путь к директории ChromaDB.
        collection_name: Название коллекции.
        model: Загруженная модель SentenceTransformer.
        batch_size: Размер батча записи.
        reset: Удалить старую коллекцию перед индексацией.

    Returns:
        Статистика индексации.
    """
    started_at = time.perf_counter()

    if not chunks:
        return {
            "chunks_indexed": 0,
            "files_indexed": 0,
            "time_seconds": 0.0,
        }

    if texts is None:
        texts = [chunk_to_text(chunk) for chunk in chunks]

    if len(texts) != len(chunks):
        raise ValueError(
            "Количество texts должно совпадать с количеством chunks."
        )

    if model is None:
        model = load_model()

    collection = get_collection(
        db_path=db_path,
        collection_name=collection_name,
        reset=reset,
    )

    print("Создаём эмбеддинги...")
    embeddings = encode_texts(
        model=model,
        texts=texts,
        show_progress_bar=True,
    )

    ids = [get_chunk_id(chunk) for chunk in chunks]
    documents = [get_document(chunk) for chunk in chunks]
    metadatas = [build_metadata(chunk) for chunk in chunks]

    print("Сохраняем чанки в ChromaDB...")

    for start in range(0, len(chunks), batch_size):
        end = start + batch_size

        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

        indexed_count = min(end, len(chunks))
        print(f"Сохранено чанков: {indexed_count}/{len(chunks)}")

    files_indexed = len(
        {
            str(chunk.get("file_path"))
            for chunk in chunks
            if chunk.get("file_path")
        }
    )

    elapsed = round(time.perf_counter() - started_at, 2)

    return {
        "chunks_indexed": len(chunks),
        "files_indexed": files_indexed,
        "time_seconds": elapsed,
    }