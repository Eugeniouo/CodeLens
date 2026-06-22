"""Модуль для сохранения векторов в ChromaDB."""

import time
import pickle
import re
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from src.config import COLLECTION_NAME, DB_PATH, BM25_INDEX_PATH
from .embedder import chunk_to_text, encode_texts, load_model

type CodeChunk = dict[str, Any]
type MetadataValue = str | int | float | bool
type IndexStats = dict[str, int | float]

def tokenize(text: str) -> list[str]:
    return re.findall(r'(?u)\b\w+\b', text.lower())

def get_collection(
    db_path: str | Any = DB_PATH,
    collection_name: str = COLLECTION_NAME,
    reset: bool = False,
):
    client = chromadb.PersistentClient(path=str(db_path))

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
    if chunk_id := chunk.get("id"):
        return str(chunk_id)

    return f"{chunk.get('file_path', 'unknown')}:{chunk.get('name', 'unknown')}:{chunk.get('start_line', '0')}"

def build_metadata(chunk: CodeChunk) -> dict[str, MetadataValue]:
    file_path = str(chunk.get("file_path", "unknown"))
    name = str(chunk.get("name", "unknown"))

    meta: dict[str, Any] = {
        "id": get_chunk_id(chunk),
        "name": name,
        "path": file_path,
        "file_path": file_path,
        "type": str(chunk.get("type", "unknown")),
        "element_path": f"File: {file_path}\nName: {name}",
    }

    for key in ("start_line", "end_line", "docstring"):
        if (val := chunk.get(key)) is not None:
            meta[key] = val

    return {
        k: v if isinstance(v, (bool, int, float, str)) else str(v)
        for k, v in meta.items()
    }

def index_chunks(
    chunks: list[CodeChunk],
    texts: list[str] | None = None,
    db_path: str | Any = DB_PATH,
    collection_name: str = COLLECTION_NAME,
    model: SentenceTransformer | None = None,
    batch_size: int = 128,
    reset: bool = False,
) -> IndexStats:
    started_at = time.perf_counter()

    if not chunks:
        return {"chunks_indexed": 0, "files_indexed": 0, "time_seconds": 0.0}

    texts = texts or [chunk_to_text(chunk) for chunk in chunks]

    if len(texts) != len(chunks):
        raise ValueError()

    model = model or load_model()
    collection = get_collection(
        db_path=db_path, collection_name=collection_name, reset=reset
    )

    embeddings = encode_texts(model=model, texts=texts, show_progress_bar=True)

    ids = [get_chunk_id(chunk) for chunk in chunks]
    documents = [str(chunk.get("source_code") or text) for chunk, text in zip(chunks, texts)]
    metadatas = [build_metadata(chunk) for chunk in chunks]

    embeddings_list = embeddings.tolist()

    for start in range(0, len(chunks), batch_size):
        end = start + batch_size

        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings_list[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    tokenized_corpus = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"model": bm25, "ids": ids}, f)

    files_indexed = len(
        {chunk["file_path"] for chunk in chunks if "file_path" in chunk}
    )

    return {
        "chunks_indexed": len(chunks),
        "files_indexed": files_indexed,
        "time_seconds": round(time.perf_counter() - started_at, 2),
    }