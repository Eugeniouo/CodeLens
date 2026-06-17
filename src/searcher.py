"""Модуль для поиска по индексированной кодовой базе через ChromaDB."""

import argparse
import time
from functools import lru_cache
from typing import Any

import chromadb

from .config import COLLECTION_NAME, DB_PATH, TOP_K
from .embedder import encode_query, load_model

type SearchResult = dict[str, Any]


@lru_cache(maxsize=1)
def init_searcher():
    """Кэшированная загрузка модели и БД для быстрого UI в Streamlit."""
    model = load_model()
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return model, client.get_collection(name=COLLECTION_NAME)


def search(query: str, top_k: int = TOP_K) -> tuple[list[SearchResult], float]:
    """Ищет top-k релевантных чанков кода и возвращает результаты + время поиска (Latency)."""
    start_time = time.perf_counter()
    
    query = query.strip()
    if not query or top_k <= 0:
        raise ValueError("Некорректный запрос или значение top_k.")

    model, collection = init_searcher()
    query_embedding = encode_query(model, query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )

    formatted_results = [
        {
            "id": c_id,
            "rank": rank,
            "score": 1.0 - float(dist),
            "name": (meta or {}).get("name", "unknown"),
            "file_path": (meta or {}).get("file_path", (meta or {}).get("path", "unknown")),
            "source_code": doc or "",
        }
        for rank, (c_id, doc, meta, dist) in enumerate(
            zip(results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]),
            start=1
        )
    ]
    
    latency = time.perf_counter() - start_time
    return formatted_results, latency


def main() -> None:
    """CLI для локального тестирования поиска."""
    parser = argparse.ArgumentParser(description="Поиск по ChromaDB.")
    parser.add_argument("query", nargs="+", help="Поисковый запрос")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Кол-во результатов")
    args = parser.parse_args()

    query = " ".join(args.query)
    print(f"\nRequest: {query}")
    
    results, latency = search(query=query, top_k=args.top_k)
    
    print(f"Latency: {latency:.4f} сек.\n" + "=" * 80)

    for result in results:
        print(f"Rank: {result['rank']} | Score: {result['score']:.4f} | ID: {result['id']}")
        print(f"File: {result['file_path']} | Name: {result['name']}")
        print("-" * 80)
        
        lines = result["source_code"].splitlines()
        print("\n".join(lines[:12]) + ("\n..." if len(lines) > 12 else "") + "\n")


if __name__ == "__main__":
    main()