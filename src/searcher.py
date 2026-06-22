"""Модуль для поиска по индексированной кодовой базе через ChromaDB."""

import argparse
import time
import pickle
from functools import lru_cache
from typing import Any

import chromadb

from .config import COLLECTION_NAME, DB_PATH, TOP_K, BM25_INDEX_PATH, HYBRID_ALPHA, RRF_K
from .embedder import encode_query, load_model
from .indexer import tokenize

type SearchResult = dict[str, Any]

@lru_cache(maxsize=1)
def init_searcher():
    model = load_model()
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)
    
    bm25_data = {"model": None, "ids": []}
    if BM25_INDEX_PATH.exists():
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25_data = pickle.load(f)
            
    return model, collection, bm25_data

def search(query: str, top_k: int = TOP_K, alpha: float = HYBRID_ALPHA) -> tuple[list[SearchResult], float]:
    start_time = time.perf_counter()
    
    query = query.strip()
    if not query or top_k <= 0:
        raise ValueError()

    model, collection, bm25_data = init_searcher()
    query_embedding = encode_query(model, query).tolist()

    n_results = min(RRF_K, collection.count())
    if n_results == 0:
        return [], time.perf_counter() - start_time

    v_res = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["metadatas", "documents"]
    )
    
    v_ids = v_res["ids"][0] if v_res["ids"] else []
    v_docs = v_res["documents"][0] if v_res["documents"] else []
    v_metas = v_res["metadatas"][0] if v_res["metadatas"] else []

    v_rank = {id_: rank for rank, id_ in enumerate(v_ids, 1)}
    id_to_doc = {id_: doc for id_, doc in zip(v_ids, v_docs)}
    id_to_meta = {id_: meta for id_, meta in zip(v_ids, v_metas)}

    l_rank = {}
    if bm25_data["model"] and bm25_data["ids"]:
        bm25_model = bm25_data["model"]
        bm25_ids = bm25_data["ids"]
        tokenized_query = tokenize(query)
        
        if tokenized_query:
            l_scores = bm25_model.get_scores(tokenized_query)
            l_scored_ids = sorted(zip(bm25_ids, l_scores), key=lambda x: x[1], reverse=True)[:n_results]
            l_rank = {id_: rank for rank, (id_, score) in enumerate(l_scored_ids, 1) if score > 0}

    missing_ids = [id_ for id_ in l_rank if id_ not in id_to_doc]
    if missing_ids:
        missing_res = collection.get(ids=missing_ids, include=["metadatas", "documents"])
        if missing_res and missing_res["ids"]:
            for id_, doc, meta in zip(missing_res["ids"], missing_res["documents"], missing_res["metadatas"]):
                id_to_doc[id_] = doc
                id_to_meta[id_] = meta

    all_ids = set(v_rank.keys()) | set(l_rank.keys())
    rrf_scores = []

    for id_ in all_ids:
        score_v = 1.0 / (RRF_K + v_rank[id_]) if id_ in v_rank else 0.0
        score_l = 1.0 / (RRF_K + l_rank[id_]) if id_ in l_rank else 0.0
        rrf_score = alpha * score_v + (1.0 - alpha) * score_l
        rrf_scores.append((rrf_score, id_))

    rrf_scores.sort(key=lambda x: x[0], reverse=True)
    top_results = rrf_scores[:top_k]

    formatted_results = [
        {
            "id": id_,
            "rank": rank,
            "score": score,
            "name": id_to_meta[id_].get("name", "unknown"),
            "file_path": id_to_meta[id_].get("file_path", "unknown"),
            "source_code": id_to_doc[id_]
        }
        for rank, (score, id_) in enumerate(top_results, 1)
    ]

    latency = time.perf_counter() - start_time
    return formatted_results, latency

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="+")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--alpha", type=float, default=HYBRID_ALPHA)
    args = parser.parse_args()

    query = " ".join(args.query)
    
    results, latency = search(query=query, top_k=args.top_k, alpha=args.alpha)
    
    for result in results:
        print(f"Rank: {result['rank']} | Score: {result['score']:.4f} | ID: {result['id']}")
        print(f"File: {result['file_path']} | Name: {result['name']}")
        print("-" * 80)
        
        lines = result["source_code"].splitlines()
        print("\n".join(lines[:12]) + ("\n..." if len(lines) > 12 else "") + "\n")

if __name__ == "__main__":
    main()