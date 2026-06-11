"""Модуль для поиска ответа через ChromaDB."""

import sys
import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "code_chunks"


def init_searcher():
    print("Loading models and connecting to database")
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=DB_PATH)
    
    collection = client.get_collection(name=COLLECTION_NAME)
    return model, collection


def search(query: str, top_k: int = 5) -> list[dict]:
    model, collection = init_searcher()

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=['metadatas', 'documents', 'distances']
    )

    formatted_results = []

    if results and results['ids'][0]:
        for i, (chunk_id, doc, metadata, distance) in enumerate(
            zip(results['ids'][0], results['documents'][0], results['metadatas'][0], results['distances'][0])
        ):

            score = 1.0 - distance 
            
            formatted_results.append({
                "id": chunk_id,
                "rank": i + 1,
                "score": score,
                "name": metadata.get("name", "unknown"),
                "file_path": metadata.get("path", "unknown"),
                "source_code": doc
            })
            
    return formatted_results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("Использование: python searcher.py <ваш запрос>")
        print("Пример: python searcher.py как создаётся токен доступа")
        sys.exit(1)
        
    print(f"\nRequest: {query}\n" + "="*60)
    
    results = search(query, top_k=5)
    
    if not results:
        print("Nothing was found")
    else:
        for res in results:
            print(f"Rank: {res['rank']} | Score: {res['score']:.4f} | ID: {res['id']}")
            print(f"File: {res['file_path']} | Name: {res['name']}")
            print("-" * 60)
            code_preview = "\n".join(res['source_code'].splitlines()[:3]) + "..."
            print(f"Код:\n{code_preview}\n")