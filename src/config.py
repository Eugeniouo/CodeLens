"""Единый файл для подключения путей, модулей и гиперпараметров."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DB_PATH = ROOT / "chroma_db"

BM25_INDEX_PATH = ROOT / "bm25_index.pkl"

MODEL_NAME = "intfloat/multilingual-e5-small"

COLLECTION_NAME = "code_chunks"

TOP_K = 5

HYBRID_ALPHA = 0.5

RRF_K = 60

RANDOM_STATE = 42