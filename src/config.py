"""Единый файл для подключения путей, модулей и гиперпараметров."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DB_PATH = ROOT / "chroma_db"

MODEL_NAME = "intfloat/multilingual-e5-small"

COLLECTION_NAME = "code_chunks"

TOP_K = 5

RANDOM_STATE = 42