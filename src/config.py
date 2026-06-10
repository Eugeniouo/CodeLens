"""Единый файл для подключения путей, модулей и гиперпараметров."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHROMA_PATH = ROOT / "chroma_db"

MODEL_NAME = "intfloat/multilingual-e5-small"

TOP_K = 5

RANDOM_STATE = 42