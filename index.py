"""Точка входа. Индексация кодовой базы"""

import argparse
import time
from pathlib import Path

from src.indexer import COLLECTION_NAME, DB_PATH, index_chunks
from src.embedder import load_model
from src.parser import SKIP_DIRS, parse_directory

SUPPORTED_EXTENSIONS = {".py", ".java"}

def count_source_files(directory_path: Path) -> int:
    total = 0
    for path in directory_path.rglob("*"):
        if path.is_file() and path.suffix in SUPPORTED_EXTENSIONS:
            if not any(part.startswith('.') or part in SKIP_DIRS for part in path.parts):
                total += 1
    return total

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--db-path", default=DB_PATH)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--reset", action="store_true")

    args = parser.parse_args()
    directory_path = Path(args.directory).resolve()

    if not directory_path.is_dir():
        raise NotADirectoryError()

    total_started_at = time.perf_counter()

    files_count = count_source_files(directory_path)

    parse_started_at = time.perf_counter()
    chunks = parse_directory(str(directory_path))
    
    if not chunks:
        return

    stats = index_chunks(
        chunks=chunks,
        db_path=args.db_path,
        collection_name=args.collection,
        model=load_model(),
        batch_size=args.batch_size,
        reset=args.reset,
    )

    total_time = round(time.perf_counter() - total_started_at, 2)
    print(f"Indexed: {stats['chunks_indexed']} chunks in {total_time} s")

if __name__ == "__main__":
    main()