"""Точка входа. Индексация кодовой базы"""

import argparse
import time
from pathlib import Path

from src.indexer import COLLECTION_NAME, DB_PATH, index_chunks
from src.embedder import load_model
from src.parser import SKIP_DIRS, parse_directory

SUPPORTED_EXTENSIONS = {".py", ".java"}


def count_source_files(directory_path: Path) -> int:
    """
    Считает файлы для обработки, используя всю мощь pathlib.
    Изящно игнорирует скрытые папки и директории из SKIP_DIRS.
    """
    total = 0
    for path in directory_path.rglob("*"):
        if path.is_file() and path.suffix in SUPPORTED_EXTENSIONS:
            if not any(part.startswith('.') or part in SKIP_DIRS for part in path.parts):
                total += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Индексация кодовой базы в ChromaDB.")
    parser.add_argument("directory", help="Путь к директории с кодовой базой.")
    parser.add_argument("--db-path", default=DB_PATH, help=f"Путь к ChromaDB ({DB_PATH})")
    parser.add_argument("--collection", default=COLLECTION_NAME, help=f"Коллекция ({COLLECTION_NAME})")
    parser.add_argument("--batch-size", type=int, default=128, help="Размер батча для БД")
    parser.add_argument("--reset", action="store_true", help="Удалить старую коллекцию")

    args = parser.parse_args()
    directory_path = Path(args.directory).resolve()

    if not directory_path.is_dir():
        raise NotADirectoryError(f"Директория не найдена: {directory_path}")

    total_started_at = time.perf_counter()

    print("=" * 80)
    print(f"Индексация кодовой базы: {directory_path}")
    print("=" * 80)

    files_count = count_source_files(directory_path)
    print(f"Файлов для возможной обработки: {files_count}")

    print("\n[1/2] Парсинг директории...")
    parse_started_at = time.perf_counter()
    
    chunks = parse_directory(str(directory_path))
    parse_time = round(time.perf_counter() - parse_started_at, 2)

    if not chunks:
        print("\nЧанки не найдены. Индексировать нечего.")
        return

    print(f"Парсинг завершен за {parse_time} сек. Создано чанков: {len(chunks)}")

    print("\n[2/2] Загрузка модели и сохранение эмбеддингов в БД...")
    
    stats = index_chunks(
        chunks=chunks,
        db_path=args.db_path,
        collection_name=args.collection,
        model=load_model(),
        batch_size=args.batch_size,
        reset=args.reset,
    )

    total_time = round(time.perf_counter() - total_started_at, 2)

    print("\n" + "=" * 80)
    print("Индексация завершена успешно!")
    print("=" * 80)
    print(f"Файлов с чанками:       {stats['files_indexed']} / {files_count}")
    print(f"Чанков сохранено в БД:  {stats['chunks_indexed']} / {len(chunks)}")
    print(f"Общее время:            {total_time} сек.")
    print("=" * 80)


if __name__ == "__main__":
    main()