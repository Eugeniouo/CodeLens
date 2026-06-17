"""Точка входа. Индексация кодовой базы"""

import argparse
import os
import time
from pathlib import Path

from src.embedder import chunk_to_text, load_model
from src.indexer import COLLECTION_NAME, DB_PATH, index_chunks
from src.parser import SKIP_DIRS, parse_directory


SUPPORTED_EXTENSIONS = (".py", ".java")


def count_source_files(directory_path: Path) -> int:
    """
    Считает файлы, которые parser.py попробует обработать.

    Логика пропуска директорий совпадает с parser.py.
    """
    total = 0

    for root, dirs, files in os.walk(directory_path):
        dirs[:] = [
            directory
            for directory in dirs
            if not directory.startswith(".") and directory not in SKIP_DIRS
        ]

        for file_name in files:
            if file_name.endswith(SUPPORTED_EXTENSIONS):
                total += 1

    return total


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Индексация кодовой базы в ChromaDB."
    )

    parser.add_argument(
        "directory",
        help="Путь к директории с кодовой базой.",
    )

    parser.add_argument(
        "--db-path",
        default=DB_PATH,
        help=f"Путь к ChromaDB. По умолчанию: {DB_PATH}",
    )

    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help=f"Название коллекции. По умолчанию: {COLLECTION_NAME}",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Размер батча для записи в ChromaDB.",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Удалить старую коллекцию перед индексацией.",
    )

    return parser


def main() -> None:
    args_parser = build_arg_parser()
    args = args_parser.parse_args()

    directory_path = Path(args.directory).resolve()

    if not directory_path.exists():
        raise FileNotFoundError(f"Директория не найдена: {directory_path}")

    if not directory_path.is_dir():
        raise NotADirectoryError(f"Это не директория: {directory_path}")

    total_started_at = time.perf_counter()

    print("=" * 80)
    print("Индексация кодовой базы")
    print("=" * 80)
    print(f"Директория:  {directory_path}")
    print(f"ChromaDB:    {args.db_path}")
    print(f"Коллекция:   {args.collection}")
    print(f"Reset:       {args.reset}")
    print("=" * 80)

    files_count = count_source_files(directory_path)
    print(f"Файлов для обработки: {files_count}")

    print("\n[1/3] Парсим директорию через parse_directory...")
    parse_started_at = time.perf_counter()

    chunks = parse_directory(str(directory_path))

    parse_time = round(time.perf_counter() - parse_started_at, 2)

    files_with_chunks = len(
        {
            str(chunk.get("file_path"))
            for chunk in chunks
            if chunk.get("file_path")
        }
    )

    print(f"Обработка файлов завершена за {parse_time} сек.")
    print(f"Файлов обработано: {files_count}")
    print(f"Файлов с чанками:  {files_with_chunks}")
    print(f"Чанков создано:    {len(chunks)}")

    if not chunks:
        print("\nЧанки не найдены. Индексировать нечего.")
        return

    print("\n[2/3] Преобразуем чанки в текст через chunk_to_text...")
    text_started_at = time.perf_counter()

    texts = [chunk_to_text(chunk) for chunk in chunks]

    text_time = round(time.perf_counter() - text_started_at, 2)

    print(f"Текстов для эмбеддинга создано: {len(texts)}")
    print(f"Время подготовки текстов:       {text_time} сек.")

    print("\n[3/3] Загружаем модель и сохраняем эмбеддинги в ChromaDB...")
    index_started_at = time.perf_counter()

    model = load_model()

    stats = index_chunks(
        chunks=chunks,
        texts=texts,
        db_path=args.db_path,
        collection_name=args.collection,
        model=model,
        batch_size=args.batch_size,
        reset=args.reset,
    )

    index_time = round(time.perf_counter() - index_started_at, 2)
    total_time = round(time.perf_counter() - total_started_at, 2)

    print("\n" + "=" * 80)
    print("Индексация завершена")
    print("=" * 80)
    print(f"Файлов обработано:      {files_count}")
    print(f"Файлов с чанками:       {stats['files_indexed']}")
    print(f"Чанков создано:         {len(chunks)}")
    print(f"Чанков сохранено в БД:  {stats['chunks_indexed']}")
    print(f"Время парсинга:         {parse_time} сек.")
    print(f"Время индексации:       {index_time} сек.")
    print(f"Общее время:            {total_time} сек.")
    print("=" * 80)


if __name__ == "__main__":
    main()