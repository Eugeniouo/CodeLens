"""Модуль для парсинга и преобразования кодовой базы в чанки, хранящие функции и классы."""

import ast
import os
from pathlib import Path

type ChunkDict = dict[str, str | int]

def parse_file(file_path: str, repo_root: str | None = None) -> list[ChunkDict]:
    """
    Парсит один Python-файл и извлекает все функции и классы.
    
    Args:
        file_path: Путь к .py файлу
        repo_root: Корневая директория репозитория 
        
    Returns:
        Список словарей с информацией о каждом фрагменте кода
    """

    chunks: list[ChunkDict] = []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()
    except (UnicodeDecodeError, IOError) as e:
        print(f"Ошибка чтения файла {file_path}: {e}")
        return chunks
    
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        print(f"Синтаксическая ошибка в {file_path}: {e}")
        return chunks
    
    if repo_root:
        rel_path = str(Path(file_path).relative_to(repo_root)).replace("\\", "/")
    else:
        rel_path = file_path
    
    # Обход с отслеживанием родителя
    def visit(node, parent_class=None):
        if isinstance(node, ast.ClassDef):
            # Добавляем сам класс как чанк
            chunk_id = f"{rel_path}:{node.name}:{node.lineno}"
            chunks.append({
                "id": chunk_id,
                "type": "class",
                "name": node.name,
                "file_path": rel_path,
                "start_line": node.lineno,
                "end_line": getattr(node, "end_lineno", node.lineno),
                "docstring": ast.get_docstring(node) or "",
                "source_code": ast.get_source_segment(source_code, node) or "",
            })
            # Рекурсивно обходим тело класса
            for child in ast.iter_child_nodes(node):
                visit(child, parent_class=node.name)
                
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if parent_class:
                # Метод класса
                chunk_id = f"{rel_path}:{parent_class}.{node.name}:{node.lineno}"
                name = f"{parent_class}.{node.name}"
            else:
                # Обычная функция
                chunk_id = f"{rel_path}:{node.name}:{node.lineno}"
                name = node.name
            
            chunks.append({
                "id": chunk_id,
                "type": "function",
                "name": name,
                "file_path": rel_path,
                "start_line": node.lineno,
                "end_line": getattr(node, "end_lineno", node.lineno),
                "docstring": ast.get_docstring(node) or "",
                "source_code": ast.get_source_segment(source_code, node) or "",
            })
        else:
            for child in ast.iter_child_nodes(node):
                visit(child, parent_class)
    
    visit(tree)
    return chunks


def parse_directory(directory_path: str) -> list[ChunkDict]:
    """
    Рекурсивно обходит директорию и парсит все .py файлы.
    
    Args:
        directory_path: Путь к директории с Python-кодом
        
    Returns:
        Список всех чанков из всех файлов
    """
    all_chunks: list[ChunkDict] = []
    repo_root = Path(directory_path).resolve()
    
    for root, dirs, files in os.walk(repo_root):
        # Пропускаем скрытые директории и виртуальные окружения
        dirs[:] = [
            d for d in dirs 
            if not d.startswith(".") and d not in {"venv", "__pycache__", "node_modules"}
        ]
        
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                chunks = parse_file(file_path, str(repo_root))
                all_chunks.extend(chunks)
    
    return all_chunks
