"""
Модуль для парсинга и преобразования кодовой базы в чанки, хранящие функции и классы.
Поддерживает Python (через ast) и Java (через tree-sitter).
"""

import ast
import os
from pathlib import Path

_JAVA_PARSER_AVAILABLE = False
try:
    from .parser_java import parse_java_file as _parse_java_file
    _JAVA_PARSER_AVAILABLE = True
except ImportError:
    _parse_java_file = None

type ChunkDict = dict[str, str | int]

SKIP_DIRS = {
    "venv",
    "__pycache__",
    "node_modules",
    "build",
    "dist",
    "target",
}

def get_relative_path(
    file_path: str,
    repo_root: str | None,
) -> str:
    """
    Безопасно получает относительный путь.
    Обрабатывает несовпадение регистров и путей на Windows.
    """

    if repo_root is None:
        return str(file_path)

    try:
        return str(
            Path(file_path)
            .resolve()
            .relative_to(Path(repo_root).resolve())
        ).replace("\\", "/")

    except (ValueError, RuntimeError):
        return str(Path(file_path)).replace("\\", "/")

def build_class_skeleton(node: ast.ClassDef) -> str:
    """
    Строит облегчённое представление класса.

    Сохраняем:
    - заголовок класса
    - docstring
    - поля класса
    - __init__
    - сигнатуры остальных методов
    """
    lines: list[str] = []

    for decorator in node.decorator_list:
        lines.append(f"@{ast.unparse(decorator)}")

    bases = ", ".join(ast.unparse(base) for base in node.bases)

    if bases:
        lines.append(f"class {node.name}({bases}):")
    else:
        lines.append(f"class {node.name}:")

    docstring = ast.get_docstring(node)

    if docstring:
        lines.append(f'"""{docstring}"""')

    for child in node.body:
        # class variables
        if isinstance(child, (ast.Assign, ast.AnnAssign)):
            lines.append(f"{ast.unparse(child)}")

        # вложенные классы
        elif isinstance(child, ast.ClassDef):
            lines.append(f"class {child.name}: ...")

        # constructor сохраняем полностью
        elif isinstance(
            child, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and child.name == "__init__":

            init_code = ast.unparse(child)

            for line in init_code.splitlines():
                lines.append(f"{line}")

        # остальные методы - только сигнатуры
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):

            is_async = isinstance(child, ast.AsyncFunctionDef)
            args = ast.unparse(child.args)
            returns = ""

            if child.returns:
                returns = f" -> {ast.unparse(child.returns)}"

            prefix = "async def" if is_async else "def"

            lines.append(f"{prefix}{child.name}({args}){returns}:")
            lines.append("...")

    if len(lines) == 1:
        lines.append("pass")

    return "\n".join(lines)

class ChunkCollector(ast.NodeVisitor):
    """
    Обходчик AST с поддержкой вложенных классов и функций.
    Использует стеки для формирования полных имён (Class.NestedClass.method).
    """

    def __init__(
        self,
        source_code: str,
        rel_path: str,
    ):
        self.source_code = source_code
        self.rel_path = rel_path

        self.class_stack: list[str] = []
        self.function_stack: list[str] = []

        self.chunks: list[ChunkDict] = []

    def _build_name(self, current_name: str) -> str:

        parts: list[str] = []

        if self.class_stack:
            parts.extend(self.class_stack)

        if self.function_stack:
            parts.extend(self.function_stack)

        parts.append(current_name)

        return ".".join(parts)

    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_stack.append(node.name)
        full_name = ".".join(self.class_stack)

        self.chunks.append({
            "id": f"{self.rel_path}:{full_name}:{node.lineno}",
            "type": "class",
            "name": full_name,
            "file_path": self.rel_path,
            "start_line": node.lineno,
            "end_line": getattr(node, "end_lineno", node.lineno),
            "docstring": ast.get_docstring(node) or "",
            "source_code": build_class_skeleton(node),
        })

        # обходит детей 
        self.generic_visit(node)

        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        full_name = self._build_name(node.name)

        self.chunks.append({
            "id": f"{self.rel_path}:{full_name}:{node.lineno}",
            "type": "function",
            "name": full_name,
            "file_path": self.rel_path,
            "start_line": node.lineno,
            "end_line": getattr(node, "end_lineno", node.lineno),
            "docstring": ast.get_docstring(node) or "",
            "source_code": (
                ast.get_source_segment(self.source_code, node) or ""
            ),
        })

        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ):
        self.visit_FunctionDef(node)

def parse_file(file_path: str, repo_root: str | None = None) -> list[ChunkDict]:
    """
    Универсальный парсер: выбирает парсер по расширению файла.
    
    Args:
        file_path: Путь к файлу (.py или .java)
        repo_root: Корневая директория репозитория
        
    Returns:
        Список словарей с информацией о каждом фрагменте
    """
    if file_path.endswith(".java"):
        if not _JAVA_PARSER_AVAILABLE:
            print(f"️  Java-парсер недоступен (не установлены tree-sitter/tree-sitter-java). Пропускаем {file_path}")
            return []
        return _parse_java_file(file_path, repo_root)
    
    elif file_path.endswith(".py"):
        return parse_python_file(file_path, repo_root)
    
    else:
        print(f"Неподдерживаемый формат файла: {file_path}")
        return []

def parse_python_file(file_path: str, repo_root: str | None = None) -> list[ChunkDict]:
    """
    Парсит один Python-файл и извлекает все функции и классы.
    
    Args:
        file_path: Путь к .py файлу
        repo_root: Корневая директория репозитория 
        
    Returns:
        Список словарей с информацией о каждом фрагменте кода
    """
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()
    except (UnicodeDecodeError, IOError) as e:
        print(f"Ошибка чтения файла {file_path}: {e}")
        return []
    
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        print(f"Синтаксическая ошибка в {file_path}: {e}")
        return []
    
    rel_path = get_relative_path(file_path, repo_root)

    collector = ChunkCollector(source_code, rel_path)
    collector.visit(tree)

    return collector.chunks


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
            if not d.startswith(".") and d not in SKIP_DIRS
        ]
        
        for file in files:
            if file.endswith((".py", ".java")):
                file_path = os.path.join(root, file)
                chunks = parse_file(file_path, str(repo_root))
                all_chunks.extend(chunks)
    
    return all_chunks