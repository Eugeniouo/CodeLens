"""
Модуль для парсинга Java-файлов через tree-sitter.
Извлекает классы и методы в том же формате, что и parser.py для Python.
"""

import os
import re
import sys
from pathlib import Path
from tree_sitter import Language, Parser

import tree_sitter_java as tsjava

JAVA_LANGUAGE = Language(tsjava.language())

type ChunkDict = dict[str, str | int]

ANNOTATION_TYPES = {
    "annotation",
    "marker_annotation",
    "normal_annotation",
}

# Типы узлов-контейнеров 
CONTAINER_TYPES = {
    "class_declaration",
    "interface_declaration",
    "enum_declaration",
    "record_declaration",
}

SKIP_DIRS = {
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

def extract_javadoc(node, source_code: bytes) -> str:
    """
    Извлекает Javadoc комментарий для узла.
    Корректно обрабатывает аннотации между Javadoc и объявлением.
    """
    prev_sibling = node.prev_sibling
    
    while prev_sibling:
        node_type = prev_sibling.type

        # Нашли блок-комментарий — проверяем, что это Javadoc
        if node_type == "block_comment":
            comment_text = prev_sibling.text.decode("utf-8")
            if comment_text.strip().startswith("/**"):
                return clean_javadoc(comment_text)

        elif node_type in ANNOTATION_TYPES:
            pass

        elif node_type in {"whitespace", "line_comment", "comment"}:
            pass

        else:
            break

        prev_sibling = prev_sibling.prev_sibling

    return ""


def clean_javadoc(comment: str) -> str:
    """Очищает Javadoc от маркеров."""
    text = re.sub(r"/\*\*", "", comment)
    text = re.sub(r"\*/", "", text)
    
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = re.sub(r"^\s*\*\s?", "", line)
        if line.strip():
            cleaned_lines.append(line.strip())
    
    return " ".join(cleaned_lines)

def build_class_skeleton(node, source_code: bytes) -> str:
    """
    Строит скелет класса/интерфейса/энама/рекорда.
    Содержит: заголовок, Javadoc, поля, конструкторы, сигнатуры методов.
    """
    lines: list[str] = []

    # Аннотации
    for child in node.children:
        if child.type in ANNOTATION_TYPES:
            lines.append(child.text.decode("utf-8"))

    # Заголовок
    name_node = node.child_by_field_name("name")
    if not name_node:
        return ""
    class_name = name_node.text.decode("utf-8")

    # Определяем тип контейнера
    if node.type == "interface_declaration":
        lines.append(f"interface {class_name} {{")
    elif node.type == "enum_declaration":
        lines.append(f"enum {class_name} {{")
    elif node.type == "record_declaration":
        # Для рекорда сохраняем параметры
        params_node = node.child_by_field_name("parameters")
        params = params_node.text.decode("utf-8") if params_node else "()"
        lines.append(f"record {class_name}{params} {{")
    else:
        # class_declaration — сохраняем superclass и implements
        superclass = node.child_by_field_name("superclass")
        interfaces = node.child_by_field_name("interfaces")
        header = f"class {class_name}"
        if superclass:
            header += f" extends {superclass.text.decode('utf-8')}"
        if interfaces:
            header += f" implements {interfaces.text.decode('utf-8')}"
        lines.append(f"{header} {{")

    # Javadoc
    javadoc = extract_javadoc(node, source_code)
    if javadoc:
        lines.append(f"    /** {javadoc} */")

    # Тело: поля, конструкторы, методы
    body = node.child_by_field_name("body")
    if not body:
        lines.append("}")
        return "\n".join(lines)

    for child in body.children:
        if child.type == "field_declaration":
            lines.append(f"    {child.text.decode('utf-8')}")

        elif child.type == "constructor_declaration":
            name = child.child_by_field_name("name")
            if name:
                params = child.child_by_field_name("parameters")
                params_text = params.text.decode("utf-8") if params else "()"
                lines.append(f"    {class_name}({params_text}) {{ ... }}")

        elif child.type == "method_declaration":
            name = child.child_by_field_name("name")
            if name:
                method_name = name.text.decode("utf-8")
                params = child.child_by_field_name("parameters")
                params_text = params.text.decode("utf-8") if params else "()"

                return_type = child.child_by_field_name("type")
                return_text = return_type.text.decode("utf-8") if return_type else "void"
                lines.append(f"    {return_text} {method_name}({params_text}) {{ ... }}")

        elif child.type in CONTAINER_TYPES:
            # Вложенный класс/интерфейс/enum
            nested_name = child.child_by_field_name("name")
            if nested_name:
                container_type = child.type.replace("_declaration", "")
                lines.append(f"    {container_type} {nested_name.text.decode('utf-8')} {{ ... }}")

    lines.append("}")
    return "\n".join(lines)

def parse_java_file(file_path: str, repo_root: str | None = None) -> list[ChunkDict]:
    """Парсит Java-файл и извлекает классы и методы."""
    chunks: list[ChunkDict] = []
    
    try:
        with open(file_path, "rb") as f:
            source_code = f.read()
    except IOError as e:
        print(f"️  Ошибка чтения файла {file_path}: {e}")
        return chunks
    
    parser = Parser(JAVA_LANGUAGE)
    tree = parser.parse(source_code)

    rel_path = get_relative_path(file_path, repo_root)
    
    def visit(node, container_stack: list[str] | None = None):
        """
        Рекурсивный обход без дублирования.
        container_stack — стек имён контейнеров (классов/интерфейсов/энамов/рекордов).
        """
        if container_stack is None:
            container_stack = []

        # Обрабатываем контейнеры (class, interface, enum, record)
        if node.type in CONTAINER_TYPES:
            name_node = node.child_by_field_name("name")
            if name_node:
                container_name = name_node.text.decode("utf-8")
                new_stack = container_stack + [container_name]
                full_name = ".".join(new_stack)

                # Чанк контейнера — скелет
                chunk_id = f"{rel_path}:{full_name}:{node.start_point[0] + 1}"
                chunks.append({
                    "id": chunk_id,
                    "type": "class",
                    "name": full_name,
                    "file_path": rel_path,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "docstring": extract_javadoc(node, source_code),
                    "source_code": build_class_skeleton(node, source_code),
                })

                body = node.child_by_field_name("body")
                if body:
                    for child in body.children:
                        visit(child, new_stack)

            return

        # Обрабатываем методы и конструкторы
        elif node.type in {"method_declaration", "constructor_declaration"}:
            name_node = node.child_by_field_name("name")
            if name_node:
                method_name = name_node.text.decode("utf-8")

                if container_stack:
                    full_name = ".".join(container_stack) + "." + method_name
                else:
                    full_name = method_name

                chunk_id = f"{rel_path}:{full_name}:{node.start_point[0] + 1}"
                chunks.append({
                    "id": chunk_id,
                    "type": "function",
                    "name": full_name,
                    "file_path": rel_path,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "docstring": extract_javadoc(node, source_code),
                    "source_code": source_code[node.start_byte:node.end_byte].decode("utf-8", errors="replace"),
                })

            return  

        for child in node.children:
            visit(child, container_stack)

    visit(tree.root_node)
    return chunks


def parse_java_directory(directory_path: str) -> list[ChunkDict]:
    """Рекурсивно обходит директорию и парсит все .java файлы."""
    all_chunks: list[ChunkDict] = []
    repo_root = Path(directory_path).resolve()
    
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS]
        
        for file in files:
            if file.endswith(".java"):
                file_path = os.path.join(root, file)
                chunks = parse_java_file(file_path, str(repo_root))
                all_chunks.extend(chunks)
    
    return all_chunks