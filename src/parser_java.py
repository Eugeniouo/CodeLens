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

def extract_javadoc(node, source_code: bytes) -> str:
    """Извлекает Javadoc комментарий для узла."""
    prev_sibling = node.prev_sibling
    
    while prev_sibling:
        if prev_sibling.type == "block_comment":
            comment_text = prev_sibling.text.decode("utf-8")
            if comment_text.strip().startswith("/**"):
                return clean_javadoc(comment_text)
        elif prev_sibling.type not in ["whitespace", "line_comment"]:
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
    
    if repo_root:
        rel_path = str(Path(file_path).relative_to(repo_root)).replace("\\", "/")
    else:
        rel_path = file_path
    
    def visit(node, parent_class=None):
        if node.type == "class_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                class_name = name_node.text.decode("utf-8")
                javadoc = extract_javadoc(node, source_code)
                
                chunk_id = f"{rel_path}:{class_name}:{node.start_point[0] + 1}"
                
                chunk: ChunkDict = {
                    "id": chunk_id,
                    "type": "class",
                    "name": class_name,
                    "file_path": rel_path,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "docstring": javadoc,
                    "source_code": source_code[node.start_byte:node.end_byte].decode("utf-8", errors="replace"),
                }
                chunks.append(chunk)
                
                for child in node.children:
                    visit(child, parent_class=class_name)
        
        elif node.type in ["method_declaration", "constructor_declaration"]:
            name_node = node.child_by_field_name("name")
            if name_node:
                method_name = name_node.text.decode("utf-8")
                javadoc = extract_javadoc(node, source_code)
                
                if parent_class:
                    chunk_id = f"{rel_path}:{parent_class}.{method_name}:{node.start_point[0] + 1}"
                    name = f"{parent_class}.{method_name}"
                else:
                    chunk_id = f"{rel_path}:{method_name}:{node.start_point[0] + 1}"
                    name = method_name
                
                chunk: ChunkDict = {
                    "id": chunk_id,
                    "type": "function",
                    "name": name,
                    "file_path": rel_path,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "docstring": javadoc,
                    "source_code": source_code[node.start_byte:node.end_byte].decode("utf-8", errors="replace"),
                }
                chunks.append(chunk)
        
        for child in node.children:
            visit(child, parent_class)
    
    visit(tree.root_node)
    return chunks


def parse_java_directory(directory_path: str) -> list[ChunkDict]:
    """Рекурсивно обходит директорию и парсит все .java файлы."""
    all_chunks: list[ChunkDict] = []
    repo_root = Path(directory_path).resolve()
    
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        
        for file in files:
            if file.endswith(".java"):
                file_path = os.path.join(root, file)
                chunks = parse_java_file(file_path, str(repo_root))
                all_chunks.extend(chunks)
    
    return all_chunks


if __name__ == "__main__":
    import json
    
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    chunks = parse_java_directory(test_dir)
    
    print(f"Найдено {len(chunks)} чанков")
    
    if chunks:
        print("\nПримеры чанков:")
        for i, chunk in enumerate(chunks[:5], 1):
            print(f"\n{i}. {chunk['id']}")
            print(f"   Тип: {chunk['type']}")
            print(f"   Имя: {chunk['name']}")
            print(f"   Строки: {chunk['start_line']}-{chunk['end_line']}")
            docstring = chunk["docstring"]
            if docstring:
                print(f"   Docstring: {docstring[:50]}{'...' if len(docstring) > 50 else ''}")
            else:
                print(f"   Docstring: <нет Javadoc>")
        
        with open("sample_java_chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"\nВсе чанки сохранены в sample_java_chunks.json")