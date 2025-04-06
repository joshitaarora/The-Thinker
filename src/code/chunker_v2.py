# -*- coding: utf-8 -*-
# command to install required package
# !pip install tree-sitter-language-pack

from __future__ import annotations
import re
import requests
from dataclasses import dataclass, field
from typing import List, Union
from tree_sitter import Node, Tree
from tree_sitter_language_pack import get_binding, get_language, get_parser

# Tree-sitter setup
python_binding = get_binding('python')
python_lang = get_language('python')
python_parser = get_parser('python')


@dataclass
class Span:
    start: int = 0
    end: int = 0

    def __post_init__(self):
        if self.end is None:
            self.end = self.start

    def extract(self, s: str) -> str:
        return s[self.start:self.end]

    def extract_lines(self, s: str) -> str:
        return "\n".join(s.splitlines()[self.start:self.end])

    def __add__(self, other: Union[Span, int]) -> Span:
        if isinstance(other, int):
            return Span(self.start + other, self.end + other)
        elif isinstance(other, Span):
            return Span(self.start, other.end)
        else:
            raise NotImplementedError()

    def __len__(self) -> int:
        return self.end - self.start


def convert_to_raw_url(url: str) -> str:
    if "github.com" in url and "/blob/" in url:
        url_parts = url.split('/')
        url_parts[2] = 'raw.githubusercontent.com'
        url_parts.remove('blob')
        return '/'.join(url_parts)
    return url


def pretty_node(node: Node) -> str:
    return f"{node.type}:{node.start_byte}-{node.end_byte}"


def print_tree(node: Node, indent: str = ""):
    if len(re.sub("\s", "", node.text.decode("utf-8"))) < 100:
        return
    print(indent + pretty_node(node))
    for child in node.children:
        print_tree(child, indent + "  ")


def char_len(s: str) -> int:
    return len(s)


def non_whitespace_len(s: str) -> int:
    return len(re.sub("\s", "", s))


def get_line_number(index: int, source_code: str) -> int:
    total_chars = 0
    for line_number, line in enumerate(source_code.splitlines(keepends=True), start=1):
        total_chars += len(line)
        if total_chars > index:
            return line_number - 1
    return line_number


def chunker(
    tree: Tree,
    source_code: str,
    MAX_CHARS: int = 512 * 3,
    coalesce: int = 50,
) -> List[Span]:

    def chunk_node(node: Node) -> List[Span]:
        chunks: List[Span] = []
        current_chunk = Span(node.start_byte, node.start_byte)
        for child in node.children:
            size = child.end_byte - child.start_byte
            current_length = len(current_chunk)
            if size > MAX_CHARS:
                chunks.append(current_chunk)
                current_chunk = Span(child.end_byte, child.end_byte)
                chunks.extend(chunk_node(child))
            elif size + current_length > MAX_CHARS:
                chunks.append(current_chunk)
                current_chunk = Span(child.start_byte, child.end_byte)
            else:
                current_chunk += Span(child.start_byte, child.end_byte)
        chunks.append(current_chunk)
        return chunks

    # Step 1: Initial chunking
    chunks = chunk_node(tree.root_node)

    # Step 2: Fill in the gaps
    for prev, curr in zip(chunks[:-1], chunks[1:]):
        prev.end = curr.start

    # Step 3: Coalesce small chunks
    new_chunks = []
    current_chunk = Span(0, 0)
    for chunk in chunks:
        current_chunk += chunk
        if non_whitespace_len(current_chunk.extract(source_code)) > coalesce and "\n" in current_chunk.extract(source_code):
            new_chunks.append(current_chunk)
            current_chunk = Span(chunk.end, chunk.end)
    if len(current_chunk) > 0:
        new_chunks.append(current_chunk)

    # Step 4: Convert to line numbers
    line_chunks = [
        Span(get_line_number(chunk.start, source_code), get_line_number(chunk.end, source_code))
        for chunk in new_chunks if len(chunk) > 0
    ]

    return line_chunks


@dataclass
class MockNode:
    start_byte: int = 0
    end_byte: int = 0
    children: List[MockNode] = field(default_factory=list)


# Example usage
if __name__ == "__main__":
    original_url = "https://github.com/doocs/leetcode/blob/main/solution/0000-0099/0001.Two%20Sum/Solution.py"
    python_code = requests.get(convert_to_raw_url(original_url)).text
    tree = python_parser.parse(python_code.encode("utf-8"))

    for chunk in chunker(tree, python_code):
        print(chunk.extract_lines(python_code) + "\n\n====================\n\n")

    for i, chunk in enumerate(chunker(tree, python_code)):
        print(f"Chunk {i}: {get_line_number(chunk.start, python_code)}-{get_line_number(chunk.end, python_code)}")
