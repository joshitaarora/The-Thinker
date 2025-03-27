# -*- coding: utf-8 -*-
#command to install required package
#!pip install tree-sitter-language-pack

import re

from tree_sitter_language_pack import get_binding, get_language, get_parser

python_binding = get_binding('python')  # this is a pycapsule object pointing to the C binding
python_lang = get_language('python')  # this is an instance of tree_sitter.Language
python_parser = get_parser('python')  # this is an instance of tree_sitter.Parser

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Span:
    # Represents a slice of a string
    start: int = 0
    end: int = 0

    def __post_init__(self):
        # If end is None, set it to start
        if self.end is None:
            self.end = self.start

    def extract(self, s: str) -> str:
        # Grab the corresponding substring of string s by bytes
        return s[self.start : self.end]

    def extract_lines(self, s: str) -> str:
        # Grab the corresponding substring of string s by lines
        return "\n".join(s.splitlines()[self.start : self.end])

    def __add__(self, other: Span | int) -> Span:
        # e.g. Span(1, 2) + Span(2, 4) = Span(1, 4) (concatenation)
        # There are no safety checks: Span(a, b) + Span(c, d) = Span(a, d)
        # and there are no requirements for b = c.
        if isinstance(other, int):
            return Span(self.start + other, self.end + other)
        elif isinstance(other, Span):
            return Span(self.start, other.end)
        else:
            raise NotImplementedError()

    def __len__(self) -> int:
        # i.e. Span(a, b) = b - a
        return self.end - self.start

def convert_to_raw_url(url):
    if "github.com" in url and "/blob/" in url:
        url_parts = url.split('/')
        # Replace 'github.com' with 'raw.githubusercontent.com'
        url_parts[2] = 'raw.githubusercontent.com'
        # Remove the 'blob' part
        url_parts.remove('blob')
        return '/'.join(url_parts)
    return url  # return the original if it's not in the expected format

# Example usage
original_url = "https://github.com/doocs/leetcode/blob/main/solution/0000-0099/0001.Two%20Sum/Solution.py"

import requests

# example_file = "https://raw.githubusercontent.com/sweepai/sweep/b267b613d4c706eaf959fe6789f11e9a856521d1/sweepai/handlers/on_check_suite.py"
#example_file = "https://raw.githubusercontent.com/doocs/leetcode/main/solution/0000-0099/0001.Two%20Sum/Solution.py"
python_code = requests.get(convert_to_raw_url(original_url)).text

tree = python_parser.parse(python_code.encode("utf-8"))


def pretty_node(node):
    return f"{node.type}:{node.start_byte}-{node.end_byte}"


def print_tree(node, indent=""):
    if len(re.sub("\s", "", node.text.decode("utf-8"))) < 100:
        return
    print(indent + pretty_node(node))
    for child in node.children:
        print_tree(child, indent=indent + "  ")


for child in tree.root_node.children:
    print_tree(child)

def connect_chunks(chunks: list[Span]):
    for prev, curr in zip(chunks[:-1], chunks[1:]):
        prev.end = curr.start
    return chunks

from tree_sitter import Node
from dataclasses import field


def chunk_node(node: Node, text: str, MAX_CHARS: int = 600) -> list[str]:
    chunks = []
    current_chunk = ""
    for child in node.children:
        if child.end_byte - child.start_byte > MAX_CHARS:
            chunks.append(current_chunk)
            current_chunk = ""
            chunks.extend(chunk_node(child, text, MAX_CHARS))
        elif child.end_byte - child.start_byte + len(current_chunk) > MAX_CHARS:
            chunks.append(current_chunk)
            current_chunk = text[child.start_byte : child.end_byte]
        else:
            current_chunk += text[child.start_byte : child.end_byte]
    chunks.append(current_chunk)

    return chunks


for chunk in chunk_node(tree.root_node, python_code):
    print(chunk + "\n\n====================\n\n")

@dataclass
class MockNode:
    start_byte: int = 0
    end_byte: int = 0
    children: list[MockNode] = field(default_factory=list)


def chunk_node(node: Node, text: str, MAX_CHARS: int = 600) -> list[str]:
    chunks = []
    current_chunk = ""
    node_children = node.children + [MockNode(node.end_byte, node.end_byte)]

    for child, next_child in zip(node_children[:-1], node_children[1:]):
        if child.end_byte - child.start_byte > MAX_CHARS:
            chunks.append(current_chunk)
            current_chunk = ""
            chunks.extend(chunk_node(child, text, MAX_CHARS))
        elif child.end_byte - child.start_byte + len(current_chunk) > MAX_CHARS:
            chunks.append(current_chunk)
            current_chunk = text[child.start_byte : next_child.start_byte]
        else:
            current_chunk += text[child.start_byte : next_child.start_byte]
    chunks.append(current_chunk)

    return chunks


for chunk in chunk_node(tree.root_node, python_code):
    print(chunk + "\n\n====================\n\n")

def chunk_node(
    node: Node,
    MAX_CHARS: int = 600,
) -> list[Span]:
    chunks: list[Span] = []
    current_chunk: Span = Span(node.start_byte, node.start_byte)
    node_children = node.children
    for child in node_children:
        if child.end_byte - child.start_byte > MAX_CHARS:
            chunks.append(current_chunk)
            current_chunk = Span(child.end_byte, child.end_byte)
            chunks.extend(chunk_node(child, MAX_CHARS))
        elif child.end_byte - child.start_byte + len(current_chunk) > MAX_CHARS:
            chunks.append(current_chunk)
            current_chunk = Span(child.start_byte, child.end_byte)
        else:
            current_chunk += Span(child.start_byte, child.end_byte)
    chunks.append(current_chunk)
    return chunks


for chunk in chunk_node(tree.root_node):
    print(chunk)

def char_len(s: str) -> int:  # old len function
    return len(s)


def non_whitespace_len(s: str) -> int:  # new len function
    return len(re.sub("\s", "", s))

def coalesce_chunks(
    chunks: list[Span], source_code: str, coalesce: int = 50
) -> list[Span]:
    new_chunks = []
    current_chunk = Span(0, 0)
    for chunk in chunks:
        current_chunk += chunk
        if len(current_chunk) > coalesce and "\n" in current_chunk.extract(source_code):
            new_chunks.append(current_chunk)
            current_chunk = Span(chunk.end, chunk.end)
    if len(current_chunk) > 0:
        new_chunks.append(current_chunk)
    return new_chunks


for chunk in coalesce_chunks(chunk_node(tree.root_node), python_code):
    print(chunk.extract(python_code))

def get_line_number(index: int, source_code: str) -> int:
    total_chars = 0
    for line_number, line in enumerate(source_code.splitlines(keepends=True), start=1):
        total_chars += len(line)
        if total_chars > index:
            return line_number - 1
    return line_number


for i, chunk in enumerate(coalesce_chunks(chunk_node(tree.root_node), python_code)):
    print(
        f"Chunk {i}: {get_line_number(chunk.start, python_code)}-{get_line_number(chunk.end, python_code)}"
    )

from tree_sitter import Tree


def chunker(
    tree: Tree,
    source_code: bytes,
    MAX_CHARS=512 * 3,
    coalesce=50,  # Any chunk less than 50 characters long gets coalesced with the next chunk
) -> list[Span]:

    # 1. Recursively form chunks based on the last post (https://docs.sweep.dev/blogs/chunking-2m-files)
    def chunk_node(node: Node) -> list[Span]:
        chunks: list[Span] = []
        current_chunk: Span = Span(node.start_byte, node.start_byte)
        node_children = node.children
        for child in node_children:
            if child.end_byte - child.start_byte > MAX_CHARS:
                chunks.append(current_chunk)
                current_chunk = Span(child.end_byte, child.end_byte)
                chunks.extend(chunk_node(child))
            elif child.end_byte - child.start_byte + len(current_chunk) > MAX_CHARS:
                chunks.append(current_chunk)
                current_chunk = Span(child.start_byte, child.end_byte)
            else:
                current_chunk += Span(child.start_byte, child.end_byte)
        chunks.append(current_chunk)
        return chunks

    chunks = chunk_node(tree.root_node)

    # 2. Filling in the gaps
    for prev, curr in zip(chunks[:-1], chunks[1:]):
        prev.end = curr.start

    # 3. Combining small chunks with bigger ones
    new_chunks = []
    current_chunk = Span(0, 0)
    for chunk in chunks:
        current_chunk += chunk
        if non_whitespace_len(
            current_chunk.extract(source_code)
        ) > coalesce and "\n" in current_chunk.extract(source_code):
            new_chunks.append(current_chunk)
            current_chunk = Span(chunk.end, chunk.end)
    if len(current_chunk) > 0:
        new_chunks.append(current_chunk)

    # 4. Changing line numbers
    line_chunks = [
        Span(
            get_line_number(chunk.start, source_code),
            get_line_number(chunk.end, source_code),
        )
        for chunk in new_chunks
    ]

    # 5. Eliminating empty chunks
    line_chunks = [chunk for chunk in line_chunks if len(chunk) > 0]

    return line_chunks


for chunk in chunker(tree, python_code):
    print(chunk.extract_lines(python_code) + "\n\n====================\n\n")

