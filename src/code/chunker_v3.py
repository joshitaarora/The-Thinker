from typing import List, Optional

from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.core.text_splitter import CodeSplitter
from llama_index.packs.code_hierarchy import CodeHierarchyNodeParser


def parse_codebase_into_chunks(
    repo_path: str,
    language: str = "python",
    max_chars: int = 1000,
    chunk_lines: int = 10,
) -> List:
    """
    Parses a local code repository into structured, hierarchical code chunks.

    Args:
        repo_path (str): Path to the cloned code repository.
        language (str): Programming language (e.g., 'python').
        max_chars (int): Max characters per chunk.
        chunk_lines (int): Max lines per chunk.

    Returns:
        List[BaseNode]: List of structured code chunks.
    """
    reader = SimpleDirectoryReader(
        input_dir=repo_path,
        recursive=True,
        required_exts=[".py"],
        file_metadata=lambda path: {"filepath": str(path)},
    )

    documents = reader.load_data()
    code_splitter = CodeSplitter(
        language=language, max_chars=max_chars, chunk_lines=chunk_lines
    )
    node_parser = CodeHierarchyNodeParser(
        language=language, code_splitter=code_splitter
    )

    return node_parser.get_nodes_from_documents(documents)
