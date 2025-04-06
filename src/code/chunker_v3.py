import os
import uuid
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import itertools

from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.core.text_splitter import CodeSplitter
from llama_index.packs.code_hierarchy import CodeHierarchyNodeParser
from src.data.data_loader import stream_codesearchnet_data
from src.code.vector_db import init_milvus_collection, get_client
from src.code.chunk_summaries import summarize_code_llama
from src.code.embedder import embed_text


# Language to file extension mapping
LANGUAGE_EXTENSIONS = {
    "python": [".py"],
    "java": [".java"],
    "javascript": [".js", ".jsx"],
    "go": [".go"],
    "ruby": [".rb"],
    "php": [".php"],
}


def parse_codebase_into_chunks(
    repo_path: str, language: str = "python", max_chars: int = 5000, chunk_lines: int = 50
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
    required_exts = LANGUAGE_EXTENSIONS.get(language.lower())
    if not required_exts:
        raise ValueError(f"Unsupported language: {language}")

    reader = SimpleDirectoryReader(
        input_dir=repo_path,
        recursive=True,
        required_exts=required_exts,
        file_metadata=lambda path: {"filepath": str(path)},
    )
    documents = reader.load_data()

    code_splitter = CodeSplitter(language=language, max_chars=max_chars, chunk_lines=chunk_lines)
    node_parser = CodeHierarchyNodeParser(language=language, code_splitter=code_splitter)

    return node_parser.get_nodes_from_documents(documents)

from typing import List
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.schema import BaseNode

# Example language extension mapping
LANGUAGE_EXTENSIONS = {
    "python": [".py"],
    "java": [".java"],
    "javascript": [".js"],
    "typescript": [".ts"],
    "c": [".c"],
    "cpp": [".cpp", ".cc", ".cxx", ".hpp"],
    "go": [".go"],
    # Add more as needed
}

def parse_codebase_into_chunks_naive(
    repo_path: str, language: str = "python", chunk_size: int = 256, chunk_overlap: int = 20
) -> List[BaseNode]:
    """
    Parses a local code repository into structured code chunks using sentence-level chunking.

    Args:
        repo_path (str): Path to the cloned code repository.
        language (str): Programming language (e.g., 'python').
        chunk_size (int): Maximum tokens per chunk.
        chunk_overlap (int): Number of overlapping tokens between chunks.

    Returns:
        List[BaseNode]: List of structured code chunks.
    """
    required_exts = LANGUAGE_EXTENSIONS.get(language.lower())
    if not required_exts:
        raise ValueError(f"Unsupported language: {language}")

    reader = SimpleDirectoryReader(
        input_dir=repo_path,
        recursive=True,
        required_exts=required_exts,
        file_metadata=lambda path: {"filepath": str(path)},
    )
    documents = reader.load_data()

    node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return node_parser.get_nodes_from_documents(documents)




def insert_batch_to_milvus(collection_name, batch):
    client = get_client()
    data = [
        {
            "id": b["id"],
            "repo_name": b["repo_name"],
            "language": b["language"],
            "chunk_file": b["chunk_file"],
            "chunk_lines": b["chunk_lines"],
            "chunk_code": b["chunk_code"],
            "summary": b["summary"],
            "embedding": b["embedding"],
        }
        for b in batch
    ]
    client.insert(collection_name=collection_name, data=data)


def process_repo_chunks(entry):
    repo_path = entry["repo_path"]
    language = entry["language"]
    results = []

    try:
        nodes = parse_codebase_into_chunks(repo_path, language=language)
    except Exception as e:
        print(f"⚠️ Failed to parse {repo_path}: {e}")
        return results

    for node in nodes:
        try:
            summary = summarize_code_llama(node.text)
            embedding = embed_text(summary)[0]

            metadata = node.metadata
            results.append(
                {
                    "id": str(uuid.uuid4()),
                    "repo_name": os.path.basename(repo_path),
                    "language": language,
                    "chunk_file": metadata.get("filepath", "unknown"),
                    "chunk_code": node.text,
                    "chunk_lines": f"{metadata.get('start_line', '?')}-{metadata.get('end_line', '?')}",
                    "summary": summary,
                    "embedding": embedding.tolist(),
                }
            )
        except Exception as err:
            print(f"❌ Failed to summarize/embed chunk: {err}")
            continue

    return results

def process_repo_chunks_naive(entry):
    repo_path = entry["repo_path"]
    language = entry["language"]
    results = []

    try:
        nodes = parse_codebase_into_chunks_naive(repo_path, language=language)
    except Exception as e:
        print(f"⚠️ Failed to parse {repo_path}: {e}")
        return results

    for node in nodes:
        try:
            summary = node.text
            embedding = embed_text(node.text)[0]

            metadata = node.metadata
            results.append(
                {
                    "id": str(uuid.uuid4()),
                    "repo_name": os.path.basename(repo_path),
                    "language": language,
                    "chunk_file": metadata.get("filepath", "unknown"),
                    "chunk_code": node.text,
                    "chunk_lines": f"{metadata.get('start_line', '?')}-{metadata.get('end_line', '?')}",
                    "summary": summary,
                    "embedding": embedding.tolist(),
                }
            )
        except Exception as err:
            print(f"❌ Failed to summarize/embed chunk: {err}")
            continue

    return results


def batched_iterable(iterable, size):
    """Yield successive batches from an iterable"""
    it = iter(iterable)
    while batch := list(itertools.islice(it, size)):
        yield batch


def create_embeddings(
    dataset,
    languages,
    collection_name="code_chunks_index",
    limit=None,
    max_workers=8,
    batch_size=10,
):
    """
    Parses code, summarizes, embeds, and inserts into Milvus in parallel.

    Args:
        dataset (str): Dataset name to process (e.g., 'codesearchnet').
        languages (list): List of programming languages to include.
        collection_name (str): Name of the Milvus collection to insert into.
        limit (int | None): Limit on number of repos to process.
        max_workers (int): Max threads for parallel processing.
        batch_size (int): Number of records per insert.
    """
    init_milvus_collection(collection_name)
    chunks_data_iter = stream_codesearchnet_data(os.environ["CODE_SEARCH_NET_PATH"], languages)

    all_results = []
    parsed_repos = set()
    total_repos_processed = 0

    for job_batch in tqdm(
        batched_iterable(chunks_data_iter, batch_size), desc="📦 Submitting batches"
    ):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for entry in job_batch:
                repo_name = entry["repo"]

                if repo_name in parsed_repos or (limit and total_repos_processed >= limit):
                    continue

                parsed_repos.add(repo_name)
                total_repos_processed += 1

                futures.append(
                    executor.submit(
                        process_repo_chunks,
                        {**entry, "repo_path": os.path.join("src/data/code_search_net_repos", repo_name)},
                    )
                )

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="🔄 Processing repositories"
            ):
                results = future.result()
                if results:
                    all_results.extend(results)

                while len(all_results) >= batch_size:
                    insert_batch_to_milvus(collection_name, all_results[:batch_size])
                    all_results = all_results[batch_size:]

        if limit and total_repos_processed >= limit:
            break

    if all_results:
        insert_batch_to_milvus(collection_name, all_results)

    print(f"✅ Embedded and inserted {total_repos_processed} repos into Milvus collection: {collection_name}")


def create_embeddings_naive(
    dataset,
    languages,
    collection_name="code_chunks_index_naive",
    limit=None,
    max_workers=8,
    batch_size=10,
):
    """
    Parses code, summarizes, embeds, and inserts into Milvus in parallel.

    Args:
        dataset (str): Dataset name to process (e.g., 'codesearchnet').
        languages (list): List of programming languages to include.
        collection_name (str): Name of the Milvus collection to insert into.
        limit (int | None): Limit on number of repos to process.
        max_workers (int): Max threads for parallel processing.
        batch_size (int): Number of records per insert.
    """
    init_milvus_collection(collection_name)
    chunks_data_iter = stream_codesearchnet_data(os.environ["CODE_SEARCH_NET_PATH"], languages)

    all_results = []
    parsed_repos = set()
    total_repos_processed = 0

    for job_batch in tqdm(
        batched_iterable(chunks_data_iter, batch_size), desc="📦 Submitting batches"
    ):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for entry in job_batch:
                repo_name = entry["repo"]

                if repo_name in parsed_repos or (limit and total_repos_processed >= limit):
                    continue

                parsed_repos.add(repo_name)
                total_repos_processed += 1

                futures.append(
                    executor.submit(
                        process_repo_chunks_naive,
                        {**entry, "repo_path": os.path.join("src/data/code_search_net_repos", repo_name)},
                    )
                )

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="🔄 Processing repositories"
            ):
                results = future.result()
                if results:
                    all_results.extend(results)

                while len(all_results) >= batch_size:
                    insert_batch_to_milvus(collection_name, all_results[:batch_size])
                    all_results = all_results[batch_size:]

        if limit and total_repos_processed >= limit:
            break

    if all_results:
        insert_batch_to_milvus(collection_name, all_results)

    print(f"✅ Embedded and inserted {total_repos_processed} repos into Milvus collection: {collection_name}")
