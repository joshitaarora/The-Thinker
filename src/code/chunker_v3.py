import os
from typing import List, Optional
from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.core.text_splitter import CodeSplitter
from llama_index.packs.code_hierarchy import CodeHierarchyNodeParser

from src.data.data_loader import stream_codesearchnet_data

from src.code.vector_db import init_milvus_collection, get_client


# Language to file extension mapping
LANGUAGE_EXTENSIONS = {
    "python": [".py"],
    "java": [".java"],
    "javascript": [".js", ".jsx"],
    "go": [".go"],
    "ruby": [".rb"],
    "php": [".php"]
}

def parse_codebase_into_chunks(
    repo_path: str,
    language: str = "python",
    max_chars: int = 10000,
    chunk_lines: int = 50,
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
    code_splitter = CodeSplitter(
        language=language, max_chars=max_chars, chunk_lines=chunk_lines
    )
    node_parser = CodeHierarchyNodeParser(
        language=language, code_splitter=code_splitter
    )

    return node_parser.get_nodes_from_documents(documents)


from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import uuid


def insert_batch_to_milvus(collection_name, batch):
    client = get_client()
    client.insert(
        collection_name=collection_name,
        data={
            "id": [b["id"] for b in batch],
            "repo_name": [b["repo_name"] for b in batch],
            "language": [b["language"] for b in batch],
            "chunk_file": [b["chunk_file"] for b in batch],
            "chunk_lines": [b["chunk_lines"] for b in batch],
            "summary": [b["summary"] for b in batch],
            "embedding": [b["embedding"] for b in batch],
        }
    )


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
            results.append({
                "id": str(uuid.uuid4()),
                "repo_name": os.path.basename(repo_path),
                "language": language,
                "chunk_file": metadata.get("filepath", "unknown"),
                "chunk_lines": f"{metadata.get('start_line', '?')}-{metadata.get('end_line', '?')}",
                "summary": summary,
                "embedding": embedding.tolist()
            })
        except Exception as err:
            print(f"❌ Failed to summarize/embed chunk: {err}")
            continue

    return results


def create_embeddings(
    dataset,
    languages,
    collection_name="code_chunks_index",
    limit=None,
    max_workers=8,
    batch_size=1000
):
    """
    Parses code, summarizes, embeds, and inserts into Milvus in parallel.

    Args:
        dataset (str): Dataset name to process (e.g., 'codesearchnet').
        collection_name (str): Name of the Milvus collection to insert into.
        limit (int | None): Limit on number of repos to process.
        max_workers (int): Max threads for parallel processing.
        batch_size (int): Number of records per insert.
    """
    init_milvus_collection(collection_name)
    chunks_data = stream_codesearchnet_data(os.environ["CODE_SEARCH_NET_PATH"], languages)

    all_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_repo_chunks, {**entry, "repo_path": os.path.join("src/data/code_search_net_repos", entry['repo']) }) for entry in chunks_data]

        for future in tqdm(as_completed(futures), total=len(futures), desc="🔄 Processing repositories"):
            results = future.result()
            if results:
                all_results.extend(results)

            while len(all_results) >= batch_size:
                insert_batch_to_milvus(collection_name, all_results[:batch_size])
                all_results = all_results[batch_size:]

    if all_results:
        insert_batch_to_milvus(collection_name, all_results)

    print(f"✅ Embedded and inserted into Milvus collection: {collection_name}")
