import argparse
import os
import dotenv

from src.code.chunker_v3 import parse_codebase_into_chunks, parse_codebase_into_chunks_naive

dotenv.load_dotenv()

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo


def sanitize_node_relationships(node):
    if node.relationships is None or not isinstance(node.relationships, dict):
        node.relationships = {}
    else:
        clean_relationships = {}
        for rel, value in node.relationships.items():
            if isinstance(value, RelatedNodeInfo):
                clean_relationships[rel] = value
            elif isinstance(value, dict):
                try:
                    # Try to reconstruct RelatedNodeInfo from dict
                    clean_relationships[rel] = RelatedNodeInfo(**value)
                except Exception:
                    continue  # skip invalid ones
            else:
                continue  # skip None or bad values
        node.relationships = clean_relationships

def main():
    parser = argparse.ArgumentParser(description="Creates chunks for a given repository.")
    parser.add_argument(
        "--repo-path",
        required=True,
        help="Path to the repository",
    )
    parser.add_argument(
        "--mode", choices=["naive", "smart"], required=True, help="Query mode"
    )

    args = parser.parse_args()
    
    repo_path = args.repo_path
    mode = args.mode
    
    if mode == "smart":
      nodes = parse_codebase_into_chunks(repo_path, 'python', max_chars = 50000, chunk_lines = 50)
    else:
      nodes = parse_codebase_into_chunks_naive(repo_path, 'python')

    
    # clean node relationships
    for node in nodes:
      sanitize_node_relationships(node)

    print(f"Total Chunks: {len(nodes)}")

    from llama_index.core.storage.docstore import SimpleDocumentStore

    repo_name = repo_path.split("/")[-1].split(".")[0]
    persist_path = f"./storage/{repo_name}_{mode}.json"

    # Save nodes in a dedicated store per repo
    # docstore = SimpleDocumentStore(namespace=repo_name)
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    docstore.persist(persist_path)



if __name__ == "__main__":
    main()
