import argparse
import os
import dotenv

from src.code.chunker_v3 import parse_codebase_into_chunks, parse_codebase_into_chunks_naive

dotenv.load_dotenv()



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
      nodes = parse_codebase_into_chunks(repo_path, 'python', max_chars = 500000, chunk_lines = 5000)
    else:
      nodes = parse_codebase_into_chunks_naive(repo_path, 'python')

    print(f"Total Chunks: {len(nodes)}")

    from llama_index.core.storage.docstore import SimpleDocumentStore

    repo_name = repo_path.split("/")[-1].split(".")[0]
    persist_path = f"./storage/{repo_name}.json"

    # Save nodes in a dedicated store per repo
    # docstore = SimpleDocumentStore(namespace=repo_name)
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    docstore.persist(persist_path)



if __name__ == "__main__":
    main()
