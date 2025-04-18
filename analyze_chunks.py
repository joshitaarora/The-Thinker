import argparse
import os
import dotenv

from src.code.chunker_v3 import parse_codebase_into_chunks, parse_codebase_into_chunks_naive

dotenv.load_dotenv()


import glob

def main():

    from llama_index.core.storage.docstore import SimpleDocumentStore

    
    for path in glob.glob('./storage/*.json'):
      # Save nodes in a dedicated store per repo
      print(f"{path=}")
      # docstore = SimpleDocumentStore.from_persist_path(path, namespace=path.split("/")[-1].split(".")[0])
      docstore = SimpleDocumentStore.from_persist_path(path)


      for k,n in docstore.docs.items():
        print(len(n))
        print(k)
        print(n)

        print()
        print()

      



if __name__ == "__main__":
    main()
