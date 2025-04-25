import argparse
import os
import dotenv

from src.code.chunker_v3 import parse_codebase_into_chunks, parse_codebase_into_chunks_naive
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.llms.openai import OpenAI

from llama_index.packs.code_hierarchy import CodeHierarchyAgentPack


dotenv.load_dotenv()

llm = OpenAI(
    model="gpt-4",
    api_key=os.environ["OPEN_AI_API_KEY"]
    # base_url="https://api.ai.it.ufl.edu",
)


import glob

def main():

  parser = argparse.ArgumentParser(description="Creates chunks for a given repository.")
  parser.add_argument(
      "--repo",
      required=True,
      help="Path to the repository",
  )
  parser.add_argument(
      "--mode", choices=["naive", "smart"], required=True, help="Query mode"
  )

  args = parser.parse_args()
  
  repo = args.repo
  mode = args.mode

  node_file = f'./storage/{repo}_{mode}.json'


  
  print(f"{node_file=}")
  
  docstore = SimpleDocumentStore.from_persist_path(node_file)

  
      
  pack = CodeHierarchyAgentPack(split_nodes=docstore.docs, llm=llm)
  # summary = pack.run(query)



if __name__ == "__main__":
    main()
