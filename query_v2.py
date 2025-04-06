import os
import argparse
from src.code.embedder import embed_text
from src.code.vector_db import query_milvus_by_embedding
from src.code.chunk_summaries import answer_query

from llama_index.core.schema import Document
from llama_index.core import SimpleDirectoryReader
from llama_index.core.text_splitter import CodeSplitter
from llama_index.llms.openai import OpenAI
from llama_index.packs.code_hierarchy import CodeHierarchyNodeParser
from llama_index.packs.code_hierarchy import CodeHierarchyAgentPack
from pathlib import Path
from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.core.text_splitter import CodeSplitter
from llama_index.packs.code_hierarchy import CodeHierarchyNodeParser

from llama_index.packs.code_hierarchy import CodeHierarchyAgentPack


def build_prompt(contexts, query):
    context_text = "\n\n---\n\n".join(contexts)
    prompt = (
        f"Here are some relevant code chunks based on the query: '{query}'\n\n"
        f"{context_text}\n\n"
        "Using the code chunks above, generate a summary of the logic relevant to the query. "
        "Keep it under 150 words, do not include irrelevant info."
    )
    return prompt


def main():
    parser = argparse.ArgumentParser(
        description="Query Milvus and summarize with LLaMA."
    )
    parser.add_argument(
        "--mode", choices=["naive", "smart"], required=True, help="Query mode"
    )
    parser.add_argument("query", type=str, help="Query string")

    args = parser.parse_args()
    mode = args.mode
    query = args.query

    print(f"Embedding query: {query}")
    query_embedding = embed_text(query)

    print(f"Retrieving top 5 contexts using {mode} mode...")
    if mode == "naive":
        collection = "code_chunks_index_naive"
    else:
        collection = "code_chunks_index"

    # contexts = query_milvus_by_embedding(query_embedding[0], collection, repo_name="pat", top_k=10)

    # documents = [
    #     Document(
    #         text=r.get("entity", {})["chunk_code"],
    #         metadata={
    #             "file_path": r.get("entity", {})["chunk_file"]
    #         }
    #     )
    #     for r in contexts
    # ]

    documents = SimpleDirectoryReader(
        input_dir="/home/abbasidaniyal/Projects/The-Thinker/src/data/code_search_net_repos/NoviceLive/pat/",
        file_metadata=lambda x: {"filepath": x},
        recursive=True,
        required_exts=[".py"],
    ).load_data()

    split_nodes = CodeHierarchyNodeParser(
        language="python",
        # You can further parameterize the CodeSplitter to split the code
        # into "chunks" that match your context window size using
        # chunck_lines and max_chars parameters, here we just use the defaults
        code_splitter=CodeSplitter(language="python", max_chars=1000, chunk_lines=10),
    ).get_nodes_from_documents(documents)

    from llama_index.llms.openai import OpenAI

    llm = OpenAI(
        model="gpt-4",
        api_key=os.environ["OPEN_AI_API_KEY"]
        # base_url="https://api.ai.it.ufl.edu",
    )

    pack = CodeHierarchyAgentPack(split_nodes=split_nodes, llm=llm)

    # import pdb; pdb.set_trace()

    # context_string = "\n".join([c['entity']['summary'] if collection == "code_chunks_index_naive" else c['entity']['summary'] + "\n\n" + c['entity']['chunk_code']  for c in contexts])

    # print(f"Retrieved {len(contexts)} contexts")
    # print(context_string)

    # summary = answer_query(query, context_string)

    # pack = CodeHierarchyAgentPack(documents=documents)
    summary = pack.run(query)

    print("\n--- Summary ---\n")
    print(summary)


if __name__ == "__main__":
    main()
