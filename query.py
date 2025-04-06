import argparse
from src.code.embedder import embed_text
from src.code.vector_db import query_milvus_by_embedding
from src.code.chunk_summaries import answer_query


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
    parser = argparse.ArgumentParser(description="Query Milvus and summarize with LLaMA.")
    parser.add_argument("--mode", choices=["naive", "smart"], required=True, help="Query mode")
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

    contexts = query_milvus_by_embedding(query_embedding[0], collection, repo_name="pat")

    # import pdb; pdb.set_trace()

    context_string = "\n".join([c['entity']['summary'] if collection == "code_chunks_index_naive" else c['entity']['summary'] + "\n\n" + c['entity']['chunk_code']  for c in contexts])

    print(f"Retrieved {len(contexts)} contexts")
    # print(context_string)

    summary = answer_query(query, context_string)


    print("\n--- Summary ---\n")
    print(summary)


if __name__ == "__main__":
    main()
