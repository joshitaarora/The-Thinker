import argparse
from src.code.embedder import embed_text
from src.code.vector_db import query_milvus_by_embedding
from src.code.chunk_summaries import compare_contexts


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
    parser.add_argument("query", type=str, help="Query string")

    args = parser.parse_args()
    query = args.query

    print(f"Embedding query: {query}")
    query_embedding = embed_text(query)

    
    collection = "code_chunks_index_naive"
    naive_context = query_milvus_by_embedding(query_embedding[0], collection)
    
    collection = "code_chunks_index"
    smart_context = query_milvus_by_embedding(query_embedding[0], collection)


    # import pdb; pdb.set_trace()

    naive_context_string = "\n".join([c['entity']['summary'] if collection == "code_chunks_index_naive" else c['entity']['summary'] + "\n\n" + c['entity']['chunk_code']  for c in naive_context])
    smart_context_string = "\n".join([c['entity']['summary'] if collection == "code_chunks_index_naive" else c['entity']['summary'] + "\n\n" + c['entity']['chunk_code']  for c in smart_context])

    summary = compare_contexts(query, smart_context_string, naive_context_string)


    print("\n--- Summary ---\n")
    print(summary)


if __name__ == "__main__":
    main()
