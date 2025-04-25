import os
import json
import random
import dotenv
import pandas as pd
import evaluate
import time
import traceback

dotenv.load_dotenv()
import openai

openai.api_key = os.environ["OPEN_AI_API_KEY"]

from src.code.chunker_v3 import parse_codebase_into_chunks, parse_codebase_into_chunks_naive
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.packs.code_hierarchy import CodeHierarchyAgentPack

# Load evaluation metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
ter_metric = evaluate.load("ter")

# Set up OpenAI and LlamaIndex settings
llm = OpenAI(model="gpt-4o-mini", api_key=os.environ["OPEN_AI_API_KEY"])
Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

def log_error(message, e):
    print(f"[ERROR] {message}: {e}")
    traceback.print_exc()

def save_results(results):
    os.makedirs('results', exist_ok=True)
    with open('results/combined_eval.json', 'w') as f:
        json.dump(results, f, indent=2)

def main():
    modes = ["naive", "smart"]
    df = pd.read_csv('Thinker Eval Database - Selected.csv')
    results = {}

    for _, item in df[::-1].iterrows():
        try:
            print(f"\nRunning for {item['Project Name']}")
            repo_name = item['Github URL'].split('/')[-1].split(".")[0]
            repo_path = f"src/data/selected_repos/{repo_name}"
            results[repo_name] = {}
            packs = {}

            for mode in modes:
                try:
                    if mode == "smart":
                        nodes = parse_codebase_into_chunks(repo_path, 'python', max_chars=50000, chunk_lines=50)
                        packs[mode] = CodeHierarchyAgentPack(split_nodes=nodes, llm=llm, verbose=True)
                    else:
                        nodes = parse_codebase_into_chunks_naive(repo_path, 'python')
                        index = VectorStoreIndex(nodes)
                        packs[mode] = index.as_query_engine(llm=llm)
                except Exception as e:
                    log_error(f"Failed to set up {mode} mode for {repo_name}", e)
                    continue

            for i in range(5):
                try:
                    query = item[f'Query {i+1}']
                    doc_ref = item[f'Doc {i+1}']
                    results[repo_name][f'query_{i+1}'] = {
                        'query': query,
                        'documentation_reference': doc_ref
                    }

                    prompt = f"""Answer the query given below only based on the context provided

Query: {query}
"""

                    response_smart = packs["smart"].run(prompt)
                    time.sleep(60)
                    response_naive = packs["naive"].query(prompt).response
                    time.sleep(60)

                    # Compute metrics
                    results[repo_name][f'query_{i+1}'].update({
                        "smart_response": response_smart,
                        "naive_response": response_naive,
                        "metrics": {
                            "smart": {
                                "bleu": bleu_metric.compute(predictions=[response_smart], references=[doc_ref])["bleu"],
                                "rougeL": rouge_metric.compute(predictions=[response_smart], references=[doc_ref])["rougeL"],
                                "ter": ter_metric.compute(predictions=[response_smart], references=[doc_ref])["score"]
                            },
                            "naive": {
                                "bleu": bleu_metric.compute(predictions=[response_naive], references=[doc_ref])["bleu"],
                                "rougeL": rouge_metric.compute(predictions=[response_naive], references=[doc_ref])["rougeL"],
                                "ter": ter_metric.compute(predictions=[response_naive], references=[doc_ref])["score"]
                            }
                        }
                    })

                    # LLM preference comparison
                    options = [
                        {"label": "A", "text": response_smart, "mode": "smart"},
                        {"label": "B", "text": response_naive, "mode": "naive"}
                    ]
                    random.shuffle(options)

                    comparison_prompt = f"""
You are a technical expert. A user asked: "{query}"

Here are two responses:
Response A:
{options[0]['text']}

Response B:
{options[1]['text']}

Based only on their usefulness and informativeness for the query, which response is better? Reply with only 'A' or 'B'.
"""
                    choice = llm.complete(comparison_prompt).text.strip()
                    time.sleep(60)
                    chosen_mode = next(opt['mode'] for opt in options if opt['label'] == choice)

                    results[repo_name][f'query_{i+1}']["llm_chosen_mode"] = chosen_mode
                    print(f"Query {i+1}: LLM chose {chosen_mode.upper()} response")

                except Exception as e:
                    log_error(f"Error processing query {i+1} for {repo_name}", e)
                    continue

            save_results(results)  # Save after each project
        except Exception as e:
            log_error(f"Error processing project {item['Project Name']}", e)
            continue

    save_results(results)

if __name__ == "__main__":
    main()
