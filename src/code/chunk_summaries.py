import openai

import os



API_KEY = os.environ["NAVIGATOR_KEY"]

client = openai.OpenAI(
    api_key=API_KEY,
    base_url="https://api.ai.it.ufl.edu",  # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
)

# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

model_id = "meta-llama/Llama-2-7b-chat-hf"
hf_token = ""

# tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     token=hf_token,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )


# def summarize_code_llama_hugging_face(code_chunk):
#     prompt = f"<s>[INST] Summarize this Python function in one sentence:\n{code_chunk} [/INST]"
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
#     outputs = model.generate(**inputs, max_new_tokens=64)
#     summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return summary.strip()


def summarize_code_llama(code_chunk):

    model = "llama-3.1-8b-instruct"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Below is a code chunk. Write a summary for this chunk's logic within 150 words. The summary should only include relavent details and not be wordy. Only return the summary in plain english. \n\n"
                + code_chunk,
            }
        ],
    )

    return response.choices[0].message.content


def answer_query(query, contexts):

    model = "llama-3.1-8b-instruct"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Below is some context. Answer the question based on the context. The answer should only include relavent details and not be wordy. Only return the answer in plain english. \n\n" + "Query: " + query + "\n\n" + "Context: "+ contexts
            }
        ],
    )

    return response.choices[0].message.content


def compare_contexts(query, smart_contexts, naive_contexts):
    model = "llama-3.1-8b-instruct"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Below is a query and two contexts. Answer which context is more relavent. Only return the answer as 'smart' or 'naive' \n\n" + "Query: " + query + "\n\n" + "Smart Context: "+ "\n".join(smart_contexts) + "\n\n" + "Naive Context: "+ "\n".join(naive_contexts),
            }
        ],
    )

    return response.choices[0].message.content


if __name__ == "__main__":

    code = """
    def get_user_by_id(user_id):
        return db.query(User).filter(User.id == user_id).first()
    """

    print("🔍 Code:")
    print(code.strip())
    print("\n📝 Summary:")
    print(summarize_code_llama(code))
