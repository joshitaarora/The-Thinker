from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Llama-2-7b-chat-hf"
hf_token = ""

tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=hf_token,
    torch_dtype=torch.float16,  
    device_map="auto"
)

def summarize_code_llama(code_chunk):
    prompt = f"<s>[INST] Summarize this Python function in one sentence:\n{code_chunk} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=64)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary.strip()


code = """
def get_user_by_id(user_id):
    return db.query(User).filter(User.id == user_id).first()
"""


print("🔍 Code:")
print(code.strip())
print("\n📝 Summary:")
print(summarize_code_llama(code))
