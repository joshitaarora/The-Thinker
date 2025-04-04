from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)



def embed_text(text: str | list[str]):
    if isinstance(text, str):
        text = [text]  # make it a list for batching

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]  # [CLS] token

    # Optional: convert to NumPy and normalize
    embeddings = embeddings.detach().cpu().numpy()
    embeddings = normalize(embeddings, axis=1)

    return embeddings




def create_embeddings(dataset, limit=None):

    if dataset == "codesearchnet":
        pass
        
    elif dataset == "codesc":
        pass
    else:
        raise ValueError("Dataset not supported")  

    

    return chunks