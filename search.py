import torch
import clip
import json
import faiss

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)

def search_text(query, k=5, index_file="faiss.index", meta_file="metadata.json"):
    with torch.no_grad():
        text_tokens = clip.tokenize([query]).to(device)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy()

    index = faiss.read_index(index_file)
    distances, indices = index.search(text_features, k)

    with open(meta_file, 'r') as f:
        metadata = json.load(f)

    results = [metadata[i] for i in indices[0]]
    return results
