import os
import json
import torch
import clip
from PIL import Image
import faiss
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def index_images(metadata_list, index_file="faiss.index", meta_file="metadata.json"):
    image_paths = [m['path'] for m in metadata_list]
    images = [preprocess(Image.open(p).convert("RGB")) for p in image_paths]
    image_tensor = torch.stack(images).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.cpu().numpy()

    index = faiss.IndexFlatL2(image_features.shape[1])
    index.add(image_features)

    faiss.write_index(index, index_file)
    with open(meta_file, 'w') as f:
        json.dump(metadata_list, f)
