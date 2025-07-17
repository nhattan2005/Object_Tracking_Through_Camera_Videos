import faiss
import json
import numpy as np
import os

def index_reid(metadata_list, index_file="reid.index", meta_file="reid_metadata.json"):
    feats = np.array([m['clip_feat'] for m in metadata_list]).astype('float32')

    if len(feats) == 0:
        raise ValueError("Không có feature nào để index! Kiểm tra đầu vào của bạn.")

    # Nếu index đã tồn tại thì load lên và append
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
        with open(meta_file, 'r') as f:
            existing_meta = json.load(f)
        all_meta = existing_meta + metadata_list
    else:
        index = faiss.IndexFlatL2(feats.shape[1])
        all_meta = metadata_list

    index.add(feats)

    faiss.write_index(index, index_file)
    with open(meta_file, 'w') as f:
        json.dump(all_meta, f)

    print(f"[INFO] FAISS index now has {index.ntotal} vectors.")
