import os
import pickle

import faiss
import numpy as np
import torch

from configs.model import OUTPUT_MODEL_PATH

"""
Build ANN index for KuaiRand item embeddings.
This script expects:
  - artifacts/kuairand_item_embeddings.pt
  - artifacts/kuairand_item_ids.npy
"""


def build_index(embeddings: np.ndarray, index_type: str = "IVF", nlist: int = 64):
    """
    Build ANN index from item embeddings.

    Args:
        embeddings: np.ndarray with shape (num_items, embedding_dim), float32.
        index_type: one of "Flat", "IVF", "HNSW".
        nlist: IVF cluster number.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if embeddings.shape[0] == 0:
        raise ValueError("embeddings is empty")

    d = embeddings.shape[1]
    if index_type == "Flat":
        index = faiss.IndexFlatL2(d)
    elif index_type == "IVF":
        quantizer = faiss.IndexFlatL2(d)
        # nlist should not exceed number of vectors.
        nlist = min(nlist, embeddings.shape[0])
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
        index.train(embeddings)
    elif index_type == "HNSW":
        index = faiss.IndexHNSWFlat(d, M=16)
    else:
        raise ValueError(f"Invalid index type: {index_type}")

    index.add(embeddings)
    return index


def save_index(index, id_remapper):
    faiss.write_index(index, os.path.join(OUTPUT_MODEL_PATH, "kuairand_videos.index"))
    with open(os.path.join(OUTPUT_MODEL_PATH, "kuairand_video_id_remapper.pkl"), "wb") as file:
        pickle.dump(id_remapper, file)


if __name__ == "__main__":
    embeddings = torch.load(os.path.join(OUTPUT_MODEL_PATH, "kuairand_item_embeddings.pt"), weights_only=False)
    video_ids = np.load(os.path.join(OUTPUT_MODEL_PATH, "kuairand_item_ids.npy"))

    embeddings = np.asarray(embeddings, dtype=np.float32)
    video_ids = np.asarray(video_ids, dtype=np.int64)
    if embeddings.shape[0] != video_ids.shape[0]:
        raise ValueError(
            f"Mismatch between embedding count ({embeddings.shape[0]}) and video id count ({video_ids.shape[0]})."
        )

    # index internal id -> original video_id
    video_id_remapper = {idx: int(video_id) for idx, video_id in enumerate(video_ids.tolist())}
    print(f"Loaded embeddings: {embeddings.shape}")

    index = build_index(embeddings, index_type="IVF")

    # quick sanity check
    query_embedding = np.expand_dims(embeddings[0], axis=0)
    distances, indices = index.search(query_embedding, 5)
    print("Sample neighbors (video_id, distance):")
    for dist, idx in zip(distances[0], indices[0]):
        print(video_id_remapper[int(idx)], float(dist))

    save_index(index, video_id_remapper)
    print("Saved index to artifacts/kuairand_videos.index")
    print("Saved remapper to artifacts/kuairand_video_id_remapper.pkl")
