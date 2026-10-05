import os
import time
from typing import List, Tuple

import numpy as np
import pandas as pd

from configs.model import MOVIE_LEN_ITEM_CARDINALITY
from data.dataset_manager import dataset_manager, DatasetType
from embedding.llm_embedder import LLMEmbedder


BATCH_SIZE = 32
MODEL_NAME = "Qwen/Qwen3-1.7B"
EMBEDDING_DIM = 2048  # this needs to be consistent with the model
# For each Qwen3 1.7B model, it would take about 1.7 * 4 ~ 7GB of memory for model, assume
# 512 sequence length, then when B = 1 the activation memory would be 100MB
# I tried to use Ray with 2 CPU to parallel the inference but my Mac couldn't handle it
NUM_CPU = 2


def embed_movie_len_data(movie_len_data: pd.DataFrame, global_embeddings: np.array):
    """
    Using Ray to parallelize this function to be run separate processes.

    The process should maintain its own model and tokenizer, thus the concurrent
    inference should be fine
    """
    num_data = movie_len_data.shape[0]
    num_batches = num_data // BATCH_SIZE + 1
    print(f"Processing {num_data} movies in {num_batches} batches")

    movie_ids = movie_len_data["movieId"].tolist()
    overviews = movie_len_data["overview"].tolist()

    embedder = LLMEmbedder(MODEL_NAME)
    for i in range(num_batches):
        start = time.time()
        idx = movie_ids[i * BATCH_SIZE : min((i + 1) * BATCH_SIZE, num_data)]
        batch = overviews[i * BATCH_SIZE : min((i + 1) * BATCH_SIZE, num_data)]
        batch_embeddings = embedder.embed(batch)
        global_embeddings[idx] = batch_embeddings

        # periodically store the embeddings to make it fault tolerance
        if i % 10 == 0:
            print(f"Processed {i*BATCH_SIZE} movies")
            s = time.time()
            np.save("artifacts/movie_len_llm_embeddings.npy", global_embeddings)
            print(f"Time taken to save: {time.time() - s} seconds")

        print(f"Time taken to process batch {i}: {time.time() - start} seconds")


if __name__ == "__main__":
    # load the links data, which contains the mapping between MovieLen id and tmdb id
    # note that in the model we are using MovieLen id directly without reindexing
    dataset_path = dataset_manager.get_dataset(DatasetType.MOVIE_LENS_LATEST_FULL)
    links = pd.read_csv(os.path.join(dataset_path, "links.csv"))
    links = links.dropna(subset=["tmdbId"])
    links["tmdb_id"] = links["tmdbId"].astype(int)

    movie_len_data = pd.read_parquet("artifacts/movie_len_data.parquet")
    movie_len_data = movie_len_data.drop(columns=["movie_id"])
    movie_len_data = movie_len_data.merge(links, on="tmdb_id", how="inner")
    movie_len_data = movie_len_data[["movieId", "overview"]]
    print(movie_len_data.head())

    rows = movie_len_data.shape[0]
    print(f"Processing {rows} movies")

    global_embeddings = np.zeros((MOVIE_LEN_ITEM_CARDINALITY, EMBEDDING_DIM))
    embed_movie_len_data(movie_len_data, global_embeddings)
