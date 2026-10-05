import os
import time
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from configs.model import MOVIE_LEN_ITEM_CARDINALITY
from data.dataset_manager import dataset_manager, DatasetType


load_dotenv()

# DashScope text-embedding-v4 supports up to 10 inputs per request.
BATCH_SIZE = 10

# Keep the same embedding dimensionality used by the local embedding pipeline.
EMBEDDING_MODEL_NAME = "text-embedding-v4"
EMBEDDING_DIM = 2048


class QwenAPIEmbedder:
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        embedding_dim: int = EMBEDDING_DIM,
    ):
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY is missing. Check that .env is configured and loaded."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model_name = model_name
        self.embedding_dim = embedding_dim

    def embed(self, texts: List[str]) -> np.ndarray:
        # Normalize missing or non-string values before sending API requests.
        cleaned_texts = []
        for t in texts:
            if t is None:
                cleaned_texts.append("")
            elif not isinstance(t, str):
                cleaned_texts.append(str(t))
            else:
                cleaned_texts.append(t)

        response = self.client.embeddings.create(
            model=self.model_name,
            input=cleaned_texts,
            dimensions=self.embedding_dim,
        )

        # Sort by response index so output order matches input order.
        data_sorted = sorted(response.data, key=lambda x: x.index)
        vectors = [item.embedding for item in data_sorted]

        return np.asarray(vectors, dtype=np.float32)


def embed_movie_len_data(movie_len_data: pd.DataFrame, global_embeddings: np.ndarray):
    num_data = movie_len_data.shape[0]
    num_batches = (num_data + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Processing {num_data} movies in {num_batches} batches")

    movie_ids = movie_len_data["movieId"].tolist()
    overviews = movie_len_data["overview"].fillna("").tolist()

    embedder = QwenAPIEmbedder()

    for i in range(num_batches):
        start = time.time()

        left = i * BATCH_SIZE
        right = min((i + 1) * BATCH_SIZE, num_data)

        idx = movie_ids[left:right]
        batch = overviews[left:right]

        if not batch:
            continue

        batch_embeddings = embedder.embed(batch)
        global_embeddings[idx] = batch_embeddings

        # Periodically checkpoint embeddings so long API jobs can resume.
        if i % 10 == 0:
            print(f"Processed {right} movies")
            s = time.time()
            np.save("artifacts/movie_len_llm_embeddings.npy", global_embeddings)
            print(f"Time taken to save: {time.time() - s:.2f} seconds")

        print(f"Time taken to process batch {i}: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
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

    # Use float32 to reduce storage and memory pressure.
    global_embeddings = np.zeros(
        (MOVIE_LEN_ITEM_CARDINALITY, EMBEDDING_DIM),
        dtype=np.float32,
    )

    embed_movie_len_data(movie_len_data, global_embeddings)

    np.save("artifacts/movie_len_llm_embeddings.npy", global_embeddings)
    print("All embeddings saved.")
