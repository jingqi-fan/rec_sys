from dataclasses import dataclass
import os
import pickle

import faiss
import numpy as np
import pandas as pd
import torch

from configs.model import OUTPUT_MODEL_PATH
from data.dataset_manager import DatasetType, dataset_manager
from server.ebr_client import EBRClient
from server.duckdb_ebr_server import DuckEBRServer
from server.feature_server import FeatureServer, FeatureServerConfig


@dataclass
class Candidate:
    id: int
    title: str
    genres: str
    distance: float


@dataclass
class RetrievalEngineConfig:
    enable_embedding_retrieval_engine: bool = False
    enable_duckdb_retrieval_engine: bool = False
    num_candidates: int = 20


class RetrievalEngine:

    def __init__(self, config: RetrievalEngineConfig, feature_server: FeatureServer):
        cached_path = dataset_manager.get_dataset(DatasetType.MOVIE_LENS_LATEST_FULL)
        self.movies = pd.read_csv(os.path.join(cached_path, "movies.csv")).set_index("movieId")

        # For now we reuse the computation logic in the feature server to retrieve the item user
        # has interacted with before.
        # In long term, this part should be refactored as
        #   1. have a separated offline pipeline to process the data and written into a shared storage
        #   2. feature server directly get the feature from the storage and do necessary post-processing (e.g. padding, truncation)
        #   3. retrieve engine directly get the interacted candidates from the storage

        self.feature_server = feature_server
        self.num_candidates = config.num_candidates

        # for simplicity, we don't support multi source ebr engine; there would be some complexity on
        # how to fairly merge the candidates obtained from different engine together
        assert not (config.enable_embedding_retrieval_engine and config.enable_duckdb_retrieval_engine), "only one ebr retrieval engine is supported as of now"

        # load FAISS index which is used for embedding retrieval
        self.ebr_client = None
        if config.enable_embedding_retrieval_engine:
            self.ebr_client = EBRClient()
        elif config.enable_duckdb_retrieval_engine:
            self.ebr_client = DuckEBRServer()

    def generate_candidates(self, user_id):
        """
        Generate candidates based on different retrieval strategy. Currently we supporting the
        following strategies

            - FAISS EBR: use the positive interaction item in user sequence as the seed to retrieve
                         similar items

        For now this function only handles a single request, there are different optimization that could
        be applied such as batching, multi-threading, etc. I will add these functionality in later
        version
        """
        # user ratings reflect user's positive/negative preference to the model; here we only use the positive
        # interaction, which means that user rates over 4.0 on the movie, as the candidate generation seed
        user_interacted_candidate = self.feature_server.extract_user_interaction_sequence(user_id=user_id, ratings=4.0)
        ebr_candidates = self.ebr_client.send_request(query=user_interacted_candidate, num_candidate=self.num_candidates)

        result = []
        for ebr_candidate in ebr_candidates:
            movie_id = ebr_candidate.id
            movie = self.movies.loc[movie_id]
            result.append(Candidate(id=movie_id, title=movie["title"], genres=movie["genres"], distance=ebr_candidate.score))

        return result


if __name__ == "__main__":

    config = RetrievalEngineConfig(enable_duckdb_retrieval_engine=True, num_candidates=20)
    feature_server = FeatureServer(
        FeatureServerConfig(movie_len_history_seq_length=15, movie_len_dataset_type=DatasetType.MOVIE_LENS_LATEST_SMALL, embedding_store_path="artifacts/movie_embeddings_v2.pt")
    )
    server = RetrievalEngine(config=config, feature_server=feature_server)
    result = server.generate_candidates(user_id=1)
    print(result)
