import argparse

import numpy as np
import torch

from configs.model import LARGE_TRANSACT_CONFIG, OUTPUT_MODEL_PATH, SEMANTIC_TRANSACT_CONFIG
from data.dataset_movie_len import prepare_movie_len_transact_dataset, DatasetType
from model.transact import TransAct
from trainer.simple_trainer import SimpleTrainer


DEFAULT_MOVIE_LEN_EMBEDDING_PATH = "artifacts/movie_embeddings_v2.pt"
SEMANTIC_MOVIE_LEN_EMBEDDING_PATH = "artifacts/movie_len_llm_embeddings.npy"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-embedding", action="store_true")
    args = parser.parse_args()

    config, model_name = None, "transact-movie-len-full"
    if args.semantic_embedding:
        embedding_store = np.load(SEMANTIC_MOVIE_LEN_EMBEDDING_PATH)
        embedding_store = embedding_store.astype(np.float32)
        config = SEMANTIC_TRANSACT_CONFIG
        model_name = "transact-movie-len-full-semantic"
    else:
        embedding_store = torch.load(DEFAULT_MOVIE_LEN_EMBEDDING_PATH, weights_only=False)
        config = LARGE_TRANSACT_CONFIG

    model = TransAct(config=config)
    train_dataset, eval_dataset = prepare_movie_len_transact_dataset(embedding_store=embedding_store, dataset_type=DatasetType.MOVIE_LENS_LATEST_FULL, history_seq_length=15)

    trainer = SimpleTrainer(model=model, train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer.train(num_epochs=1)

    trainer.save(model_name, movie_index=None, path=OUTPUT_MODEL_PATH)
