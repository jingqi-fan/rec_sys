import unittest
import torch
from torch.utils.data import DataLoader

import data.dataset_movie_len as dataset


class TestDataset(unittest.TestCase):

    def test_movie_len_dataset(self):
        train_dataset, _, _ = dataset.prepare_movie_len_dataset(history_seq_length=8, reindex=True)

        loader = DataLoader(dataset=train_dataset, batch_size=5)

        it = iter(loader)
        batch = next(it)

        assert batch[0]["user_id"].shape == (5, 1)
        assert batch[0]["item_id"].shape == (5, 1)
        assert batch[0]["user_history_behavior"].shape == (5, 8)

        assert batch[1].shape == (5,)

        # check type of the output
        assert batch[0]["user_id"].dtype == torch.int64
        assert batch[0]["item_id"].dtype == torch.int64

        print("MovieLenDataset batch shape test passed...")

    def test_movie_len_rating_dataset(self):
        train_dataset, _ = dataset.prepare_movie_len_rating_dataset(dataset_type=dataset.DatasetType.MOVIE_LENS_LATEST_SMALL)

        loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
        it = iter(loader)
        batch = next(it)

        assert len(batch) == 3
        users, movies, ratings = batch
        assert len(users) == 5
        assert len(movies) == 5
        assert len(ratings) == 5

        print("MovieLenRatingDataset batch shape test passed...")

    def test_movie_len_transact_dataset(self):
        embedding_store = torch.ones((300000, 64))
        train_dataset, _ = dataset.prepare_movie_len_transact_dataset(
            embedding_store=embedding_store, dataset_type=dataset.DatasetType.MOVIE_LENS_LATEST_SMALL, history_seq_length=15
        )

        loader = DataLoader(train_dataset, batch_size=5)
        it = iter(loader)
        batch, label = next(it)

        assert batch["user_id"].shape == (5, 1)
        assert batch["action_sequence"].shape == (5, 14)
        assert batch["item_sequence"].shape == (5, 14, 64)
        assert batch["candidate"].shape == (5, 64)
        assert batch["user_viewed_genres"].shape == (5, 5)
        assert batch["user_num_viewed_movies"].shape == (5, 1)
        assert batch["candidate_genres"].shape == (5, 5)

        print("MovieLenTransActDataset batch shape test passed...")

    def test_movie_len_semantic_embedding_dataset(self):
        loader = DataLoader(dataset.MovieLenSemanticEmbeddingDataset(), batch_size=5)
        it = iter(loader)
        batch = next(it)

        assert batch.shape == (5, 2048)

        print("MovieLenSemanticEmbeddingDataset batch shape test passed...")


if __name__ == "__main__":
    unittest.main()
