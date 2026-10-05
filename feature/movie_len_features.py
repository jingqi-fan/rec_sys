import hashlib
import pandas as pd
from collections import Counter


"""
    This module contains all the util functions for feature engineering
    of the MovieLen dataset
"""


GENRES = [
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Fantasy",
    "Romance",
    "Drama",
    "Action",
    "Crime",
    "Thriller",
    "Horror",
    "Mystery",
    "Sci-Fi",
    "IMAX",
    "Documentary",
    "War",
    "Musical",
    "Western",
    "Film-Noir",
    "(no genres listed)",
]


class MovieLenFeatureStore:

    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame, seq_length: int = 10):
        """
        Prepare features for MovieLen dataset

        For now we would support it on teh MovieLen latest dataset, which is
        small enough to be hold in memory; later once we need to process larger
        dataset, we would write to files for training and dedicated feature store
        to host for serving
        """
        self.ratings = ratings
        self.movies = movies
        self.seq_length = seq_length

        # indexed by userId, sort by userId to make the access consistent
        self.item_sequence = MovieLenFeatureStore.fetch_item_sequence(ratings=ratings, seq_length=seq_length).sort_values(by="userId")
        self.action_sequence, self.unique_ids = MovieLenFeatureStore.fetch_action_sequence(ratings=ratings, seq_length=seq_length)
        self.action_sequence = self.action_sequence.sort_values(by="userId")
        self.user_viewed_genres = MovieLenFeatureStore.fetch_user_viewed_genres(ratings=ratings, movies=movies)
        self.user_num_viewed_movies = MovieLenFeatureStore.fetch_num_viewed_movies(ratings=ratings)

        # indexed by movieId
        self.movie_genres = MovieLenFeatureStore.fetch_movie_genres(movies=movies)

    @classmethod
    def fetch_item_sequence(cls, ratings: pd.DataFrame, seq_length: int = 10):
        """
        A helper function to extract user rated movie ids in chronologic order, from earliest to latest

        If the sequence length is insufficient, the row would be dropped for now
        TODO: support a default PADDING token instead of dropping the row to preserve as many samples as possible

        return DataFrame in the following format

            |userId: integer|item_sequence: list[integer]|
        """
        item_sequence = (
            ratings.groupby("userId")
            .apply(lambda x: x.sort_values("timestamp").head(seq_length)["movieId"].tolist())  # for group less than `seq_length`, this would return all rows
            .reset_index(name="item_sequence")
        )  # userId, item_sequence
        item_sequence = item_sequence[item_sequence["item_sequence"].apply(len) >= seq_length].reset_index(drop=True)
        return item_sequence

    @classmethod
    def fetch_action_sequence(cls, ratings: pd.DataFrame, seq_length: int = 10):
        """
        A helper function to extract user rate as categorical feature, a.k.a an action in traditional recsys

        return DataFrame in the following format

            |userId: int|action_sequence: list[int]|
        """
        # convert the ratings to category
        ratings["rating_cat"], unique_ids = pd.factorize(ratings["rating"])

        action_sequence = (
            ratings.groupby("userId").apply(lambda x: x.sort_values("timestamp").head(seq_length)["rating_cat"].tolist()).reset_index(name="action_sequence")
        )  # userId, action_sequence
        action_sequence = action_sequence[action_sequence["action_sequence"].apply(len) >= seq_length].reset_index(drop=True)

        # revert the addition of column to make it side effect free
        ratings.drop(columns=["rating_cat"], inplace=True)

        return action_sequence, unique_ids

    @classmethod
    def fetch_movie_genres(cls, movies: pd.DataFrame):
        """
        A helper function to convert the movie genres into a list of categorical type

        return DataFrame in the following format

            |movieId: int|genres: list[int]|
        """
        # make a copy to avoid change original data, side effect free
        df = movies[["movieId", "genres"]].copy()
        # split and explode each genre into a single row
        df["genres"] = df["genres"].str.split("|")
        df = df.explode("genres")
        # categorical encoding
        df["genres"] = pd.Categorical(values=df["genres"], categories=GENRES).codes
        # reduce, agg implicitly have a `x.tolist()`
        df = df.groupby("movieId")["genres"].agg(lambda x: sorted(list(set(x)))).reset_index()
        return df

    @classmethod
    def fetch_user_viewed_genres(cls, ratings: pd.DataFrame, movies: pd.DataFrame):
        """
        A helper function to extract all genres that a user has viewed

        return DataFrame in the following format

            |user_id: int|genres: list[int]|counts: list[int]|
        """
        movie_genres = cls.fetch_movie_genres(movies=movies)  # could be optimized
        df = ratings[["userId", "movieId"]].merge(movie_genres, on="movieId")

        def func(x):
            """
            `x` is the group result from Pandas, each row in `x` is a list of
            the genres the movie user rated
            """
            # Flatten the list of lists and count occurrences
            genre_counts = Counter(g for genres in x for g in genres)
            genre_counts = list(genre_counts.items())
            # sort based on count in reverse order
            genre_counts = sorted(genre_counts, key=lambda x: (-x[1], x[0]))  # sort based on counts, genre id
            return [x[0] for x in genre_counts], [x[1] for x in genre_counts]

        df = df.groupby("userId")["genres"].agg(lambda x: func(x)).reset_index()  # user_id, genres
        df["sorted_genres"] = df["genres"].apply(lambda x: x[0])
        df["count_genres"] = df["genres"].apply(lambda x: x[1])
        df.drop(columns=["genres"], inplace=True)
        return df

    @classmethod
    def fetch_num_viewed_movies(cls, ratings: pd.DataFrame):
        """
        A helper function to retrieve the number of viewed movies for each user, as
        a dense numeric feature

        return DataFrame in the following format

            |user_id: int|num_viewed_movies: float|
        """
        df = ratings.groupby("userId")["movieId"].agg(lambda x: len(x)).reset_index(name="num_viewed_movies")
        return df


if __name__ == "__main__":
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "genres": ["Adventure|Animation", "Drama", "Action|Animation"],
        }
    )
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2],
            "movieId": [1, 2, 3, 2, 3],
            "ratings": [5.0, 2.0, 3.0, 1.0, 2.0],
        }
    )

    df = MovieLenFeatureStore.fetch_movie_genres(movies=movies)
    expected_df = pd.DataFrame({"movieId": [1, 2, 3], "genres": [[0, 1], [6], [1, 7]]})
    result = df.merge(expected_df, on="movieId", suffixes=["x", "y"])
    check = result["genresx"] == result["genresy"]
    assert all(check)
    print("MovieLen movie genres feature test pass...")

    df = MovieLenFeatureStore.fetch_user_viewed_genres(ratings=ratings, movies=movies)
    expected_df = pd.DataFrame({"userId": [1, 2], "sorted_genres": [[1, 0, 6, 7], [1, 6, 7]], "count_genres": [[2, 1, 1, 1], [1, 1, 1]]})
    result = df.merge(expected_df, on="userId", suffixes=["_x", "_y"])
    check = result["sorted_genres_x"] == result["sorted_genres_y"]
    assert all(check)
    check = result["count_genres_x"] == result["count_genres_y"]
    assert all(check)
    print("MovieLen user viewed genres test pass...")

    df = MovieLenFeatureStore.fetch_num_viewed_movies(ratings=ratings)
    expected_df = pd.DataFrame(
        {
            "userId": [1, 2],
            "num_viewed_movies": [3, 2],
        }
    )
    result = df.merge(expected_df, on="userId", suffixes=["_x", "_y"])
    check = result["num_viewed_movies_x"] == result["num_viewed_movies_y"]
    assert all(check)
    print("MovieLen user num viewed movies test pass...")
