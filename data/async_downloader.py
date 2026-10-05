import os
import random
import time
from typing import List

import asyncio
import ray
import pandas as pd

import dotenv
import jsonlines
from themoviedb import aioTMDb

from data.dataset_manager import dataset_manager, DatasetType

dotenv.load_dotenv()

NUM_CPU = 8


@ray.remote
class DownloaderActor:
    """
    This is a ray actor that downloads data using TMDB API asynchronously.

    For each batch, we would directly append the results to the local jsonl file
    for persistence; this is not the most efficient way but it a simplified way to
    achieve snapshotting.
    """

    def __init__(self, actor_id: int, movie_ids: List[int]):
        if not os.environ.get("TMDB_API_KEY"):
            raise ValueError("TMDB_API_KEY is not set")
        self.tmdb = aioTMDb(key=os.environ.get("TMDB_API_KEY"))
        self.actor_id = actor_id
        self.movie_ids = movie_ids

        self.batch_size = 10
        self.cooldown_time = random.randint(3, 10)

        self.output_file = os.path.join("/tmp", f"tmdb_data_{self.actor_id}.jsonl")

    async def download_movie_data(self, movie_id: int) -> dict:
        try:
            movie = await self.tmdb.movie(movie_id).details(append_to_response="credits,external_ids")
            return {
                "movie_id": movie_id,
                "title": movie.title,
                "year": movie.year,
                "overview": movie.overview,
            }
        except Exception as e:
            print(f"Error downloading movie data for {movie_id}: {e}")
            return {
                "movie_id": movie_id,
                "title": "",
                "year": "",
                "overview": "",
            }

    async def download_batch(self) -> bool:
        """
        We download all the movie data, we would download the movie data in batches.
        """
        num_batches = len(self.movie_ids) // self.batch_size
        print(f"Downloading {num_batches} batches")
        for i in range(num_batches):
            try:
                batch_movie_ids = self.movie_ids[i * self.batch_size : min((i + 1) * self.batch_size, len(self.movie_ids))]
                tasks = [self.download_movie_data(movie_id) for movie_id in batch_movie_ids]
                results = await asyncio.gather(*tasks)

                with jsonlines.open(self.output_file, "a") as f:
                    f.write_all(results)
            except Exception as e:
                print(f"Error downloading batch {i}: {e}")
                continue

            await asyncio.sleep(self.cooldown_time)
            if i % 10 == 0:
                print(f"Downloaded {i*self.batch_size} movies")

        return True


def main():
    dataset_path = dataset_manager.get_dataset(DatasetType.MOVIE_LENS_LATEST_FULL)
    links = pd.read_csv(os.path.join(dataset_path, "links.csv"))
    tmdb_ids = links["tmdbId"].tolist()
    # we could remove this and download all the data, but I'm a little bit concern if it would be
    # rate limited by TMDB. Thus I process 10k at a time and the new result is appended to the file.
    tmdb_ids = tmdb_ids[70000:]

    # we need to sort the imdb ids to ensure each actor gets the same job to do each time
    tmdb_ids = sorted(tmdb_ids)

    # Create multiple actors
    num_actors = NUM_CPU
    batch_size = len(tmdb_ids) // num_actors
    actors = [DownloaderActor.remote(i, tmdb_ids[i * batch_size : (i + 1) * batch_size]) for i in range(num_actors)]

    # Start time
    start_time = time.time()

    # Submit tasks to actors
    futures = [actor.download_batch.remote() for actor in actors]

    # Get results
    results = ray.get(futures)

    # End time
    end_time = time.time()

    # Print results
    print(f"Download completed in {end_time - start_time:.2f} seconds, results: {results}")


if __name__ == "__main__":
    print("Ray initializing...")
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=NUM_CPU)
    main()
