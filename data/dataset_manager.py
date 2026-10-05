from enum import Enum, auto
import os
import logging
import requests
import zipfile
import io
import tarfile


class DatasetType(Enum):

    # MovieLens dataset: https://grouplens.org/datasets/movielens/latest/
    MOVIE_LENS_LATEST_SMALL = auto()  # 100k ratings, 3600 tags, 9000 movies, 600 users, updated 9/2018
    MOVIE_LENS_LATEST_FULL = auto()  # 33M ratings, 2M tags, 86000 movies, 330975 users, updated 9/2018

    KUAI_RAND_1K = auto()  # 1k users, 4M videos

    def get_url(self) -> str:
        urls = {
            DatasetType.MOVIE_LENS_LATEST_SMALL: "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
            DatasetType.MOVIE_LENS_LATEST_FULL: "https://files.grouplens.org/datasets/movielens/ml-latest.zip",
            DatasetType.KUAI_RAND_1K: "https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz",
        }
        return urls[self]

    def get_name(self) -> str:
        names = {
            DatasetType.MOVIE_LENS_LATEST_SMALL: "ml-latest-small",
            DatasetType.MOVIE_LENS_LATEST_FULL: "ml-latest",
            DatasetType.KUAI_RAND_1K: "kuairand-1k",
        }
        return names[self]

    def get_suffix(self) -> str:
        suffixes = {
            DatasetType.KUAI_RAND_1K: ".tar.gz",
            DatasetType.MOVIE_LENS_LATEST_SMALL: ".zip",
            DatasetType.MOVIE_LENS_LATEST_FULL: ".zip",
        }
        return suffixes[self]


class DatasetManager:
    """
    This is a manager class to handle the dataset on the host, the dataset would
    be downloaded and cached in the /tmp folder by default
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DatasetManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, tmp_folder: str = "./artifacts"):
        if self._initialized:
            return

        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)

        self._tmp_folder = tmp_folder
        self._initialized = True

    def _dataset_exists(self, dataset_path):
        return os.path.exists(dataset_path) and os.listdir(dataset_path)

    def get_dataset(self, dataset_type: DatasetType) -> str:
        dataset_path = os.path.join(self._tmp_folder, dataset_type.get_name())
        if not self._dataset_exists(dataset_path):
            logging.info(f"Requested dataset {dataset_type} does not exists in cache, start downloading")
            dataset_path = self.download_dataset(dataset_type=dataset_type)

        return dataset_path

    def download_dataset(self, dataset_type: DatasetType) -> str:
        logging.info(f"Start downloading {dataset_type}...")
        dataset_path = os.path.join(self._tmp_folder, dataset_type.get_name())
        if self._dataset_exists(dataset_path):
            return dataset_path
        suffix = dataset_type.get_suffix()

        response = requests.get(dataset_type.get_url())
        if response.status_code == 200:
            if suffix == ".zip":
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                    zip_ref.extractall(self._tmp_folder)
                    logging.info(f"Dataset {dataset_type.get_name()} downloaded")
            elif suffix == ".tar.gz":
                with tarfile.open(fileobj=io.BytesIO(response.content)) as tar:
                    tar.extractall(self._tmp_folder)
                    logging.info(f"Dataset {dataset_type.get_name()} downloaded")
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")

        return dataset_path


dataset_manager = DatasetManager()
