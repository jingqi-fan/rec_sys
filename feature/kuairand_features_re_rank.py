from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence
import os

import numpy as np
import pandas as pd

from data.dataset_manager import DatasetType, dataset_manager


def _parse_tag_values(tag_value) -> list[int | str]:
    if isinstance(tag_value, list):
        return [tag for tag in tag_value if not pd.isna(tag)]
    if isinstance(tag_value, str):
        return [tag.strip() for tag in tag_value.split(",") if tag.strip()]
    # if pd.isna(tag_value):
    #     return []
    return [tag_value]


def _get_embedding(item_embeddings, item_id: int) -> np.ndarray | None:
    if isinstance(item_embeddings, Mapping):
        embedding = item_embeddings.get(item_id)
        if embedding is None:
            return None
        return np.asarray(embedding, dtype=np.float64)

    embeddings = np.asarray(item_embeddings)
    if item_id < 0 or item_id >= len(embeddings):
        return None
    return np.asarray(embeddings[item_id], dtype=np.float64)


def _pairwise_cosine_similarity_stats(history_item_ids: Sequence[int], item_embeddings) -> tuple[float, float, float]:
    history_embeddings = []
    for item_id in history_item_ids:
        embedding = _get_embedding(item_embeddings, item_id)
        if embedding is not None:
            history_embeddings.append(embedding)

    if len(history_embeddings) < 2:
        return 0.0, 0.0, 0.0

    embeddings = np.asarray(history_embeddings, dtype=np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normalized_embeddings = embeddings / norms
    similarity_matrix = normalized_embeddings @ normalized_embeddings.T

    upper_triangle = similarity_matrix[np.triu_indices(len(normalized_embeddings), k=1)]
    if upper_triangle.size == 0:
        return 0.0, 0.0, 0.0

    return float(np.mean(upper_triangle)), float(np.max(upper_triangle)), float(np.std(upper_triangle))


def _dominant_tag_ratio(history_item_ids: Sequence[int], video_features: pd.DataFrame, top_k: int = 2) -> float:
    if not history_item_ids:
        return 0.0

    tag_counter = Counter()
    for item_id in history_item_ids:
        if item_id not in video_features.index:
            continue
        tag_values = _parse_tag_values(video_features.loc[item_id, "tag"])
        tag_counter.update(tag_values)

    total_tag_count = sum(tag_counter.values())
    if total_tag_count == 0:
        return 0.0

    top_tag_count = sum(count for _, count in tag_counter.most_common(top_k))
    return float(top_tag_count / total_tag_count)


def _novelty_ratio_mean(history_item_ids: Sequence[int], video_features: pd.DataFrame) -> float:
    if len(history_item_ids) < 2:
        return 0.0

    new_author_count = 0
    new_tag_count = 0
    valid_author_events = 0
    valid_tag_events = 0

    seen_authors = set()
    seen_tags = set()

    for item_id in history_item_ids:
        if item_id not in video_features.index:
            continue

        row = video_features.loc[item_id]

        author_id = row.get("author_id")
        if not pd.isna(author_id):
            valid_author_events += 1
            if author_id not in seen_authors:
                new_author_count += 1
                seen_authors.add(author_id)

        tag_values = _parse_tag_values(row.get("tag"))
        if tag_values:
            valid_tag_events += 1
            if any(tag not in seen_tags for tag in tag_values):
                new_tag_count += 1
            seen_tags.update(tag_values)

    new_author_ratio = float(new_author_count / valid_author_events) if valid_author_events > 0 else 0.0
    new_tag_ratio = float(new_tag_count / valid_tag_events) if valid_tag_events > 0 else 0.0
    return float((new_author_ratio + new_tag_ratio) / 2.0)


def build_kuairand_linear_ts_context(
    history_item_ids: Sequence[int],
    video_features: pd.DataFrame,
    item_embeddings,
    history_size: int = 20,
) -> np.ndarray:
    """
    Build a compact contextual feature vector for alpha selection in DPP reranking.

    Output layout:
        [
            history_sim_mean,
            history_sim_max,
            history_sim_std,
            dominant_tag_ratio,
            novelty_ratio_mean,
        ]

    Args:
        history_item_ids: user clicked item ids ordered from old to new.
        video_features: video side features indexed by video_id. Expected to contain
            at least `tag` and `author_id`.
        item_embeddings: either a mapping {item_id: embedding} or an array indexed by item id.
        history_size: only the most recent history_size items are used.
    """
    if history_size <= 0:
        raise ValueError("history_size must be positive")

    if "tag" not in video_features.columns or "author_id" not in video_features.columns:
        raise ValueError("video_features must contain `tag` and `author_id` columns")

    history_window = list(history_item_ids)[-history_size:]
    sim_mean, sim_max, sim_std = _pairwise_cosine_similarity_stats(history_window, item_embeddings)
    dominant_tag_ratio = _dominant_tag_ratio(history_window, video_features, top_k=2)
    novelty_ratio_mean = _novelty_ratio_mean(history_window, video_features)

    return np.asarray(
        [
            sim_mean,
            sim_max,
            sim_std,
            dominant_tag_ratio,
            novelty_ratio_mean,
        ],
        dtype=np.float64,
    )


def _find_first_existing_file(dataset_path: str, candidate_names: Sequence[str]) -> str | None:
    for root, _, files in os.walk(dataset_path):
        file_name_map = {file_name.lower(): file_name for file_name in files}
        for candidate_name in candidate_names:
            matched_name = file_name_map.get(candidate_name.lower())
            if matched_name is not None:
                return os.path.join(root, matched_name)
    return None


def inspect_kuairand_dataset(max_rows: int = 5) -> None:
    """
    Download KuaiRand if needed and print the structure of several core files.
    """
    dataset_path = dataset_manager.get_dataset(DatasetType.KUAI_RAND_1K)
    print(f"KuaiRand dataset path: {dataset_path}")

    print("\nFiles under dataset path:")
    for root, _, files in os.walk(dataset_path):
        relative_root = os.path.relpath(root, dataset_path)
        for file_name in sorted(files):
            relative_path = file_name if relative_root == "." else os.path.join(relative_root, file_name)
            print(f"  {relative_path}")

    file_candidates = {
        "log": [
            "log_standard_4_08_to_4_21_1k.csv",
            "log_random_4_22_to_5_08_1k.csv",
            "kuairand-1k-test.csv",
        ],
        "user_features": [
            "user_features_pure.csv",
            "user-features-test.csv",
        ],
        "video_features_basic": [
            "video_features_basic_pure.csv",
        ],
        "video_features_statistic": [
            "video_features_statistic_pure.csv",
        ],
    }

    print("\nSample previews:")
    for label, candidates in file_candidates.items():
        file_path = _find_first_existing_file(dataset_path, candidates)
        if file_path is None:
            print(f"\n[{label}] file not found. candidates={candidates}")
            continue

        frame = pd.read_csv(file_path, nrows=max_rows)
        print(f"\n[{label}] {file_path}")
        print(f"shape(sample)={frame.shape}")
        print(f"columns={list(frame.columns)}")
        print(frame.head(max_rows).to_string(index=False))


if __name__ == "__main__":
    inspect_kuairand_dataset()
