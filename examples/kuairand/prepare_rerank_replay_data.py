import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from configs.model import OUTPUT_MODEL_PATH
from data.dataset_manager import DatasetType, dataset_manager
from feature.kuairand_features import extract_tag
from rerank.reward_f import compute_kuairand_reward_from_record


MIN_CLICK_HISTORY = 10
HISTORY_SIZE = 20
NUM_USERS = 20
MAX_EVENTS_PER_USER = 50
CANDIDATE_SIZE = 50
RANDOM_SEED = 42

OUTPUT_DATA_PATH = os.path.join(OUTPUT_MODEL_PATH, "kuairand_rerank_test_data.npz")
OUTPUT_META_PATH = os.path.join(OUTPUT_MODEL_PATH, "kuairand_rerank_test_data_meta.json")


def load_embedding_store():
    embeddings = torch.load(os.path.join(OUTPUT_MODEL_PATH, "kuairand_item_embeddings.pt"), weights_only=False)
    video_ids = np.load(os.path.join(OUTPUT_MODEL_PATH, "kuairand_item_ids.npy"))
    embeddings = np.asarray(embeddings, dtype=np.float64)
    video_ids = np.asarray(video_ids, dtype=np.int64)
    return {int(video_id): embeddings[idx] for idx, video_id in enumerate(video_ids.tolist())}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm <= 1e-12 or b_norm <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def score_candidates(candidate_ids, history_item_ids, embedding_store):
    history_embeddings = [embedding_store[item_id] for item_id in history_item_ids if item_id in embedding_store]
    if not history_embeddings:
        return np.zeros(len(candidate_ids), dtype=np.float64)

    user_profile = np.mean(np.asarray(history_embeddings, dtype=np.float64), axis=0)
    scores = []
    for candidate_id in candidate_ids:
        candidate_embedding = embedding_store.get(candidate_id)
        if candidate_embedding is None:
            scores.append(0.0)
        else:
            scores.append(cosine_similarity(user_profile, candidate_embedding))
    return np.asarray(scores, dtype=np.float64)


def pad_history(history_item_ids, max_length):
    history_window = list(history_item_ids)[-max_length:]
    padded = np.zeros(max_length, dtype=np.int64)
    if history_window:
        padded[-len(history_window) :] = np.asarray(history_window, dtype=np.int64)
    return padded


def load_kuairand_tables():
    cached_path = dataset_manager.get_dataset(DatasetType.KUAI_RAND_1K)
    log = pd.read_csv(os.path.join(cached_path, "log_standard_4_08_to_4_21_1k.csv"))
    video_features = pd.read_csv(os.path.join(cached_path, "video_features_basic_1k.csv"))

    visible_videos = video_features[
        (video_features["visible_status"] == 1) & (video_features["tag"].notna())
    ][["video_id", "author_id", "tag"]].copy()
    visible_videos["tag_list"] = visible_videos["tag"].apply(extract_tag)
    visible_videos["primary_tag"] = visible_videos["tag_list"].apply(lambda tags: int(tags[0]) if tags else 0)
    visible_videos = visible_videos.drop_duplicates(subset=["video_id"]).set_index("video_id")

    log = log.merge(visible_videos.reset_index()[["video_id"]], on="video_id", how="inner")
    log = log.sort_values(["user_id", "time_ms"]).reset_index(drop=True)
    return log, visible_videos


def sample_candidates(target_item_id, item_pool, rng):
    negative_pool = item_pool[item_pool != target_item_id]
    sample_size = min(CANDIDATE_SIZE - 1, len(negative_pool))
    sampled_negatives = rng.choice(negative_pool, size=sample_size, replace=False)
    candidate_ids = np.concatenate([np.asarray([target_item_id], dtype=np.int64), sampled_negatives.astype(np.int64)])
    rng.shuffle(candidate_ids)
    return candidate_ids


def build_examples(log, video_features, embedding_store, rng):
    item_pool = np.asarray(
        sorted(set(embedding_store.keys()).intersection(set(video_features.index.tolist()))),
        dtype=np.int64,
    )
    if len(item_pool) < CANDIDATE_SIZE:
        raise ValueError(f"Need at least {CANDIDATE_SIZE} embedded visible items, got {len(item_pool)}")

    examples = []
    selected_user_ids = []
    events_by_user = defaultdict(int)

    for user_id, user_log in log.groupby("user_id", sort=False):
        clicked_history = []

        for row in user_log.itertuples(index=False):
            target_item_id = int(row.video_id)

            if len(clicked_history) >= MIN_CLICK_HISTORY and target_item_id in embedding_store:
                target_record = {
                    "is_hate": int(row.is_hate),
                    "play_time_ms": float(row.play_time_ms),
                    "duration_ms": float(row.duration_ms),
                    "is_click": int(row.is_click),
                    "is_like": int(row.is_like),
                    "is_follow": int(row.is_follow),
                }
                target_reward = compute_kuairand_reward_from_record(target_record)
                candidate_ids = sample_candidates(target_item_id, item_pool, rng)
                candidate_scores = score_candidates(candidate_ids, clicked_history, embedding_store)
                candidate_embeddings = np.asarray([embedding_store[int(item_id)] for item_id in candidate_ids], dtype=np.float64)
                candidate_primary_tags = np.asarray(
                    [int(video_features.loc[int(item_id), "primary_tag"]) for item_id in candidate_ids],
                    dtype=np.int64,
                )

                binary_relevance = np.zeros(CANDIDATE_SIZE, dtype=np.float64)
                graded_relevance = np.zeros(CANDIDATE_SIZE, dtype=np.float64)
                target_position = int(np.where(candidate_ids == target_item_id)[0][0])
                binary_relevance[target_position] = 1.0 if target_reward > 0.0 else 0.0
                graded_relevance[target_position] = max(float(target_reward), 0.0)

                examples.append(
                    {
                        "user_id": int(user_id),
                        "time_ms": int(row.time_ms),
                        "history_item_ids": pad_history(clicked_history, HISTORY_SIZE),
                        "history_length": min(len(clicked_history), HISTORY_SIZE),
                        "target_item_id": target_item_id,
                        "target_reward": float(target_reward),
                        "candidate_ids": candidate_ids,
                        "candidate_scores": candidate_scores,
                        "candidate_embeddings": candidate_embeddings,
                        "candidate_primary_tags": candidate_primary_tags,
                        "binary_relevance": binary_relevance,
                        "graded_relevance": graded_relevance,
                    }
                )

                events_by_user[int(user_id)] += 1
                if events_by_user[int(user_id)] >= MAX_EVENTS_PER_USER:
                    break

            if int(row.is_click) == 1 and target_item_id in embedding_store:
                clicked_history.append(target_item_id)

        if events_by_user[int(user_id)] > 0:
            selected_user_ids.append(int(user_id))
        if len(selected_user_ids) >= NUM_USERS:
            break

    if not examples:
        raise ValueError("No examples generated. Try reducing MIN_CLICK_HISTORY.")

    return examples, selected_user_ids


def stack_examples(examples):
    return {
        "user_ids": np.asarray([example["user_id"] for example in examples], dtype=np.int64),
        "time_ms": np.asarray([example["time_ms"] for example in examples], dtype=np.int64),
        "history_item_ids": np.stack([example["history_item_ids"] for example in examples]),
        "history_lengths": np.asarray([example["history_length"] for example in examples], dtype=np.int64),
        "target_item_ids": np.asarray([example["target_item_id"] for example in examples], dtype=np.int64),
        "target_rewards": np.asarray([example["target_reward"] for example in examples], dtype=np.float64),
        "candidate_ids": np.stack([example["candidate_ids"] for example in examples]),
        "candidate_scores": np.stack([example["candidate_scores"] for example in examples]),
        "candidate_embeddings": np.stack([example["candidate_embeddings"] for example in examples]),
        "candidate_primary_tags": np.stack([example["candidate_primary_tags"] for example in examples]),
        "binary_relevance": np.stack([example["binary_relevance"] for example in examples]),
        "graded_relevance": np.stack([example["graded_relevance"] for example in examples]),
    }


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    embedding_store = load_embedding_store()
    log, video_features = load_kuairand_tables()
    examples, selected_user_ids = build_examples(log, video_features, embedding_store, rng)
    arrays = stack_examples(examples)

    os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)
    np.savez_compressed(OUTPUT_DATA_PATH, **arrays)

    meta = {
        "description": "Small offline KuaiRand reranking test set. Each row has a target exposure plus randomly sampled negative candidates.",
        "output_data_path": OUTPUT_DATA_PATH,
        "num_examples": len(examples),
        "selected_user_ids": selected_user_ids,
        "min_click_history": MIN_CLICK_HISTORY,
        "history_size": HISTORY_SIZE,
        "num_users": NUM_USERS,
        "max_events_per_user": MAX_EVENTS_PER_USER,
        "candidate_size": CANDIDATE_SIZE,
        "random_seed": RANDOM_SEED,
        "fields": {
            "candidate_scores": "Baseline ranking score: cosine(user clicked-history mean embedding, candidate embedding).",
            "binary_relevance": "Only the logged target item can be relevant. It is 1 when target_reward > 0, otherwise 0.",
            "graded_relevance": "Only the logged target item can be relevant. It is max(target_reward, 0). Use this for NDCG if desired.",
            "candidate_embeddings": "Use for ILD@10.",
            "candidate_primary_tags": "Use for Category Entropy@10.",
            "candidate_ids": "Use recommended top-k ids for Coverage@10.",
        },
        "recommended_baseline": "Sort each row by candidate_scores descending and take top 10.",
    }
    with open(OUTPUT_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {len(examples)} examples to {OUTPUT_DATA_PATH}")
    print(f"Saved metadata to {OUTPUT_META_PATH}")
    print(f"Selected users: {selected_user_ids}")


if __name__ == "__main__":
    main()
