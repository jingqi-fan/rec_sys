import os
from collections import Counter, defaultdict

import numpy as np
import torch

from configs.model import OUTPUT_MODEL_PATH
from data.dataset_kuairand import prepare_kuairand_rerank_replay_data
from feature.kuairand_features_re_rank import build_kuairand_linear_ts_context
from rerank.dpp import build_kernel_matrix, dpp
from rerank.linear_ts import LinearTS
from rerank.reward_f import compute_kuairand_reward_from_record

# Offline replay script that simulates online reranking decisions.

ALPHA_CANDIDATES = [0.6, 0.8, 1.0, 1.2, 1.4]
NUM_USERS = 4
MIN_CLICK_HISTORY = 10
MAX_EVENTS_PER_USER = 100
CANDIDATE_SIZE = 20
RERANK_TOP_K = 10
RANDOM_SEED = 42


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
            continue
        scores.append(cosine_similarity(user_profile, candidate_embedding))
    return np.asarray(scores, dtype=np.float64)


def generate_candidates(target_item_id, embedding_store, rng, candidate_size):
    pool = [item_id for item_id in embedding_store.keys() if item_id != target_item_id]
    sample_size = max(candidate_size - 1, 0)
    sampled_negatives = rng.choice(pool, size=min(sample_size, len(pool)), replace=False).tolist()
    candidate_ids = [target_item_id] + sampled_negatives
    rng.shuffle(candidate_ids)
    return candidate_ids


def compute_discounted_reward(target_reward, reranked_item_ids, target_item_id):
    if target_item_id not in reranked_item_ids:
        return 0.0
    rank = reranked_item_ids.index(target_item_id)
    return float(target_reward / np.log2(rank + 2.0))


if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_SEED)
    embedding_store = load_embedding_store()
    replay_events, video_features, selected_user_ids = prepare_kuairand_rerank_replay_data(
        num_users=NUM_USERS,
        min_click_history=MIN_CLICK_HISTORY,
        max_events_per_user=MAX_EVENTS_PER_USER,
    )

    bandit = LinearTS(
        alpha_candidates=ALPHA_CANDIDATES,
        context_dim=5,
        lambda_prior=1.0,
        noise_variance=1.0,
        random_state=RANDOM_SEED,
    )

    alpha_counter = Counter()
    reward_by_user = defaultdict(list)
    hit_by_user = defaultdict(list)
    total_events = 0

    for event in replay_events:
        history_item_ids = [item_id for item_id in event["history_item_ids"] if item_id in embedding_store]
        if len(history_item_ids) < MIN_CLICK_HISTORY:
            continue
        if event["target_item_id"] not in embedding_store:
            continue

        context = build_kuairand_linear_ts_context(
            history_item_ids=history_item_ids,
            video_features=video_features,
            item_embeddings=embedding_store,
            history_size=MIN_CLICK_HISTORY,
        )
        alpha = bandit.select_alpha(context)
        alpha_counter[alpha] += 1

        candidate_ids = generate_candidates(
            target_item_id=event["target_item_id"],
            embedding_store=embedding_store,
            rng=rng,
            candidate_size=CANDIDATE_SIZE,
        )
        candidate_embeddings = np.asarray([embedding_store[item_id] for item_id in candidate_ids], dtype=np.float64)
        candidate_scores = score_candidates(candidate_ids, history_item_ids, embedding_store)

        kernel_matrix = build_kernel_matrix(candidate_embeddings, candidate_scores, alpha=alpha)
        reranked_positions = dpp(kernel_matrix, min(RERANK_TOP_K, len(candidate_ids)))
        reranked_item_ids = [candidate_ids[position] for position in reranked_positions]

        target_reward = compute_kuairand_reward_from_record(event["target_record"])
        realized_reward = compute_discounted_reward(target_reward, reranked_item_ids, event["target_item_id"])
        bandit.update_alpha(alpha, context, realized_reward)

        reward_by_user[event["user_id"]].append(realized_reward)
        hit_by_user[event["user_id"]].append(1.0 if event["target_item_id"] in reranked_item_ids else 0.0)
        total_events += 1

    avg_reward = (
        float(np.mean([reward for rewards in reward_by_user.values() for reward in rewards]))
        if reward_by_user
        else 0.0
    )
    avg_hit_rate = (
        float(np.mean([hit for hits in hit_by_user.values() for hit in hits]))
        if hit_by_user
        else 0.0
    )
    avg_reward_by_user = {
        user_id: float(np.mean(rewards))
        for user_id, rewards in reward_by_user.items()
        if rewards
    }

    print(f"Selected users: {selected_user_ids}")
    print(f"Processed replay events: {total_events}")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Average hit rate@{RERANK_TOP_K}: {avg_hit_rate:.4f}")
    print(f"Alpha usage: {dict(alpha_counter)}")
    print(f"Average reward by user: {avg_reward_by_user}")
