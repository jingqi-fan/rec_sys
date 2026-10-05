from __future__ import annotations

import os
from collections import Counter

import numpy as np

from configs.model import OUTPUT_MODEL_PATH
from rerank.dpp import build_kernel_matrix, dpp
from rerank.linear_ts import LinearTS


TOP_K = 10
DPP_ALPHA = 1.0
ALPHA_CANDIDATES = [0.6, 0.8, 1.0, 1.2, 1.4]
RANDOM_SEED = 42

DATA_PATH = os.path.join(OUTPUT_MODEL_PATH, "kuairand_rerank_test_data.npz")
RESULT_PATH = os.path.join(OUTPUT_MODEL_PATH, "kuairand_rerank_test_result.txt")


def topk_by_score(candidate_scores: np.ndarray, top_k: int) -> np.ndarray:
    return np.argsort(-candidate_scores)[:top_k]


def topk_by_dpp(candidate_embeddings: np.ndarray, candidate_scores: np.ndarray, alpha: float, top_k: int) -> np.ndarray:
    kernel_matrix = build_kernel_matrix(candidate_embeddings, candidate_scores, alpha=alpha)
    selected = dpp(kernel_matrix, min(top_k, len(candidate_scores)))
    return np.asarray(selected, dtype=np.int64)


def dcg_at_k(relevance: np.ndarray, top_positions: np.ndarray, top_k: int) -> float:
    ranked_relevance = relevance[top_positions[:top_k]]
    discounts = 1.0 / np.log2(np.arange(len(ranked_relevance), dtype=np.float64) + 2.0)
    return float(np.sum(ranked_relevance * discounts))


def ndcg_at_k(relevance: np.ndarray, top_positions: np.ndarray, top_k: int) -> float:
    dcg = dcg_at_k(relevance, top_positions, top_k)
    ideal_positions = np.argsort(-relevance)[:top_k]
    ideal_dcg = dcg_at_k(relevance, ideal_positions, top_k)
    if ideal_dcg <= 1e-12:
        return 0.0
    return float(dcg / ideal_dcg)


def recall_at_k(binary_relevance: np.ndarray, top_positions: np.ndarray, top_k: int) -> float:
    total_relevant = float(np.sum(binary_relevance))
    if total_relevant <= 1e-12:
        return 0.0
    hit_relevant = float(np.sum(binary_relevance[top_positions[:top_k]]))
    return float(hit_relevant / total_relevant)


def hitrate_at_k(binary_relevance: np.ndarray, top_positions: np.ndarray, top_k: int) -> float:
    return float(np.any(binary_relevance[top_positions[:top_k]] > 0.0))


def ild_at_k(candidate_embeddings: np.ndarray, top_positions: np.ndarray, top_k: int) -> float:
    embeddings = candidate_embeddings[top_positions[:top_k]]
    if len(embeddings) < 2:
        return 0.0

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normalized_embeddings = embeddings / norms
    similarity_matrix = normalized_embeddings @ normalized_embeddings.T
    pairwise_similarities = similarity_matrix[np.triu_indices(len(embeddings), k=1)]
    return float(np.mean(1.0 - pairwise_similarities))


def category_entropy_at_k(candidate_tags: np.ndarray, top_positions: np.ndarray, top_k: int) -> float:
    tags = candidate_tags[top_positions[:top_k]]
    if len(tags) == 0:
        return 0.0

    _, counts = np.unique(tags, return_counts=True)
    probabilities = counts.astype(np.float64) / float(np.sum(counts))
    return float(-np.sum(probabilities * np.log(probabilities + 1e-12)))


def discounted_reward(graded_relevance: np.ndarray, top_positions: np.ndarray, top_k: int) -> float:
    rewards = graded_relevance[top_positions[:top_k]]
    discounts = 1.0 / np.log2(np.arange(len(rewards), dtype=np.float64) + 2.0)
    return float(np.sum(rewards * discounts))


def build_linear_ts_context(candidate_scores: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
    score_std = float(np.std(candidate_scores))
    score_gap = float(np.max(candidate_scores) - np.mean(candidate_scores))

    norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normalized_embeddings = candidate_embeddings / norms
    similarity_matrix = normalized_embeddings @ normalized_embeddings.T
    pairwise_similarities = similarity_matrix[np.triu_indices(len(candidate_embeddings), k=1)]

    if pairwise_similarities.size == 0:
        sim_mean = 0.0
        sim_max = 0.0
        sim_std = 0.0
    else:
        sim_mean = float(np.mean(pairwise_similarities))
        sim_max = float(np.max(pairwise_similarities))
        sim_std = float(np.std(pairwise_similarities))

    return np.asarray([score_std, score_gap, sim_mean, sim_max, sim_std], dtype=np.float64)


def evaluate_rankings(data, rankings: np.ndarray, method_name: str) -> dict:
    ndcgs = []
    recalls = []
    hitrates = []
    ilds = []
    entropies = []
    recommended_items = set()
    item_universe = set(data["candidate_ids"].reshape(-1).tolist())

    for row_index, top_positions in enumerate(rankings):
        candidate_ids = data["candidate_ids"][row_index]
        candidate_embeddings = data["candidate_embeddings"][row_index]
        candidate_tags = data["candidate_primary_tags"][row_index]
        binary_relevance = data["binary_relevance"][row_index]
        graded_relevance = data["graded_relevance"][row_index]

        top_positions = np.asarray(top_positions[:TOP_K], dtype=np.int64)
        recommended_items.update(candidate_ids[top_positions].tolist())

        ndcgs.append(ndcg_at_k(graded_relevance, top_positions, TOP_K))
        recalls.append(recall_at_k(binary_relevance, top_positions, TOP_K))
        hitrates.append(hitrate_at_k(binary_relevance, top_positions, TOP_K))
        ilds.append(ild_at_k(candidate_embeddings, top_positions, TOP_K))
        entropies.append(category_entropy_at_k(candidate_tags, top_positions, TOP_K))

    return {
        "method": method_name,
        "NDCG@10": float(np.mean(ndcgs)),
        "Recall@10": float(np.mean(recalls)),
        "HitRate@10": float(np.mean(hitrates)),
        "ILD@10": float(np.mean(ilds)),
        "Coverage@10": float(len(recommended_items) / max(len(item_universe), 1)),
        "CategoryEntropy@10": float(np.mean(entropies)),
    }


def build_original_rankings(data) -> np.ndarray:
    return np.stack([topk_by_score(scores, TOP_K) for scores in data["candidate_scores"]])


def build_dpp_rankings(data, alpha: float) -> np.ndarray:
    rankings = []
    for row_index in range(data["candidate_scores"].shape[0]):
        rankings.append(
            topk_by_dpp(
                data["candidate_embeddings"][row_index],
                data["candidate_scores"][row_index],
                alpha=alpha,
                top_k=TOP_K,
            )
        )
    return np.stack(rankings)


def build_linear_ts_rankings(data) -> tuple[np.ndarray, Counter]:
    bandit = LinearTS(
        alpha_candidates=ALPHA_CANDIDATES,
        context_dim=5,
        lambda_prior=1.0,
        noise_variance=1.0,
        random_state=RANDOM_SEED,
    )
    rankings = []
    alpha_counter = Counter()

    for row_index in range(data["candidate_scores"].shape[0]):
        candidate_scores = data["candidate_scores"][row_index]
        candidate_embeddings = data["candidate_embeddings"][row_index]
        graded_relevance = data["graded_relevance"][row_index]

        context = build_linear_ts_context(candidate_scores, candidate_embeddings)
        alpha = bandit.select_alpha(context)
        alpha_counter[alpha] += 1

        top_positions = topk_by_dpp(candidate_embeddings, candidate_scores, alpha=alpha, top_k=TOP_K)
        reward = discounted_reward(graded_relevance, top_positions, TOP_K)
        bandit.update_alpha(alpha, context, reward)
        rankings.append(top_positions)

    return np.stack(rankings), alpha_counter


def format_results(results: list[dict], alpha_counter: Counter | None, num_examples: int) -> str:
    lines = []
    lines.append("KuaiRand rerank offline test")
    lines.append(f"data_path: {DATA_PATH}")
    lines.append(f"num_examples: {num_examples}")
    lines.append(f"top_k: {TOP_K}")
    lines.append("")
    lines.append("Methods:")
    lines.append("Original: no reranking, sort by candidate_scores descending.")
    lines.append(f"DPP: fixed alpha={DPP_ALPHA}.")
    lines.append(f"DPP+LinearTS: alpha candidates={ALPHA_CANDIDATES}.")
    lines.append("")

    headers = ["Method", "NDCG@10", "Recall@10", "HitRate@10", "ILD@10", "Coverage@10", "CategoryEntropy@10"]
    rows = []
    for result in results:
        rows.append(
            [
                result["method"],
                f"{result['NDCG@10']:.6f}",
                f"{result['Recall@10']:.6f}",
                f"{result['HitRate@10']:.6f}",
                f"{result['ILD@10']:.6f}",
                f"{result['Coverage@10']:.6f}",
                f"{result['CategoryEntropy@10']:.6f}",
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    lines.append("Results:")
    lines.append(" | ".join(header.ljust(width) for header, width in zip(headers, widths)))
    lines.append("-+-".join("-" * width for width in widths))
    for row in rows:
        lines.append(" | ".join(cell.ljust(width) for cell, width in zip(row, widths)))

    if alpha_counter is not None:
        lines.append("")
        lines.append(f"DPP+LinearTS alpha usage: {dict(sorted(alpha_counter.items()))}")

    lines.append("")
    lines.append("Notes:")
    lines.append("Coverage@10 is computed over unique items appearing in this prepared candidate set.")
    lines.append("NDCG@10 uses graded_relevance; Recall@10 and HitRate@10 use binary_relevance.")
    return "\n".join(lines) + "\n"


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing prepared data: {DATA_PATH}. Run kuairand_re_rank_test_prepare_data.py first.")

    data = np.load(DATA_PATH)
    original_rankings = build_original_rankings(data)
    dpp_rankings = build_dpp_rankings(data, alpha=DPP_ALPHA)
    linear_ts_rankings, alpha_counter = build_linear_ts_rankings(data)

    results = [
        evaluate_rankings(data, original_rankings, "Original"),
        evaluate_rankings(data, dpp_rankings, f"DPP(alpha={DPP_ALPHA})"),
        evaluate_rankings(data, linear_ts_rankings, "DPP+LinearTS"),
    ]

    result_text = format_results(results, alpha_counter, num_examples=data["candidate_ids"].shape[0])
    os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        f.write(result_text)

    print(result_text)
    print(f"Saved result to {RESULT_PATH}")


if __name__ == "__main__":
    main()
