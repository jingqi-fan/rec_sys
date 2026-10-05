import numpy as np
import math
from typing import List


def build_kernel_matrix(item_embeddings: np.ndarray, item_scores: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Build a DPP kernel matrix from item embeddings and ranking scores.

    The scores are transformed into positive quality weights because DPP
    expects a positive semi-definite kernel with positive diagonal entries.
    alpha controls the tradeoff between relevance and diversity.
    """
    embeddings = np.asarray(item_embeddings, dtype=np.float64)
    scores = np.asarray(item_scores, dtype=np.float64).reshape(-1)

    if embeddings.ndim != 2:
        raise ValueError("item_embeddings must be a 2-d array")
    if scores.ndim != 1:
        raise ValueError("item_scores must be a 1-d array")
    if embeddings.shape[0] != scores.shape[0]:
        raise ValueError("item_embeddings and item_scores must have the same number of items")

    embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embedding_norms = np.maximum(embedding_norms, 1e-12)
    normalized_embeddings = embeddings / embedding_norms
    similarities = np.matmul(normalized_embeddings, normalized_embeddings.T)

    stabilized_scores = scores - np.max(scores)
    quality_scores = np.exp(stabilized_scores)
    quality_scores = np.power(quality_scores, alpha)
    return quality_scores[:, None] * similarities * quality_scores[None, :]


def dpp(kernel_matrix: np.ndarray, max_length: int, epsilon: float = 1e-10) -> List[int]:
    """
    Our proposed fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items


def dpp_sw(kernel_matrix: np.ndarray, window_size: int, max_length: int, epsilon: float = 1e-10) -> List[int]:
    """
    Sliding window version of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param window_size: positive int
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    v = np.zeros((max_length, max_length))
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    window_left_index = 0
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[window_left_index:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        v[k, window_left_index:k] = ci_optimal
        v[k, k] = di_optimal
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[window_left_index:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        if len(selected_items) >= window_size:
            window_left_index += 1
            for ind in range(window_left_index, k + 1):
                t = math.sqrt(v[ind, ind] ** 2 + v[ind, window_left_index - 1] ** 2)
                c = t / v[ind, ind]
                s = v[ind, window_left_index - 1] / v[ind, ind]
                v[ind, ind] = t
                v[ind + 1:k + 1, ind] += s * v[ind + 1:k + 1, window_left_index - 1]
                v[ind + 1:k + 1, ind] /= c
                v[ind + 1:k + 1, window_left_index - 1] *= c
                v[ind + 1:k + 1, window_left_index - 1] -= s * v[ind + 1:k + 1, ind]
                cis[ind, :] += s * cis[window_left_index - 1, :]
                cis[ind, :] /= c
                cis[window_left_index - 1, :] *= c
                cis[window_left_index - 1, :] -= s * cis[ind, :]
            di2s += np.square(cis[window_left_index - 1, :])
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items
