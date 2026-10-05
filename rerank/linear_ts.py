from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def build_context(user_features, item_features) -> np.ndarray:
    """
    Build the contextual feature vector used by the bandit.

    This method is intentionally left as a placeholder so the project can
    later plug in user-side and item-side features that are appropriate for
    alpha selection.
    """
    raise NotImplementedError("build_context should be implemented with user/item features")


@dataclass
class LinearGaussianArmState:
    precision_matrix: np.ndarray
    precision_weighted_mean: np.ndarray


class LinearTS:
    """
    Linear Thompson Sampling for contextual bandits with Gaussian rewards.

    Each arm keeps an independent Bayesian linear regression posterior:

        theta_a ~ N(0, lambda_prior^-1 I)
        reward | x, theta_a ~ N(x^T theta_a, noise_variance)

    This implementation is appropriate when alpha is discretized into a
    finite candidate set and the observed reward is Gaussian-like.
    """

    def __init__(
        self,
        alpha_candidates: Sequence[float],
        context_dim: int,
        lambda_prior: float = 1.0,
        noise_variance: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        if not alpha_candidates:
            raise ValueError("alpha_candidates must not be empty")
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if lambda_prior <= 0:
            raise ValueError("lambda_prior must be positive")
        if noise_variance <= 0:
            raise ValueError("noise_variance must be positive")

        self.alpha_candidates = list(alpha_candidates)
        self.context_dim = context_dim
        self.lambda_prior = float(lambda_prior)
        self.noise_variance = float(noise_variance)
        self.rng = np.random.default_rng(random_state)

        prior_precision = self.lambda_prior * np.eye(self.context_dim, dtype=np.float64)
        prior_mean_term = np.zeros(self.context_dim, dtype=np.float64)
        self.arm_states = [
            LinearGaussianArmState(
                precision_matrix=prior_precision.copy(),
                precision_weighted_mean=prior_mean_term.copy(),
            )
            for _ in self.alpha_candidates
        ]

    def _validate_context(self, context: Iterable[float]) -> np.ndarray:
        context_array = np.asarray(context, dtype=np.float64).reshape(-1)
        if context_array.shape[0] != self.context_dim:
            raise ValueError(f"context must have shape ({self.context_dim},)")
        return context_array

    def _posterior_mean_and_covariance(self, arm_index: int) -> tuple[np.ndarray, np.ndarray]:
        state = self.arm_states[arm_index]
        covariance = np.linalg.inv(state.precision_matrix)
        mean = covariance @ state.precision_weighted_mean
        return mean, covariance

    def sample_theta(self, arm_index: int) -> np.ndarray:
        mean, covariance = self._posterior_mean_and_covariance(arm_index)
        return self.rng.multivariate_normal(mean=mean, cov=covariance)

    def score_arms(self, context: Iterable[float]) -> np.ndarray:
        context_array = self._validate_context(context)
        sampled_scores = np.zeros(len(self.alpha_candidates), dtype=np.float64)
        for arm_index in range(len(self.alpha_candidates)):
            theta_sample = self.sample_theta(arm_index)
            sampled_scores[arm_index] = float(context_array @ theta_sample)
        return sampled_scores

    def select_arm(self, context: Iterable[float]) -> int:
        sampled_scores = self.score_arms(context)
        return int(np.argmax(sampled_scores))

    def select_alpha(self, context: Iterable[float]) -> float:
        arm_index = self.select_arm(context)
        return self.alpha_candidates[arm_index]

    def update(self, arm_index: int, context: Iterable[float], reward: float) -> None:
        if arm_index < 0 or arm_index >= len(self.alpha_candidates):
            raise IndexError("arm_index out of range")

        context_array = self._validate_context(context)
        state = self.arm_states[arm_index]

        outer_product = np.outer(context_array, context_array) / self.noise_variance
        reward_term = (reward * context_array) / self.noise_variance

        state.precision_matrix += outer_product
        state.precision_weighted_mean += reward_term

    def update_alpha(self, alpha: float, context: Iterable[float], reward: float) -> None:
        arm_index = self.alpha_candidates.index(alpha)
        self.update(arm_index=arm_index, context=context, reward=reward)

    def posterior_mean(self, arm_index: int) -> np.ndarray:
        mean, _ = self._posterior_mean_and_covariance(arm_index)
        return mean

    def posterior_covariance(self, arm_index: int) -> np.ndarray:
        _, covariance = self._posterior_mean_and_covariance(arm_index)
        return covariance

    def get_alpha_candidates(self) -> list[float]:
        return list(self.alpha_candidates)
