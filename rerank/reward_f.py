from __future__ import annotations

from typing import Mapping


def compute_kuairand_reward(
    *,
    is_hate: int | float,
    play_time_ms: int | float,
    duration_ms: int | float,
    is_click: int | float,
    is_like: int | float,
    is_follow: int | float,
) -> float:
    """
    Compute KuaiRand reward for reranking / bandit update.

    Rule:
        if is_hate == 1:
            reward = -1
        else:
            reward = 0.2 * min(play_time_ms / duration_ms, 1)
                     + 0.5 * is_click
                     + 0.3 * int(is_like or is_follow)
    """
    if int(is_hate) == 1:
        return -1.0

    safe_duration_ms = max(float(duration_ms), 1.0)
    watch_ratio = min(float(play_time_ms) / safe_duration_ms, 1.0)
    social_feedback = 1.0 if (int(is_like) == 1 or int(is_follow) == 1) else 0.0

    reward = (
        0.2 * watch_ratio
        + 0.5 * float(is_click)
        + 0.3 * social_feedback
    )
    return float(reward)


def compute_kuairand_reward_from_record(record: Mapping[str, int | float]) -> float:
    """
    Compute reward from a dict-like KuaiRand interaction record.
    Required keys:
        is_hate, play_time_ms, duration_ms, is_click, is_like, is_follow
    """
    return compute_kuairand_reward(
        is_hate=record["is_hate"],
        play_time_ms=record["play_time_ms"],
        duration_ms=record["duration_ms"],
        is_click=record["is_click"],
        is_like=record["is_like"],
        is_follow=record["is_follow"],
    )
