from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from configs.model import OUTPUT_MODEL_PATH
from data.dataset_manager import DatasetType, dataset_manager
from feature.kuairand_features import (
    FANS_USER_NUM_RANGE,
    FOLLOW_USER_NUM_RANGE,
    FRIEND_USER_NUM_RANGE,
    REGISTER_DAYS_RANGE,
    USER_ACTIVE_DEGREE,
    extract_tag,
)


class KuairandTwoTowerDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        # for user tower
        self.users = torch.LongTensor(data["user_id"].values)
        self.user_active_degrees = torch.LongTensor(data["user_active_degree"].values)
        self.follow_user_num_ranges = torch.LongTensor(data["follow_user_num_range"].values)
        self.fans_user_num_ranges = torch.LongTensor(data["fans_user_num_range"].values)
        self.friend_user_num_ranges = torch.LongTensor(data["friend_user_num_range"].values)
        self.register_days_ranges = torch.LongTensor(data["register_days_range"].values)

        # for item tower
        self.videos = torch.LongTensor(data["video_id"].values)
        self.authors = torch.LongTensor(data["author_id"].values)
        self.music_types = torch.LongTensor(data["music_type"].values)
        self.video_durations = torch.FloatTensor(data["video_duration"].values)
        self.tags = torch.LongTensor(data["tag_padded"].tolist())

        # label
        self.labels = torch.FloatTensor(data["label"].values)

    def __len__(self):
        return self.users.shape[0]

    def __getitem__(self, index):
        feature = {
            "user_id": self.users[index],
            "user_active_degree": self.user_active_degrees[index],
            "follow_user_num_range": self.follow_user_num_ranges[index],
            "fans_user_num_range": self.fans_user_num_ranges[index],
            "friend_user_num_range": self.friend_user_num_ranges[index],
            "register_days_range": self.register_days_ranges[index],
            "video_id": self.videos[index],
            "author_id": self.authors[index],
            "music_type": self.music_types[index],
            "video_duration": self.video_durations[index],
            "tag": self.tags[index],
        }
        return feature, self.labels[index]

    @classmethod
    def prepare_data(cls):
        """
        Prepare KuaiRand-1K data for two-tower training.

        Steps:
          1. load raw logs/user/video tables
          2. keep click interactions only
          3. encode user categorical features to bucket ids
          4. keep visible videos only
          5. parse and pad tag list into fixed-length ids
        """
        cached_path = dataset_manager.get_dataset(DatasetType.KUAI_RAND_1K)
        log_kuairand = pd.read_csv(os.path.join(cached_path, "log_standard_4_08_to_4_21_1k.csv"))
        user_f = pd.read_csv(os.path.join(cached_path, "user_features_1k.csv"))
        video_f_basic = pd.read_csv(os.path.join(cached_path, "video_features_basic_1k.csv"))

        # 1) click interactions only
        click_log = log_kuairand[log_kuairand["is_click"] == 1][["user_id", "video_id", "time_ms"]].copy()
        click_log["label"] = 1.0

        # 2) user feature bucket encoding
        user_features = user_f[
            [
                "user_id",
                "user_active_degree",
                "follow_user_num_range",
                "fans_user_num_range",
                "friend_user_num_range",
                "register_days_range",
            ]
        ].copy()
        user_features["user_active_degree"] = (
            user_features["user_active_degree"].fillna("UNKNOWN").astype(str).str.strip()
        )
        user_features.loc[
            ~user_features["user_active_degree"].isin(USER_ACTIVE_DEGREE),
            "user_active_degree",
        ] = "UNKNOWN"
        user_features["fans_user_num_range"] = (
            user_features["fans_user_num_range"].fillna("").astype(str).str.strip()
        )
        user_features = user_features[user_features["fans_user_num_range"].isin(FANS_USER_NUM_RANGE)].copy()
        user_features["user_active_degree"] = pd.Categorical(
            user_features["user_active_degree"], categories=USER_ACTIVE_DEGREE, ordered=True
        ).codes
        user_features["follow_user_num_range"] = pd.Categorical(
            user_features["follow_user_num_range"], categories=FOLLOW_USER_NUM_RANGE, ordered=True
        ).codes
        user_features["fans_user_num_range"] = pd.Categorical(
            user_features["fans_user_num_range"], categories=FANS_USER_NUM_RANGE, ordered=True
        ).codes
        user_features["friend_user_num_range"] = pd.Categorical(
            user_features["friend_user_num_range"], categories=FRIEND_USER_NUM_RANGE, ordered=True
        ).codes
        user_features["register_days_range"] = pd.Categorical(
            user_features["register_days_range"], categories=REGISTER_DAYS_RANGE, ordered=True
        ).codes

        # 3) visible videos only (visible_status == 1 in raw table)
        visible_videos = video_f_basic[video_f_basic["visible_status"] == 1][
            ["video_id", "author_id", "music_type", "video_duration", "tag"]
        ].copy()
        visible_videos["music_type"] = visible_videos["music_type"].fillna(-1).astype(int) + 1
        visible_videos["video_duration"] = visible_videos["video_duration"].fillna(0.0).astype(float)

        # 4) parse and pad tag list
        visible_videos["tag"] = visible_videos["tag"].apply(extract_tag)
        max_tag_len = int(visible_videos["tag"].apply(len).max()) if not visible_videos.empty else 1
        visible_videos["tag_padded"] = visible_videos["tag"].apply(
            lambda tags: tags + [0] * (max_tag_len - len(tags))
        )

        # 5) merge to get training rows
        data = click_log.merge(user_features, on="user_id", how="inner").merge(
            visible_videos, on="video_id", how="inner"
        )
        data = data.drop(columns=["time_ms"]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        return data


def prepare_kuairand_two_tower_dataset(eval_ratio=0.1, dataset_type=DatasetType.KUAI_RAND_1K):
    if dataset_type != DatasetType.KUAI_RAND_1K:
        raise ValueError("prepare_kuairand_two_tower_dataset currently only supports KUAI_RAND_1K")

    data = KuairandTwoTowerDataset.prepare_data()

    eval_data = data.sample(frac=eval_ratio, random_state=42)
    train_data = data.drop(index=eval_data.index)
    train_data = train_data.reset_index(drop=True)
    eval_data = eval_data.reset_index(drop=True)
    return KuairandTwoTowerDataset(train_data), KuairandTwoTowerDataset(eval_data)


class KuairandDINDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self.data = data.reset_index(drop=True)

    @classmethod
    def prepare_data(cls, history_seq_length: int = 20):
        if history_seq_length <= 0:
            raise ValueError("history_seq_length must be positive")

        cached_path = dataset_manager.get_dataset(DatasetType.KUAI_RAND_1K)
        log_kuairand = pd.read_csv(os.path.join(cached_path, "log_standard_4_08_to_4_21_1k.csv"))
        user_f = pd.read_csv(os.path.join(cached_path, "user_features_1k.csv"))
        video_f_basic = pd.read_csv(os.path.join(cached_path, "video_features_basic_1k.csv"))

        user_features = user_f[
            [
                "user_id",
                "user_active_degree",
                "follow_user_num_range",
                "fans_user_num_range",
                "friend_user_num_range",
                "register_days_range",
            ]
        ].copy()
        user_features["user_active_degree"] = (
            user_features["user_active_degree"].fillna("UNKNOWN").astype(str).str.strip()
        )
        user_features.loc[
            ~user_features["user_active_degree"].isin(USER_ACTIVE_DEGREE),
            "user_active_degree",
        ] = "UNKNOWN"
        user_features["fans_user_num_range"] = (
            user_features["fans_user_num_range"].fillna("").astype(str).str.strip()
        )
        user_features = user_features[user_features["fans_user_num_range"].isin(FANS_USER_NUM_RANGE)].copy()
        user_features["user_active_degree"] = pd.Categorical(
            user_features["user_active_degree"], categories=USER_ACTIVE_DEGREE, ordered=True
        ).codes
        user_features["follow_user_num_range"] = pd.Categorical(
            user_features["follow_user_num_range"], categories=FOLLOW_USER_NUM_RANGE, ordered=True
        ).codes
        user_features["fans_user_num_range"] = pd.Categorical(
            user_features["fans_user_num_range"], categories=FANS_USER_NUM_RANGE, ordered=True
        ).codes
        user_features["friend_user_num_range"] = pd.Categorical(
            user_features["friend_user_num_range"], categories=FRIEND_USER_NUM_RANGE, ordered=True
        ).codes
        user_features["register_days_range"] = pd.Categorical(
            user_features["register_days_range"], categories=REGISTER_DAYS_RANGE, ordered=True
        ).codes

        visible_videos = video_f_basic[video_f_basic["visible_status"] == 1][
            ["video_id", "author_id", "tag"]
        ].copy()
        visible_videos["tag"] = visible_videos["tag"].apply(extract_tag)

        log_kuairand = log_kuairand.merge(visible_videos, on="video_id", how="inner")
        log_kuairand = log_kuairand.sort_values(["user_id", "time_ms"]).reset_index(drop=True)

        # Build click history before each exposure event; current event label is current click.
        records = []
        max_tag_len = int(visible_videos["tag"].apply(len).max()) if not visible_videos.empty else 1

        for _, user_log in log_kuairand.groupby("user_id", sort=False):
            clicked_history = []
            for row in user_log.itertuples(index=False):
                if len(clicked_history) == 0:
                    if row.is_click == 1:
                        clicked_history.append(
                            {
                                "video_id": int(row.video_id),
                                "author_id": int(row.author_id),
                                "tag": list(row.tag),
                            }
                        )
                    continue

                history_window = clicked_history[-history_seq_length:]
                history_length = len(history_window)
                history_video_ids = [item["video_id"] for item in history_window]
                history_author_ids = [item["author_id"] for item in history_window]
                history_tag_ids = [item["tag"] + [0] * (max_tag_len - len(item["tag"])) for item in history_window]

                history_video_ids = history_video_ids + [0] * (history_seq_length - history_length)
                history_author_ids = history_author_ids + [0] * (history_seq_length - history_length)
                history_tag_ids = history_tag_ids + [[0] * max_tag_len for _ in range(history_seq_length - history_length)]

                candidate_tag_ids = list(row.tag)
                candidate_tag_ids = candidate_tag_ids + [0] * (max_tag_len - len(candidate_tag_ids))

                records.append(
                    {
                        "user_id": int(row.user_id),
                        "user_active_degree": None,
                        "follow_user_num_range": None,
                        "fans_user_num_range": None,
                        "friend_user_num_range": None,
                        "register_days_range": None,
                        "candidate_video_id": int(row.video_id),
                        "candidate_author_id": int(row.author_id),
                        "candidate_tag_ids": candidate_tag_ids,
                        "history_video_ids": history_video_ids,
                        "history_author_ids": history_author_ids,
                        "history_tag_ids": history_tag_ids,
                        "history_length": history_length,
                        "label": float(row.is_click),
                    }
                )

                if row.is_click == 1:
                    clicked_history.append(
                        {
                            "video_id": int(row.video_id),
                            "author_id": int(row.author_id),
                            "tag": list(row.tag),
                        }
                    )

        data = pd.DataFrame(records)
        if data.empty:
            raise ValueError("No KuaiRand DIN samples were generated. Check history_seq_length and input data.")

        data = data.merge(user_features, on="user_id", how="inner", suffixes=("", "_user"))
        # Fill placeholders written before merge with merged columns.
        for column in [
            "user_active_degree",
            "follow_user_num_range",
            "fans_user_num_range",
            "friend_user_num_range",
            "register_days_range",
        ]:
            user_col = f"{column}_user"
            if user_col in data.columns:
                data[column] = data[user_col]
                data = data.drop(columns=[user_col])

        return data.reset_index(drop=True)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        row = self.data.iloc[index]
        feature = {
            "user_id": torch.tensor(row["user_id"], dtype=torch.long),
            "user_active_degree": torch.tensor(row["user_active_degree"], dtype=torch.long),
            "follow_user_num_range": torch.tensor(row["follow_user_num_range"], dtype=torch.long),
            "fans_user_num_range": torch.tensor(row["fans_user_num_range"], dtype=torch.long),
            "friend_user_num_range": torch.tensor(row["friend_user_num_range"], dtype=torch.long),
            "register_days_range": torch.tensor(row["register_days_range"], dtype=torch.long),
            "candidate_video_id": torch.tensor(row["candidate_video_id"], dtype=torch.long),
            "candidate_author_id": torch.tensor(row["candidate_author_id"], dtype=torch.long),
            "candidate_tag_ids": torch.tensor(row["candidate_tag_ids"], dtype=torch.long),
            "history_video_ids": torch.tensor(row["history_video_ids"], dtype=torch.long),
            "history_author_ids": torch.tensor(row["history_author_ids"], dtype=torch.long),
            "history_tag_ids": torch.tensor(row["history_tag_ids"], dtype=torch.long),
            "history_length": torch.tensor(row["history_length"], dtype=torch.long),
        }
        return feature, torch.tensor(row["label"], dtype=torch.float32)


def prepare_kuairand_din_dataset(history_seq_length: int = 20, eval_ratio: float = 0.1):
    data = KuairandDINDataset.prepare_data(history_seq_length=history_seq_length)
    eval_data = data.sample(frac=eval_ratio, random_state=42)
    train_data = data.drop(index=eval_data.index)
    train_data = train_data.reset_index(drop=True)
    eval_data = eval_data.reset_index(drop=True)
    return KuairandDINDataset(train_data), KuairandDINDataset(eval_data)


def prepare_kuairand_rerank_replay_data(
    num_users: int = 5,
    min_click_history: int = 50,
    max_events_per_user: int = 5000,
):
    """
    Prepare replay events for linear-TS + DPP reranking experiments.

    Each replay event represents a single online step:
        - a user with click history
        - the currently exposed target item
        - the reward fields attached to that exposure

    Returns:
        replay_events: list[dict]
        video_features: DataFrame indexed by video_id with at least author_id/tag
        user_ids: list[int]
    """
    if num_users <= 0:
        raise ValueError("num_users must be positive")
    if min_click_history <= 0:
        raise ValueError("min_click_history must be positive")
    if max_events_per_user <= 0:
        raise ValueError("max_events_per_user must be positive")

    cached_path = dataset_manager.get_dataset(DatasetType.KUAI_RAND_1K)
    log_kuairand = pd.read_csv(os.path.join(cached_path, "log_standard_4_08_to_4_21_1k.csv"))
    video_f_basic = pd.read_csv(os.path.join(cached_path, "video_features_basic_1k.csv"))

    video_features = video_f_basic[
        (video_f_basic["visible_status"] == 1) & (video_f_basic["tag"].notna())
    ][["video_id", "author_id", "tag"]].copy()
    video_features["tag"] = video_features["tag"].astype(str).str.strip()
    video_features = video_features[video_features["tag"] != ""].copy()
    video_features["tag"] = video_features["tag"].apply(extract_tag)
    video_features = video_features.drop_duplicates(subset=["video_id"]).set_index("video_id")

    log_kuairand = log_kuairand.merge(video_features.reset_index()[["video_id"]], on="video_id", how="inner")
    log_kuairand = log_kuairand.sort_values(["user_id", "time_ms"]).reset_index(drop=True)

    replay_events = []
    selected_user_ids = []

    for user_id, user_log in log_kuairand.groupby("user_id", sort=False):
        clicked_history = []
        user_events = []

        for row in user_log.itertuples(index=False):
            if len(clicked_history) >= min_click_history:
                # print('0')
                user_events.append(
                    {
                        "user_id": int(user_id),
                        "time_ms": int(row.time_ms),
                        "history_item_ids": list(clicked_history),
                        "target_item_id": int(row.video_id),
                        "target_record": {
                            "is_hate": int(row.is_hate),
                            "play_time_ms": float(row.play_time_ms),
                            "duration_ms": float(row.duration_ms),
                            "is_click": int(row.is_click),
                            "is_like": int(row.is_like),
                            "is_follow": int(row.is_follow),
                        },
                    }
                )

            if int(row.is_click) == 1:
                clicked_history.append(int(row.video_id))

            if len(user_events) >= max_events_per_user:
                break

        if len(user_events) >= max_events_per_user:
            selected_user_ids.append(int(user_id))
            replay_events.extend(user_events[:max_events_per_user])

        if len(selected_user_ids) >= num_users:
            break

    if not replay_events:
        raise ValueError("No replay events generated. Try reducing min_click_history.")

    if len(selected_user_ids) < num_users:
        print(
            f"Warning: requested {num_users} users for replay, "
            f"but only found {len(selected_user_ids)} users meeting the event threshold."
        )

    return replay_events, video_features, selected_user_ids


def _load_kuairand_semantic_id_mapping(
    video_ids_path: str | None = None,
    semantic_ids_path: str | None = None,
) -> dict[int, str]:
    if video_ids_path is None:
        video_ids_path = os.path.join(OUTPUT_MODEL_PATH, "kuairand_rqvae_video_ids.npy")
    if semantic_ids_path is None:
        semantic_ids_path = os.path.join(OUTPUT_MODEL_PATH, "kuairand_rqvae_semantic_ids.npy")

    if not os.path.exists(video_ids_path):
        raise FileNotFoundError(f"Missing KuaiRand semantic-id video mapping file: {video_ids_path}")
    if not os.path.exists(semantic_ids_path):
        raise FileNotFoundError(f"Missing KuaiRand semantic-id file: {semantic_ids_path}")

    video_ids = np.load(video_ids_path)
    semantic_ids = np.load(semantic_ids_path, allow_pickle=True)

    if video_ids.shape[0] != semantic_ids.shape[0]:
        raise ValueError(
            f"Video id count ({video_ids.shape[0]}) does not match semantic id count ({semantic_ids.shape[0]})"
        )

    return {
        int(video_id): str(semantic_id)
        for video_id, semantic_id in zip(video_ids.tolist(), semantic_ids.tolist())
    }


def _format_kuairand_tag_text(tags: list[int]) -> str:
    if not tags:
        return "unknown"
    return ", ".join(str(int(tag)) for tag in tags)


def prepare_kuairand_cpt_data(
    history_seq_length: int = 20,
    min_history_length: int = 1,
    video_ids_path: str | None = None,
    semantic_ids_path: str | None = None,
) -> pd.DataFrame:
    """
    Build KuaiRand CPT text samples with the format:

    user history:
    - video_sid: ... play_ratio: ...

    video_sid + video_tag:
    - video_sid: ... video_tag: ...

    Filtering rules:
      - is_click == 1
      - tag is not empty
      - is_rand == 0
      - is_hate == 0
    """
    if history_seq_length <= 0:
        raise ValueError("history_seq_length must be positive")
    if min_history_length <= 0:
        raise ValueError("min_history_length must be positive")
    if min_history_length > history_seq_length:
        raise ValueError("min_history_length cannot be greater than history_seq_length")

    semantic_id_mapping = _load_kuairand_semantic_id_mapping(
        video_ids_path=video_ids_path,
        semantic_ids_path=semantic_ids_path,
    )

    cached_path = dataset_manager.get_dataset(DatasetType.KUAI_RAND_1K)
    log_kuairand = pd.read_csv(os.path.join(cached_path, "log_standard_4_08_to_4_21_1k.csv"))
    video_f_basic = pd.read_csv(os.path.join(cached_path, "video_features_basic_1k.csv"))

    video_features = video_f_basic[
        (video_f_basic["visible_status"] == 1) & (video_f_basic["tag"].notna())
    ][["video_id", "tag"]].copy()
    video_features["tag"] = video_features["tag"].astype(str).str.strip()
    video_features = video_features[video_features["tag"] != ""].copy()
    video_features["tag"] = video_features["tag"].apply(extract_tag)
    video_features = video_features[video_features["tag"].apply(len) > 0].copy()
    video_features = video_features.drop_duplicates(subset=["video_id"])

    filtered_log = log_kuairand[
        (log_kuairand["is_click"] == 1)
        & (log_kuairand["is_rand"] == 0)
        # & (log_kuairand["is_hate"] == 0)
        # & (log_kuairand["play_time_ms"] > 0)
        # & (log_kuairand["duration_ms"] > 0)
    ][["user_id", "video_id", "time_ms", "play_time_ms", "duration_ms"]].copy()

    filtered_log = filtered_log.merge(video_features, on="video_id", how="inner")
    filtered_log = filtered_log[filtered_log["video_id"].isin(semantic_id_mapping)].copy()
    filtered_log = filtered_log.sort_values(["user_id", "time_ms"]).reset_index(drop=True)

    records = []
    for user_id, user_log in filtered_log.groupby("user_id", sort=False):
        num_rows = user_log.shape[0]

        for end_idx in range(min_history_length, num_rows + 1):
            start_idx = max(0, end_idx - history_seq_length)
            history = user_log.iloc[start_idx:end_idx].copy()

            history_lines = []
            sid_tag_lines = []
            seen_video_ids = set()

            for row in history.itertuples(index=False):
                duration_ms = float(row.duration_ms)
                play_time_ms = float(row.play_time_ms)
                if duration_ms == 0. and play_time_ms != 0.:
                    play_ratio = max(np.random.normal(0.5, 0.2), 0.1)
                elif duration_ms == 0. and play_time_ms == 0.:
                    play_ratio = 0
                else:
                    play_ratio = min(1.0, play_time_ms / duration_ms)
                video_sid = semantic_id_mapping[int(row.video_id)]
                tag_text = _format_kuairand_tag_text(list(row.tag))

                history_lines.append(
                    f"- video_sid: {video_sid}, "
                    f"play_ratio: {play_ratio:.4f}"
                )

                if int(row.video_id) not in seen_video_ids:
                    sid_tag_lines.append(f"- video_sid: {video_sid}, video_tag: {tag_text}")
                    seen_video_ids.add(int(row.video_id))

            text = (
                "user history:\n"
                + "\n".join(history_lines)
                + "\n\nvideo_sid + video_tag:\n"
                + "\n".join(sid_tag_lines)
            )

            records.append(
                {
                    "user_id": int(user_id),
                    "history_length": int(history.shape[0]),
                    "text": text,
                }
            )

    data = pd.DataFrame(records)
    if data.empty:
        raise ValueError("No KuaiRand CPT samples generated. Check semantic-id files and filter conditions.")
    return data.reset_index(drop=True)


def _format_kuairand_watch_ratio(play_time_ms: float, duration_ms: float) -> float:
    safe_play_time_ms = float(play_time_ms)
    safe_duration_ms = float(duration_ms)
    if safe_duration_ms <= 0.0:
        return 0.0 if safe_play_time_ms <= 0.0 else max(np.random.normal(0.5, 0.2), 0.1)
    return min(1.0, max(0.0, safe_play_time_ms / safe_duration_ms))


def _compute_kuairand_sft_reward(
    play_time_ms: float,
    duration_ms: float,
    is_hate: int,
    is_like: int,
    is_follow: int,
    is_comment: int,
    is_forward: int,
) -> float:
    base_reward = 0.5
    play_ratio = _format_kuairand_watch_ratio(play_time_ms=play_time_ms, duration_ms=duration_ms)

    if int(is_hate) == 1:
        return 0.0

    reward = base_reward
    if play_ratio < 0.2:
        reward = base_reward - play_ratio

    if int(is_like) == 1 and int(is_follow) == 1:
        reward = base_reward + 0.3
    elif int(is_like) == 1 or int(is_follow) == 1:
        reward = base_reward + 0.2

    if int(is_comment) == 1 or int(is_forward) == 1:
        reward += 0.1
    if play_ratio > 0.6:
        reward += (play_ratio - 0.6)

    return round(float(max(0.0, min(1.0, reward))), 4)


def prepare_kuairand_llm_post_training_data(
    history_seq_length: int = 20,
    min_history_length: int = 1,
    # current_click_only: bool = True,
    # history_click_only: bool = False,
    video_ids_path: str | None = None,
    semantic_ids_path: str | None = None,
) -> pd.DataFrame:
    """
    Build KuaiRand post-training data for generative retrieval / SFT.

    Each sample contains:
      - user info: user_id, user_active_degree
      - watching history: historical video SIDs + play_time_ms/duration_ms
      - current video SID as the supervised target

    Output columns:
      - prompt: textual input for SFT
      - target: current video SID sequence
      - text: prompt + target, convenient for plain causal-LM training
    """
    if history_seq_length <= 0:
        raise ValueError("history_seq_length must be positive")
    if min_history_length <= 0:
        raise ValueError("min_history_length must be positive")
    if min_history_length > history_seq_length:
        raise ValueError("min_history_length cannot be greater than history_seq_length")

    semantic_id_mapping = _load_kuairand_semantic_id_mapping(
        video_ids_path=video_ids_path,
        semantic_ids_path=semantic_ids_path,
    )

    cached_path = dataset_manager.get_dataset(DatasetType.KUAI_RAND_1K)
    log_kuairand = pd.read_csv(os.path.join(cached_path, "log_standard_4_08_to_4_21_1k.csv"))
    user_f = pd.read_csv(os.path.join(cached_path, "user_features_1k.csv"))

    user_features = user_f[["user_id", "user_active_degree"]].copy()
    user_features["user_active_degree"] = (
        user_features["user_active_degree"].fillna("UNKNOWN").astype(str).str.strip()
    )
    user_features.loc[
        ~user_features["user_active_degree"].isin(USER_ACTIVE_DEGREE),
        "user_active_degree",
    ] = "UNKNOWN"

    log_columns = [
        "user_id",
        "video_id",
        "time_ms",
        "is_click",
        "play_time_ms",
        "duration_ms",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
    ]
    filtered_log = log_kuairand[log_columns].copy()
    filtered_log = filtered_log[filtered_log["is_click"] == 1].copy()
    filtered_log = filtered_log[filtered_log["video_id"].isin(semantic_id_mapping)].copy()
    filtered_log = filtered_log.merge(user_features, on="user_id", how="inner")
    filtered_log = filtered_log.sort_values(["user_id", "time_ms"]).reset_index(drop=True)

    records = []
    for user_id, user_log in filtered_log.groupby("user_id", sort=False):
        history_events: list[dict[str, float | int | str]] = []

        for row in user_log.itertuples(index=False):
            if len(history_events) >= min_history_length:
                history_window = history_events[-history_seq_length:]
                history_lines = []
                for history_event in history_window:
                    history_lines.append(
                        f"- video_sid: {history_event['video_sid']}, "
                        f"play_time_ms: {int(history_event['play_time_ms'])}, "
                        f"duration_ms: {int(history_event['duration_ms'])}"
                    )

                current_video_sid = semantic_id_mapping[int(row.video_id)]
                reward = _compute_kuairand_sft_reward(
                    play_time_ms=float(row.play_time_ms),
                    duration_ms=float(row.duration_ms),
                    is_hate=int(row.is_hate),
                    is_like=int(row.is_like),
                    is_follow=int(row.is_follow),
                    is_comment=int(row.is_comment),
                    is_forward=int(row.is_forward),
                )
                prompt = (
                    "user info:\n"
                    f"- user_id: {int(user_id)}\n"
                    f"- user_active_degree: {row.user_active_degree}\n\n"
                    "watching history:\n"
                    + "\n".join(history_lines)
                    + "\n\ncurrent video sid:\n"
                    + "predict the next clicked video sid"
                )
                target = current_video_sid

                records.append(
                    {
                        "user_id": int(user_id),
                        "user_active_degree": str(row.user_active_degree),
                        "history_length": int(len(history_window)),
                        "current_video_id": int(row.video_id),
                        "current_video_sid": current_video_sid,
                        "reward": reward,
                        "prompt": prompt,
                        "target": target,
                        "text": f"{prompt}\n{target}",
                    }
                )

            history_events.append(
                {
                    "video_id": int(row.video_id),
                    "video_sid": semantic_id_mapping[int(row.video_id)],
                    "play_time_ms": float(row.play_time_ms),
                    "duration_ms": float(row.duration_ms),
                }
            )

    data = pd.DataFrame(records)
    if data.empty:
        raise ValueError(
            "No KuaiRand LLM post-training samples generated. Check semantic-id files and history filters."
        )
    return data.reset_index(drop=True)
