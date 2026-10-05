from __future__ import annotations

import torch
import torch.nn as nn


class KuaiRandTwoTowerModel(nn.Module):
    """
    Two-tower model for KuaiRand multi-feature input.

    Input feature keys expected by forward:
        user_id, user_active_degree, follow_user_num_range, fans_user_num_range,
        friend_user_num_range, register_days_range,
        video_id, author_id, music_type, video_duration, tag
    """

    def __init__(
        self,
        user_cardinality: int,
        video_cardinality: int,
        author_cardinality: int,
        music_type_cardinality: int,
        tag_cardinality: int,
        embedding_dim: int = 32,
        tower_hidden_dim: int = 128,
        output_dim: int = 64,
    ) -> None:
        super().__init__()

        self.user_embedding = nn.Embedding(user_cardinality, embedding_dim)
        self.user_active_degree_embedding = nn.Embedding(4, 8)
        self.follow_user_num_range_embedding = nn.Embedding(8, 8)
        self.fans_user_num_range_embedding = nn.Embedding(7, 8)
        self.friend_user_num_range_embedding = nn.Embedding(7, 8)
        self.register_days_range_embedding = nn.Embedding(7, 8)

        self.video_embedding = nn.Embedding(video_cardinality, embedding_dim)
        self.author_embedding = nn.Embedding(author_cardinality, embedding_dim)
        self.music_type_embedding = nn.Embedding(music_type_cardinality, 8)
        self.tag_embedding = nn.Embedding(tag_cardinality, 16, padding_idx=0)
        self.video_duration_projection = nn.Linear(1, 8)

        user_input_dim = embedding_dim + 8 * 5
        item_input_dim = embedding_dim + embedding_dim + 8 + 16 + 8

        self.user_tower = nn.Sequential(
            nn.Linear(user_input_dim, tower_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(tower_hidden_dim, output_dim),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(item_input_dim, tower_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(tower_hidden_dim, output_dim),
        )

    @classmethod
    def from_dataframe(cls, data, embedding_dim: int = 32, tower_hidden_dim: int = 128, output_dim: int = 64):
        return cls(
            user_cardinality=int(data["user_id"].max()) + 1,
            video_cardinality=int(data["video_id"].max()) + 1,
            author_cardinality=int(data["author_id"].max()) + 1,
            music_type_cardinality=int(data["music_type"].max()) + 1,
            tag_cardinality=max(int(max(max(tags) for tags in data["tag_padded"] if len(tags) > 0)) + 1, 1),
            embedding_dim=embedding_dim,
            tower_hidden_dim=tower_hidden_dim,
            output_dim=output_dim,
        )

    def _pool_tag_embedding(self, tag_ids: torch.Tensor) -> torch.Tensor:
        tag_embeddings = self.tag_embedding(tag_ids)  # B x T x D
        tag_mask = (tag_ids != 0).float().unsqueeze(-1)  # B x T x 1
        summed = (tag_embeddings * tag_mask).sum(dim=1)  # B x D
        denom = tag_mask.sum(dim=1).clamp_min(1.0)  # B x 1
        return summed / denom

    def encode_user(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        user_id = features["user_id"].long()
        user_active_degree = features["user_active_degree"].long()
        follow_user_num_range = features["follow_user_num_range"].long()
        fans_user_num_range = features["fans_user_num_range"].long()
        friend_user_num_range = features["friend_user_num_range"].long()
        register_days_range = features["register_days_range"].long()

        user_feature = torch.cat(
            [
                self.user_embedding(user_id),
                self.user_active_degree_embedding(user_active_degree),
                self.follow_user_num_range_embedding(follow_user_num_range),
                self.fans_user_num_range_embedding(fans_user_num_range),
                self.friend_user_num_range_embedding(friend_user_num_range),
                self.register_days_range_embedding(register_days_range),
            ],
            dim=-1,
        )
        return self.user_tower(user_feature)

    def encode_item(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        video_id = features["video_id"].long()
        author_id = features["author_id"].long()
        music_type = features["music_type"].long()
        video_duration = features["video_duration"].float().unsqueeze(-1)
        tag_ids = features["tag"].long()

        item_feature = torch.cat(
            [
                self.video_embedding(video_id),
                self.author_embedding(author_id),
                self.music_type_embedding(music_type),
                self._pool_tag_embedding(tag_ids),
                self.video_duration_projection(video_duration),
            ],
            dim=-1,
        )
        return self.item_tower(item_feature)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        user_vec = self.encode_user(features)
        item_vec = self.encode_item(features)
        # dot product logit, suited for BCEWithLogitsLoss
        return (user_vec * item_vec).sum(dim=-1, keepdim=True)
