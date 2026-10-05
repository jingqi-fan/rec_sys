from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalActivationUnit(nn.Module):
    """
    Candidate-aware attention scorer used by DIN.
    """

    def __init__(self, embedding_dim: int, hidden_dims: tuple[int, int] = (128, 64)) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim * 4, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)

    def forward(self, query: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: B x 1 x D
            history: B x T x D
        Returns:
            attention logits: B x T
        """
        seq_len = history.size(1)
        repeated_query = query.expand(-1, seq_len, -1)
        attn_input = torch.cat(
            [
                repeated_query,
                history,
                repeated_query - history,
                repeated_query * history,
            ],
            dim=-1,
        )
        scores = F.relu(self.fc1(attn_input))
        scores = F.relu(self.fc2(scores))
        scores = self.fc3(scores).squeeze(-1)
        return scores


class AttentionSequencePoolingLayer(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.local_att = LocalActivationUnit(embedding_dim=embedding_dim)

    def forward(
        self,
        query: torch.Tensor,
        history: torch.Tensor,
        history_length: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: B x 1 x D
            history: B x T x D
            history_length: B
        Returns:
            pooled history vector: B x D
        """
        attn_logits = self.local_att(query, history)  # B x T
        positions = torch.arange(history.size(1), device=history.device).unsqueeze(0)
        mask = positions < history_length.unsqueeze(1)

        attn_logits = attn_logits.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = attn_weights.unsqueeze(1)  # B x 1 x T
        pooled = torch.bmm(attn_weights, history).squeeze(1)  # B x D
        return pooled


class KuaiRandDINModel(nn.Module):
    """
    DIN model for KuaiRand.

    Expected input feature keys:
        user_id
        user_active_degree
        follow_user_num_range
        fans_user_num_range
        friend_user_num_range
        register_days_range
        candidate_video_id
        candidate_author_id
        candidate_tag_ids
        history_video_ids
        history_author_ids
        history_tag_ids
        history_length
    """

    def __init__(
        self,
        user_cardinality: int,
        video_cardinality: int,
        author_cardinality: int,
        tag_cardinality: int,
        embedding_dim: int = 32,
        tag_embedding_dim: int = 16,
        mlp_hidden_dims: tuple[int, int] = (256, 128),
    ) -> None:
        super().__init__()

        self.user_embedding = nn.Embedding(user_cardinality, embedding_dim)
        self.user_active_degree_embedding = nn.Embedding(4, 8)
        self.follow_user_num_range_embedding = nn.Embedding(8, 8)
        self.fans_user_num_range_embedding = nn.Embedding(7, 8)
        self.friend_user_num_range_embedding = nn.Embedding(7, 8)
        self.register_days_range_embedding = nn.Embedding(7, 8)

        self.video_embedding = nn.Embedding(video_cardinality, embedding_dim, padding_idx=0)
        self.author_embedding = nn.Embedding(author_cardinality, embedding_dim, padding_idx=0)
        self.tag_embedding = nn.Embedding(tag_cardinality, tag_embedding_dim, padding_idx=0)

        self.item_representation_dim = embedding_dim + embedding_dim + tag_embedding_dim
        self.attention = AttentionSequencePoolingLayer(embedding_dim=self.item_representation_dim)

        user_feature_dim = embedding_dim + 8 * 5
        input_dim = user_feature_dim + self.item_representation_dim * 2 + 1
        self.head = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dims[0], mlp_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dims[1], 1),
        )

    def _pool_tags(self, tag_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tag_ids:
                candidate tags -> B x T
                history tags   -> B x H x T
        """
        tag_embeddings = self.tag_embedding(tag_ids)
        tag_mask = (tag_ids != 0).float().unsqueeze(-1)
        tag_sum = (tag_embeddings * tag_mask).sum(dim=-2)
        tag_count = tag_mask.sum(dim=-2).clamp_min(1.0)
        return tag_sum / tag_count

    def _encode_user(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [
                self.user_embedding(features["user_id"].long()),
                self.user_active_degree_embedding(features["user_active_degree"].long()),
                self.follow_user_num_range_embedding(features["follow_user_num_range"].long()),
                self.fans_user_num_range_embedding(features["fans_user_num_range"].long()),
                self.friend_user_num_range_embedding(features["friend_user_num_range"].long()),
                self.register_days_range_embedding(features["register_days_range"].long()),
            ],
            dim=-1,
        )

    def _encode_candidate(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        candidate_tag = self._pool_tags(features["candidate_tag_ids"].long())
        return torch.cat(
            [
                self.video_embedding(features["candidate_video_id"].long()),
                self.author_embedding(features["candidate_author_id"].long()),
                candidate_tag,
            ],
            dim=-1,
        )

    def _encode_history(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        history_video = self.video_embedding(features["history_video_ids"].long())
        history_author = self.author_embedding(features["history_author_ids"].long())
        history_tag = self._pool_tags(features["history_tag_ids"].long())
        return torch.cat([history_video, history_author, history_tag], dim=-1)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Returns:
            logits with shape B x 1
        """
        user_repr = self._encode_user(features)
        candidate_repr = self._encode_candidate(features)
        history_repr = self._encode_history(features)

        history_length = features["history_length"].long().view(-1)
        pooled_history = self.attention(
            candidate_repr.unsqueeze(1),
            history_repr,
            history_length,
        )

        history_length_feature = history_length.float().unsqueeze(-1)
        model_input = torch.cat(
            [
                user_repr,
                candidate_repr,
                pooled_history,
                history_length_feature,
            ],
            dim=-1,
        )
        return self.head(model_input)
