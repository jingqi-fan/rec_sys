import torch
import torch.nn as nn

from configs.model import (
    DCNv2Config,
    TransActModuleConfig,
    TransActModelConfig,
)
from model.dcnv2 import DCNV2

"""
This is a trial to re-implement the model architecture in paper

    TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest
    https://arxiv.org/abs/2306.00248

with certain simplification

    1. the time window masking is removed for now since we would use MovieLen dataset initially

Code reference:
    1. https://github.com/pinterest/transformer_user_action/tree/main
"""


class TransActModule(nn.Module):
    def __init__(self, config: TransActModuleConfig):
        super().__init__()

        self.max_seq_len = config.max_seq_len  # the number of sequence length to use
        self.d_action = config.action_emb_dim
        self.d_item = config.item_emb_dim
        self.top_k = config.top_k

        # this is the embedding for action type in the sequences
        self.action_embedding = nn.Embedding(num_embeddings=config.num_action, embedding_dim=self.d_action, padding_idx=0)
        # for now we would reuse the existing encoder layer available in pytorch for simplicity
        # we would use our customized encoder implementation
        d_model = self.d_action + 2 * self.d_item
        self.encoding_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.transformer_num_head,
            dim_feedforward=config.transformer_hidden_dim,
            batch_first=True,  # this makes sure it is B x seq x dim
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoding_layer, num_layers=config.num_transformer_block)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, action_sequence, item_sequence, candidate):
        """
        This is a simplified version, where we would ignore the request time in the original
        paper which is used for time window masking

        The input item_sequence embedding dimension might be different from the d_item in the config, and we
        would apply a learnable projection layer to do the dimension reduction

            action_sequence: the action type in user history behavior, B x seq
            item_sequence: the item embedding in user history behavior, B x seq x d_item
            candidate: the candidate item embedding to be scored, B x d_item
        """

        action_seq = action_sequence[:, : self.max_seq_len]  # B x seq
        action_seq_embedding = self.action_embedding(action_seq + 1)  # B x seq x d_action
        item_seq_embedding = item_sequence[:, : self.max_seq_len, :]  # B x seq x d_item

        # for know we would assume that there is no padding in the input seq (which could be controlled outside)
        padding_mask = action_seq <= 0  # B x seq
        attn_mask = torch.zeros_like(padding_mask, dtype=torch.float)
        attn_mask.masked_fill_(padding_mask, float("-inf"))

        action_item_embedding = torch.concat((action_seq_embedding, item_seq_embedding), dim=2)  # B x seq x (d_action + d_item)

        # the paper also mentioned that they adopted concat candidate embedding to each action_pin embedding instead
        # of append it to the sequence (which would result in seq + 1)
        candidate_embedding = candidate.unsqueeze(1).expand(-1, self.max_seq_len, -1)  # B x seq x d_item
        action_item_embedding = torch.concat((action_item_embedding, candidate_embedding), dim=-1)  # B x seq x (d_action + 2 * d_item)

        trans_out = self.encoder(src=action_item_embedding, src_key_padding_mask=attn_mask)  # B x seq x d_model

        # max pooling
        max_pool_embedding = trans_out.max(dim=1).values  # B x d_model
        max_pool_embedding = self.out_linear(max_pool_embedding)
        # in the official implementation, there is a linear transformation after the max pooling, but
        # in the paper it is not mentioned
        out = torch.concat((trans_out[:, : self.top_k, :].flatten(1), max_pool_embedding), dim=1)  # B x (top_k + 1) * d_model
        return out


class TransAct(nn.Module):
    """
    This is the main model of transact follow the architecture of the paper.

    We would simplify the PinnerFormer part and directly use the raw features
    as input.
    """

    def __init__(self, config: TransActModelConfig):
        super().__init__()

        self.transact_module = TransActModule(config.transact_module_config)
        self.dcnv2 = DCNV2(config.dcnv2_config)

        self.user_embedding = nn.Embedding(num_embeddings=config.user_cardinality, embedding_dim=32)
        self.genre_embedding = nn.Embedding(num_embeddings=32, embedding_dim=32)

        # if we use semantic embedding, add a projection layer to do the dimension reduction
        self.embedding_projection = nn.Linear(config.semantic_embedding_dim, config.transact_module_config.item_emb_dim) if config.use_semantic_embedding else nn.Identity()

    def forward(self, features):
        """
        features: dict[str, tensor], represent the input in a dict format to simplify
        the logic here thought might not most efficient
        """
        # sequence input for transact module
        action_sequence = features["action_sequence"]
        item_sequence = features["item_sequence"]
        candidate = features["candidate"]

        # dimension reduction
        item_sequence = self.embedding_projection(item_sequence)
        candidate = self.embedding_projection(candidate)

        # sparse feature
        user_id = features["user_id"]  # B,
        # we use fixed length of number of genres here for simplicity
        # TODO replace with varlen impl
        candidate_genres = features["candidate_genres"]  # B x num_gender
        user_viewed_genres = features["user_viewed_genres"]  # B x num_gender

        # dense feature
        num_movies_viewed = features["user_num_viewed_movies"]  # B,

        transact_out = self.transact_module(action_sequence, item_sequence, candidate)  # B x trans_dim
        user_emb = self.user_embedding(user_id).squeeze(1)  # B x user_emb_dim
        candidate_genre_emb = self.genre_embedding(candidate_genres)  # B x num_genres x emb_dim
        # Suppose 0 is the padding index
        mask = (candidate_genres != 0).unsqueeze(-1)  # B x num_genres x 1
        # Mean pooling
        candidate_genre_emb_pooled = (candidate_genre_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        # Now candidate_genre_emb_pooled is B x emb_dim
        user_viewed_genre_emb = self.genre_embedding(user_viewed_genres)  # B x num_genres x emb_dim
        user_viewed_mask = (user_viewed_genres != 0).unsqueeze(-1)  # B x num_genres x 1
        user_viewed_genre_emb_pooled = (user_viewed_genre_emb * user_viewed_mask).sum(dim=1) / user_viewed_mask.sum(dim=1).clamp(min=1)

        dcnv2_in = torch.concat((transact_out, user_emb, candidate_genre_emb_pooled, user_viewed_genre_emb_pooled, num_movies_viewed), dim=1)

        out = self.dcnv2(dcnv2_in)
        return out


if __name__ == "__main__":

    transact_module_config = TransActModuleConfig(
        max_seq_len=5,
        action_emb_dim=16,
        item_emb_dim=64,
        num_action=16,
        top_k=3,
        transformer_num_head=4,
        transformer_hidden_dim=1024,
        num_transformer_block=2,
    )

    dcnv2_config = DCNv2Config(
        feature_config={},
        # TODO, need a better approach to compute the input dim to dcnv2 here
        # the current 97 is from user_emb, user_genre_viewed, candidate_genre, num_dense_feature
        input_dim=transact_module_config.transact_out_dim() + 97,
        num_cross_layers=3,
        deep_net_hidden_dims=[128, 64, 32],
        head_hidden_dim=128,
    )

    trans_act_config = TransActModelConfig(
        transact_module_config=transact_module_config,
        dcnv2_config=dcnv2_config,
        user_cardinality=1000,
        use_semantic_embedding=True,
        semantic_embedding_dim=2048,
    )

    trans_act = TransAct(trans_act_config)

    features = {
        "user_id": torch.tensor([1, 2], dtype=torch.int),
        "num_movies_viewed": torch.tensor([1.0, 3.0]).unsqueeze(-1),
        "candidate_genres": torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.int),
        "user_viewed_genres": torch.tensor([[1], [2]], dtype=torch.int),
        "action_sequence": torch.tensor([[1, 2, 3, 1, 2], [2, 3, 3, 3, 1]], dtype=torch.int),
        "item_sequence": torch.ones((2, 5, 2048), dtype=torch.float) / 2048,
        "candidate": torch.ones((2, 2048), dtype=torch.float) / 2048,
        "user_num_viewed_movies": torch.tensor([[1], [3]], dtype=torch.float),
    }

    trans_act = TransAct(trans_act_config)

    with torch.no_grad():
        out = trans_act(features)

        print(out)
