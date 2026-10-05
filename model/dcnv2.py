import torch
import torch.nn as nn

from configs.model import DCNv2Config

"""
This is a trial to implement the DCN v2 module in the paper

    DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems
    https://arxiv.org/pdf/2008.13535

with certain simplification

    1. to make it easier for understanding, the low rank approx of the W matrix is not implemented

Code reference:
    1. https://gist.github.com/lzqlzzq/4cd81048119dfe6baf2acec85d1fe790
    2. https://github.com/pytorch/torchrec/blob/main/torchrec/modules/crossnet.py
"""


class CrossNet(nn.Module):
    """
    CrossNet performs the feature crossing to capture the higher order feature
    interactions.
    """

    def __init__(self, hidden_dim, num_layers: int = 3):
        super().__init__()
        # create the layers via the sequential
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim, bias=True),
                    nn.BatchNorm1d(hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x0):
        """
        x0: B x D

        In the original feature, x0 is the output of concat of the dense vector
        after the embedding lookup. This support varlen format of the embedding
        dimension for each categorical features. In some impl (e.g. ref 1), the
        x0 is represented as B x seq x D, where the batch norm need some change
        to be integrated with a transpose layer
        """
        x = x0
        # This is fine but might not very efficient, Pytorch would generate temporary
        # activation of x for the auto differential graph computation to compute the
        # gradient correctly
        for layer in self.layers:
            x = layer(x) * x0 + x
        return x


class DeepNet(nn.Module):
    """
    DeepNet is a regular MLP with non linear activations

    hidden_dims: a list of hidden dimensions for the deep net linear layers
    """

    def __init__(self, hidden_dims):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.Dropout(),
                )
                for i in range(len(hidden_dims) - 1)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DCNV2(nn.Module):

    def __init__(self, config: DCNv2Config):
        super().__init__()

        # embedding layer of the category features, for each
        # feature we would have a lookup table to simulate one-hot-encoding
        self.embeddings = {}
        if config.feature_config:
            input_dim = 0
            for feature_name, (num_embedding, dim) in config.feature_config.items():
                self.embeddings[feature_name] = nn.Embedding(num_embedding, dim)
                input_dim += dim
        else:
            input_dim = config.input_dim

        print(f"Final input dim is {input_dim}")
        self.cross_net = CrossNet(input_dim, config.num_cross_layers)
        self.deep_net = DeepNet([input_dim] + config.deep_net_hidden_dims)

        # we create a final prediction head to generate the probability, this
        # depends on the hidden dimension of cross net and deep net, as well as
        # the mode how these 2 part is combined
        self.head = nn.Sequential(nn.Linear(input_dim + config.deep_net_hidden_dims[-1], config.head_hidden_dim), nn.ReLU(), nn.Linear(config.head_hidden_dim, 1), nn.Sigmoid())

    def forward(self, features):
        if self.embeddings:
            emb = []
            for k, v in features.items():
                emb.append(self.embeddings[k](v))
            x = torch.concat(emb, dim=-1)  # we concat all embeddings together
        else:
            x = features

        cross_out = self.cross_net(x)  # B x hidden
        deep_out = self.deep_net(x)  # B x hidden

        # only implement the concat mode as of now
        hidden = torch.concat((cross_out, deep_out), dim=-1)
        out = self.head(hidden)
        return out


if __name__ == "__main__":

    dcnv2_config = DCNv2Config(
        feature_config={"feature_a": (256, 16), "feature_b": (1024, 64), "feature_c": (1024, 16), "feature_d": (256, 32)},
        num_cross_layers=3,
        deep_net_hidden_dims=[128, 64, 32],
        head_hidden_dim=128,
    )

    dcnv2 = DCNV2(dcnv2_config)

    features = {
        "feature_a": torch.LongTensor([1, 2, 3]),  # 3,
        "feature_b": torch.LongTensor([100, 200, 300]),
        "feature_c": torch.LongTensor([1001, 1002, 1003]),
        "feature_d": torch.LongTensor([11, 21, 31]),
    }

    out = dcnv2(features)
    print(out)
