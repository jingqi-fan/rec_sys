class DeepInterestModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.user_embedding = nn.Embedding(
            num_embeddings=config.user_cardinality,
            embedding_dim=config.user_embedding_dim,
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=config.item_cardinality,
            embedding_dim=config.item_embedding_dim,
        )
        self.attn = UserHistoryBehaviorPoolingModule()
        self.head = nn.Sequential(
            nn.Linear(
                in_features=config.user_embedding_dim + 2 * config.item_embedding_dim + config.num_dense_features,
                out_features=256,
                bias=True,
            ),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1, bias=False),
            nn.Sigmoid(),
        )
