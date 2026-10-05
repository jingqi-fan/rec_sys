import torch
from torch.utils.data import DataLoader

from data.dataset_movie_len import MovieLenSemanticEmbeddingDataset
from model.rqvae import RQVAE
from trainer.rqvae_trainer import RQVAETrainer


def count_codebook_usage(indices, codebook_size):
    usage_counts = torch.bincount(indices, minlength=codebook_size)
    return usage_counts


if __name__ == "__main__":
    train_dataset = MovieLenSemanticEmbeddingDataset()
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    model = RQVAE(input_dim=2048, hidden_dim=1024, output_dim=1024, num_quantize=3)
    trainer = RQVAETrainer(model=model, data_loader=train_loader)
    trainer.train(num_epochs=1)

    # Add some simple test to count the codebook usage
    # by default, the codebook size is 32, number of codebook is 3, just use cpu to count the usage
    codebook_counters = {i: torch.zeros(32) for i in range(3)}
    model.to("cpu")
    model.eval()

    test_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    for batch in test_loader:
        _, ind = model(batch)  # ind is a list of 3 tensors
        for i, ind_i in enumerate(ind):
            codebook_counters[i] += count_codebook_usage(ind_i, 32)
    print(codebook_counters)
