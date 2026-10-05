import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from configs.model import OUTPUT_MODEL_PATH
from model.rqvae import RQVAE
from trainer.rqvae_trainer import RQVAETrainer


BATCH_SIZE = 512
NUM_EPOCHS = 5
NUM_QUANTIZE = 3
CODEBOOK_SIZE = 32
HIDDEN_DIM = 256
OUTPUT_DIM = 64


def count_codebook_usage(indices: torch.Tensor, codebook_size: int) -> torch.Tensor:
    return torch.bincount(indices, minlength=codebook_size)


class KuaiRandItemEmbeddingDataset(Dataset):
    """
    Load KuaiRand item embeddings exported by the two-tower model and prepare
    them for RQ-VAE training.
    """

    def __init__(self):
        embeddings_path = os.path.join(OUTPUT_MODEL_PATH, "kuairand_item_embeddings.pt")
        item_ids_path = os.path.join(OUTPUT_MODEL_PATH, "kuairand_item_ids.npy")

        embeddings = torch.load(embeddings_path, weights_only=False)
        item_ids = np.load(item_ids_path)

        embeddings = np.asarray(embeddings, dtype=np.float32)
        item_ids = np.asarray(item_ids, dtype=np.int64)

        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D item embeddings, got shape {embeddings.shape}")
        if embeddings.shape[0] != item_ids.shape[0]:
            raise ValueError(
                f"Embedding count ({embeddings.shape[0]}) does not match item id count ({item_ids.shape[0]})"
            )
        if embeddings.shape[0] == 0:
            raise ValueError("No KuaiRand item embeddings found")

        norms = np.linalg.norm(embeddings, axis=1)
        valid_mask = norms > 0
        self.video_ids = item_ids[valid_mask]
        self.embeddings = embeddings[valid_mask] / norms[valid_mask, None]

        print(f"Loaded {item_ids.shape[0]} item embeddings")
        print(f"Filtered {int((~valid_mask).sum())} zero-norm embeddings")
        print(f"Training on {self.embeddings.shape[0]} normalized item embeddings")

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    def __len__(self) -> int:
        return int(self.embeddings.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(self.embeddings[index], dtype=torch.float32)


def export_semantic_ids(model: RQVAE, dataset: KuaiRandItemEmbeddingDataset):
    device = next(model.parameters()).device
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    codebook_counters = {i: torch.zeros(CODEBOOK_SIZE) for i in range(NUM_QUANTIZE)}
    semantic_codes = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            _, quantized_indices = model(batch)

            stacked_codes = torch.stack(quantized_indices, dim=1).cpu().numpy()
            semantic_codes.append(stacked_codes)

            for i, ind_i in enumerate(quantized_indices):
                codebook_counters[i] += count_codebook_usage(ind_i.cpu(), CODEBOOK_SIZE)

    semantic_codes = np.concatenate(semantic_codes, axis=0)
    semantic_id_strings = np.asarray(["-".join(map(str, codes.tolist())) for codes in semantic_codes], dtype=object)

    np.save(os.path.join(OUTPUT_MODEL_PATH, "kuairand_rqvae_video_ids.npy"), dataset.video_ids)
    np.save(os.path.join(OUTPUT_MODEL_PATH, "kuairand_rqvae_codes.npy"), semantic_codes)
    np.save(os.path.join(OUTPUT_MODEL_PATH, "kuairand_rqvae_semantic_ids.npy"), semantic_id_strings)

    print("Saved semantic ids to artifacts/kuairand_rqvae_semantic_ids.npy")
    print("Saved quantization codes to artifacts/kuairand_rqvae_codes.npy")
    print("Saved aligned video ids to artifacts/kuairand_rqvae_video_ids.npy")
    print(f"Codebook usage: {codebook_counters}")


if __name__ == "__main__":
    dataset = KuaiRandItemEmbeddingDataset()
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = RQVAE(
        input_dim=dataset.embedding_dim,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_quantize=NUM_QUANTIZE,
    )

    trainer = RQVAETrainer(model=model, data_loader=train_loader)
    trainer.train(num_epochs=NUM_EPOCHS)
    export_semantic_ids(model, dataset)
