from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class KuaiRandEmbeddingTrainer:
    """
    Trainer for KuaiRand two-tower model with binary click labels.
    """

    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset,
        batch_size: int = 2048,
        eval_batch_size: int = 4096,
        learning_rate: float = 1e-3,
    ):
        self.model = model
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

        self.loss = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)

    def _to_device(self, feature_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: value.to(self.device) for name, value in feature_dict.items()}

    def train(self, num_epochs: int = 3):
        print("Start KuaiRand embedding training...")
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            for step, (features, labels) in enumerate(self.train_dataloader):
                labels = labels.to(self.device)
                features = self._to_device(features)

                self.optimizer.zero_grad()
                logits = self.model(features).squeeze(1)
                loss = self.loss(logits, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if step % 100 == 99:
                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}] Step [{step + 1}/{len(self.train_dataloader)}] "
                        f"Loss: {running_loss / 100:.4f}"
                    )
                    running_loss = 0.0

    def eval(self):
        self.model.eval()
        total_loss = 0.0
        num_batch = 0

        with torch.no_grad():
            for features, labels in self.eval_dataloader:
                labels = labels.to(self.device)
                features = self._to_device(features)
                logits = self.model(features).squeeze(1)
                loss = self.loss(logits, labels)
                total_loss += loss.item()
                num_batch += 1

        avg_loss = total_loss / max(num_batch, 1)
        print(f"KuaiRand eval loss: {avg_loss:.4f}")
        return avg_loss

    def export_item_embeddings(self, item_feature_dataframe, batch_size: int = 4096):
        """
        Export item-tower embeddings from a de-duplicated item feature dataframe.
        Required columns:
            video_id, author_id, music_type, video_duration, tag_padded
        """
        self.model.eval()

        unique_items = item_feature_dataframe.drop_duplicates(subset=["video_id"]).copy()
        unique_items = unique_items.sort_values("video_id").reset_index(drop=True)

        video_ids = unique_items["video_id"].to_numpy(dtype=np.int64)
        item_embeddings = []

        with torch.no_grad():
            for start in range(0, len(unique_items), batch_size):
                batch = unique_items.iloc[start : start + batch_size]
                batch_features = {
                    "video_id": torch.LongTensor(batch["video_id"].to_numpy()),
                    "author_id": torch.LongTensor(batch["author_id"].to_numpy()),
                    "music_type": torch.LongTensor(batch["music_type"].to_numpy()),
                    "video_duration": torch.FloatTensor(batch["video_duration"].to_numpy()),
                    "tag": torch.LongTensor(batch["tag_padded"].to_list()),
                }
                batch_features = self._to_device(batch_features)
                batch_embeddings = self.model.encode_item(batch_features).cpu().numpy()
                item_embeddings.append(batch_embeddings)

        return video_ids, np.concatenate(item_embeddings, axis=0) if item_embeddings else np.zeros((0, 0))
