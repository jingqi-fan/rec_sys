import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from torch.utils.data import DataLoader


class KuaiRandDINTrainer:
    """
    Trainer for KuaiRand DIN model.

    The model is expected to output raw logits. This trainer applies
    BCEWithLogitsLoss during training and sigmoid during evaluation.
    """

    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
    ):
        self.model = model
        self.train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        self.eval_data_loader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False)

        self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)

    def _move_to_device(self, inputs, labels):
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        labels = labels.to(self.device)
        return inputs, labels

    def train(self, num_epochs: int = 10):
        print("KuaiRand DIN trainer start to train")

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            for step, (inputs, labels) in enumerate(self.train_data_loader):
                inputs, labels = self._move_to_device(inputs, labels)
                self.optimizer.zero_grad()

                logits = self.model(inputs).squeeze(1)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if step % 100 == 99:
                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}], "
                        f"Step [{step + 1}/{len(self.train_data_loader)}], "
                        f"Loss: {running_loss / 100:.4f}"
                    )
                    running_loss = 0.0

            self.eval()

        print("KuaiRand DIN trainer finished training")

    def eval(self):
        """
        Evaluate with AUCPR and ROC-AUC.
        """
        print("Start KuaiRand DIN evaluation")
        self.model.eval()

        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels in self.eval_data_loader:
                inputs, labels = self._move_to_device(inputs, labels)
                logits = self.model(inputs).squeeze(1)
                probs = torch.sigmoid(logits)

                all_predictions.extend(probs.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        precision, recall, _ = precision_recall_curve(all_targets, all_predictions)
        aucpr = auc(recall, precision)
        print(f"AUCPR: {aucpr:.4f}")

        try:
            roc_auc = roc_auc_score(all_targets, all_predictions)
            print(f"ROC-AUC: {roc_auc:.4f}")
        except ValueError:
            roc_auc = None
            print("ROC-AUC: skipped because eval set has a single class")

        return {
            "aucpr": aucpr,
            "roc_auc": roc_auc,
        }

    def save(self, model_name: str, path: str = "./artifacts"):
        if not os.path.exists(path):
            print(f"Unable to find path {path}, creating it...")
            os.makedirs(path)

        filename = os.path.join(path, model_name)
        file_path = Path(filename)
        if not file_path.suffix:
            filename += ".pth"
        elif file_path.suffix != ".pth":
            raise ValueError(f"Invalid extension provided {file_path.suffix}")

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
        }
        torch.save(checkpoint, filename)
