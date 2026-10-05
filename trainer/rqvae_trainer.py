import torch
import torch.nn as nn

import torch.optim as optim


class RQVAETrainer:
    """
    A slightly modified version of the simple trainer to train the RQ-VAE model
    since the RQ-VAE model use a slight different way to compute the loss (no label)
    """

    def __init__(self, model, data_loader):

        self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        self.model = model
        self.data_loader = data_loader

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=0.01)

        self.model.train()
        self.model.to(self.device)

    def train(self, num_epochs: int = 5):
        print(f"Training the RQ-VAE model for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1} / {num_epochs}")

            running_loss = 0
            for i, batch in enumerate(self.data_loader):
                batch = batch.to(self.device)

                loss, quantized_indices = self.model(batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(self.data_loader)}], Loss: {running_loss / 100:.4f}")
                    print(f"Quantized indices: {quantized_indices}")
                    running_loss = 0.0
