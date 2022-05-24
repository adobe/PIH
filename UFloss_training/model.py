import numpy as np
import torch
import sys


class Model(torch.nn.Module):
    def __init__(
        self,
        network_class,
        feature_dim=128,
        temperature=1.0,
        device=torch.device("cpu"),
        data_length=921600,
    ):
        super().__init__()
        print(f"Using parameters:\n Temperature: {temperature}")

        self.device = device
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.network = network_class(num_classes=feature_dim).to(device)
        self.data_length = data_length

        self.register_buffer(
            "memory_bank", torch.randn(feature_dim, self.data_length)
        )  # Save memory bank with model
        self.memory_bank = torch.nn.functional.normalize(
            self.memory_bank.to(self.device), p=2, dim=0
        )

    @torch.no_grad()
    def update_target_network(self):
        """Momentum update for delayed (target) network."""
        for target_param, param in zip(
            self.target_network.parameters(), self.network.parameters()
        ):
            target_param.data = target_param.data * self.momentum + param.data * (
                1.0 - self.momentum
            )

    @torch.no_grad()
    def update_memory_bank(self, embeddings, indices):
        """Update memory bank 

        Parameters
        ----------
        embeddings : Tensor
        indices : int array
        """
        #         print(embeddings.shape)
        #         sys.exit()

        self.memory_bank[:, indices] = embeddings.T

    def forward(self, images):
        embeddings = self.network(images)[0]  # B, H, W, F (channels last)

        return embeddings

    def predict(self, images):
        with torch.no_grad():
            embeddings = self.network(images)[0]  # B, H, W, F (channels last)

            return embeddings

    def get_logits_labels(self, embeddings):
        """Compute logits and labels used for computing the loss.

        Parameters
        ----------
        embeddings
        target_embeddings
        sample_rate

        Returns
        -------

        """

        # Compare each patch against the memory bank. (N, F) x (F, S) = (N, S).
        #         print(self.temperature)
        logits = torch.mm(embeddings, self.memory_bank) / self.temperature

        return logits
