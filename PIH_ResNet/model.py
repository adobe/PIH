import numpy as np
import torch
import sys


class Model(torch.nn.Module):
    def __init__(
        self,
        network_class,
        feature_dim=128,
        device=torch.device("cpu")
    ):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.network = network_class(num_classes=feature_dim).to(device)

    
    def forward(self, images):
        embeddings = self.network(images)[0]  # B, H, W, F (channels last)

        return embeddings
