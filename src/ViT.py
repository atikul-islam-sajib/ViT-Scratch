import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from patch_embedding import PatchEmbedding
from transformer import TransformerEncoder


class ViT(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        image_size: int = 224,
        path_size: int = 16,
        nheads: int = 8,
        num_encoder_layers: int = 8,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
        epsilon: float = 1e-5,
        activation: str = "relu",
        bias: bool = True,
    ):

        super(ViT, self).__init__()

        self.image_channels = image_channels
        self.image_size = image_size
        self.path_size = path_size
        self.nheads = nheads
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.epsilon = epsilon
        self.activation = activation
        self.bias = bias

    def forward(self, x: torch.Tensor, mask=None):
        if isinstance(x, torch.Tensor):
            pass

        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    vision_transformer = ViT(
        image_channels=3,
        image_size=224,
        path_size=16,
        nheads=8,
        num_encoder_layers=8,
        dropout=0.1,
        dim_feedforward=2048,
        epsilon=1e-5,
        activation="relu",
        bias=True,
    )
