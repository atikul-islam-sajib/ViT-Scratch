import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config
from patch_embedding import PatchEmbedding
from transformer import TransformerEncoder


class ViT(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        image_size: int = 224,
        labels: int = 4,
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
        self.labels = labels
        self.path_size = path_size
        self.nheads = nheads
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.epsilon = epsilon
        self.activation = activation
        self.bias = bias

        self.num_patches = (self.image_size // self.path_size) ** 2
        self.num_dimension = (self.path_size**2) * self.image_channels

        assert (
            self.num_dimension % self.nheads == 0
        ), "nheads must be a divisor of num_dimension".capitalize()

        assert (
            len(config()["dataloader"]["labels"]) == self.labels
        ), "labels must be equal to the number of classes in the dataset, check the config.yml file".capitalize()

        self.patch_embedding = PatchEmbedding(
            image_size=self.image_size,
            image_channels=self.image_channels,
            patch_size=self.path_size,
        )

        self.transformerEncoder = TransformerEncoder(
            dimension=self.num_dimension,
            nheads=self.nheads,
            num_encoder_layers=self.num_encoder_layers,
            dropout=self.dropout,
            dim_feedforward=self.dim_feedforward,
            epsilon=self.epsilon,
            activation=self.activation,
            bias=self.bias,
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(
                in_features=self.num_dimension,
                out_features=self.labels,
                bias=self.bias,
            ),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor, mask=None):
        if isinstance(x, torch.Tensor):
            x = self.patch_embedding(x)
            x = self.transformerEncoder(x, mask=mask)

            x = self.mlp_head(x[:, 0, :])

            return x

        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    vision_transformer = ViT(
        image_channels=3,
        image_size=224,
        path_size=16,
        nheads=8,
        num_encoder_layers=2,
        dropout=0.1,
        dim_feedforward=2048,
        epsilon=1e-5,
        activation="relu",
        bias=True,
    )

    print(vision_transformer(torch.randn(64, 3, 224, 224)).size())
