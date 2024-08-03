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
        patch_size: int = 16,
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
        self.patch_size = patch_size
        self.nheads = nheads
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.epsilon = epsilon
        self.activation = activation
        self.bias = bias

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_dimension = (self.patch_size**2) * self.image_channels

        assert (
            self.num_dimension % self.nheads == 0
        ), "nheads must be a divisor of num_dimension".capitalize()

        assert (
            len(config()["dataloader"]["labels"]) == self.labels
        ), "labels must be equal to the number of classes in the dataset, check the config.yml file".capitalize()

        self.patch_embedding = PatchEmbedding(
            image_size=self.image_size,
            image_channels=self.image_channels,
            patch_size=self.patch_size,
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
    parser = argparse.ArgumentParser(description="Vit Model".title())
    parser.add_argument(
        "--image_channels",
        type=int,
        default=config()["dataloader"]["channels"],
        help="Number of image channels".capitalize(),
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=config()["dataloader"]["image_size"],
        help="Image size".capitalize(),
    )
    parser.add_argument(
        "--labels",
        type=int,
        default=len(config()["dataloader"]["labels"]),
        help="Number of labels".capitalize(),
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=config()["ViT"]["patch_size"],
        help="patch_size size".capitalize(),
    )
    parser.add_argument(
        "--nheads",
        type=int,
        default=config()["ViT"]["nheads"],
        help="Number of heads".capitalize(),
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=config()["ViT"]["num_layers"],
        help="Number of encoder layers".capitalize(),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=config()["ViT"]["dropout"],
        help="Dropout rate".capitalize(),
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=config()["ViT"]["dim_feedforward"],
        help="Dimension of feedforward".capitalize(),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=config()["ViT"]["eps"],
        help="Epsilon".capitalize(),
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=config()["ViT"]["activation"],
        help="Activation function".capitalize(),
    )
    parser.add_argument(
        "--bias",
        type=bool,
        default=True,
        help="Bias".capitalize(),
    )

    args = parser.parse_args()

    vision_transformer = ViT(
        image_channels=args.image_channels,
        image_size=args.image_size,
        labels=args.labels,
        patch_size=args.patch_size,
        nheads=args.nheads,
        num_encoder_layers=args.num_encoder_layers,
        dropout=args.dropout,
        dim_feedforward=args.dim_feedforward,
        epsilon=args.epsilon,
        activation=args.activation,
        bias=args.bias,
    )

    batch_size = config()["dataloader"]["batch_size"]

    assert vision_transformer(
        torch.randn(
            batch_size,
            args.image_channels,
            args.image_size,
            args.image_size,
        )
    ).size() == (batch_size, args.labels)
