import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        in_features: int = 512,
        out_features: int = 2048,
        dropout: float = 0.5,
        activation: str = "relu",
        bias: bool = True,
    ):
        super(FeedForwardNetwork, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.activation = activation
        self.bias = bias

        if self.activation == "elu":
            self.activation = nn.ELU(inplace=True)

        elif self.activation == "gelu":
            self.activation = nn.GELU()

        elif self.activation == "leaky_relu":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

        self.layers = []

        for index in range(2):
            self.layers.append(
                nn.Linear(
                    in_features=self.in_features,
                    out_features=self.out_features,
                    bias=self.bias,
                )
            )
            self.in_features = self.out_features
            self.out_features = in_features

            if index == 0:
                self.layers.append(self.activation)
                self.layers.append(nn.Dropout(p=self.dropout))

        self.network = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            return self.network(x)

        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pointwise Feed Forward Network for Transformer".title()
    )
    parser.add_argument(
        "--in_features",
        type=int,
        default=config()["ViT"]["dimension"],
        help="Number of input features".capitalize(),
    )
    parser.add_argument(
        "--out_features",
        type=int,
        default=config()["ViT"]["dim_feedforward"],
        help="Number of output features".capitalize(),
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="gelu",
        choices=["gelu", "relu", "silu", "leaky_relu", "elu"],
        help="Activation function".capitalize(),
    )

    args = parser.parse_args()

    batch_size = config()["ViT"]["batch_size"]
    dimension = args.in_features

    x = torch.randn((batch_size, 200, dimension))

    net = FeedForwardNetwork(
        in_features=args.in_features,
        out_features=args.out_features,
        activation=args.activation,
        bias=True,
    )

    assert net(x=x).size() == (
        batch_size,
        200,
        dimension,
    ), "Output shape is incorrect in PointWise FeedForward Network".capitalize()
