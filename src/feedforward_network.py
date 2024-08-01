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

        if self.activation == "relu":
            self.activation = nn.ReLU(inplace=True)

        elif self.activation == "gelu":
            self.activation = nn.GELU()

        elif self.activation == "leaky_relu":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)

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
    net = FeedForwardNetwork(
        in_features=512,
        out_features=2048,
        activation="gelu",
        bias=True,
    )

    print(net)
