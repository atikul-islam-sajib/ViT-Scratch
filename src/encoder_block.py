import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from feedforward_network import FeedForwardNetwork
from multihead_attention import MultiHeadAttention
from layer_normalization import LayerNormalization


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dimension: int = 512,
        nheads: int = 8,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
        epsilon: float = 1e-5,
        activation: str = "relu",
        bias: bool = True,
    ):
        super(TransformerEncoder, self).__init__()

        self.dimension = dimension
        self.nheads = nheads
        self.dropout = dropout
        self.epsilon = epsilon
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.bias = bias

        self.multihead_attention = MultiHeadAttention(
            dimension=self.dimension,
            nheads=self.nheads,
            dropout=self.dropout,
            bias=self.bias,
        )

        self.feedforward_network = FeedForwardNetwork(
            in_features=self.dimension,
            out_features=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            bias=self.bias,
        )

        self.layernorm = LayerNormalization(
            normalized_shape=self.dimension, epsilon=self.epsilon
        )

    def forward(self, x: torch.Tensor, mask=None):
        if isinstance(x, torch.Tensor):
            residual = x

            x = self.multihead_attention(x=x, mask=mask)
            x = torch.dropout(input=x, p=self.dropout, train=self.training)
            x = torch.add(x, residual)
            x = self.layernorm(x)

            residual = x

            x = self.feedforward_network(x=x)
            x = torch.dropout(input=x, p=self.dropout, train=self.training)
            x = torch.add(x, residual)
            x = self.layernorm(x)

            return x

        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    transformerEncoder = TransformerEncoder(
        dimension=512,
        nheads=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        bias=True,
        epsilon=1e-6,
    )

    print(transformerEncoder(torch.randn(40, 200, 512)).size())
