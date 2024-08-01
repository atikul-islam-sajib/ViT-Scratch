import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config
from feedforward_network import FeedForwardNetwork
from multihead_attention import MultiHeadAttention
from layer_normalization import LayerNormalization


class TransformerEncoderBlock(nn.Module):
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
        super(TransformerEncoderBlock, self).__init__()

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
    parser = argparse.ArgumentParser(
        description="Encoder Block for the Transformer".title()
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=config()["ViT"]["dimension"],
        help="Dimension of the input tensor".capitalize(),
    )
    parser.add_argument(
        "--nheads",
        type=int,
        default=config()["ViT"]["nheads"],
        help="Number of heads in the multi-head attention".capitalize(),
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=config()["ViT"]["dim_feedforward"],
        help="Dimension of the feedforward network".capitalize(),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=config()["ViT"]["dropout"],
        help="Dropout rate".capitalize(),
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=config()["ViT"]["activation"],
        help="Activation function".capitalize(),
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=config()["ViT"]["eps"],
        help="Epsilon value for the layer normalization".capitalize(),
    )

    args = parser.parse_args()

    batch_size = config()["ViT"]["batch_size"]

    transformerEncoder = TransformerEncoderBlock(
        dimension=args.dimension,
        nheads=args.nheads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation=args.activation,
        bias=True,
        epsilon=args.eps,
    )

    assert transformerEncoder(torch.randn(batch_size, 200, args.dimension)).size() == (
        batch_size,
        200,
        args.dimension,
    ), "Encoder block is not working properly".capitalize()
