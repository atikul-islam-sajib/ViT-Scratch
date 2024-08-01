import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config
from encoder_block import TransformerEncoderBlock


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dimension: int = 512,
        nheads: int = 8,
        num_encoder_layers: int = 8,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
        epsilon: float = 1e-5,
        activation: str = "relu",
        bias: bool = True,
    ):

        super(TransformerEncoder, self).__init__()

        self.dimension = dimension
        self.nheads = nheads
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.epsilon = epsilon
        self.activation = activation
        self.bias = bias

        self.transformerEncoder = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    dimension=self.dimension,
                    nheads=self.nheads,
                    dropout=self.dropout,
                    dim_feedforward=self.dim_feedforward,
                    epsilon=self.epsilon,
                    activation=self.activation,
                    bias=self.bias,
                )
                for _ in range(self.num_encoder_layers)
            ]
        )

    def forward(self, x: torch.Tensor, mask=None):
        if isinstance(x, torch.Tensor):
            for layer in self.transformerEncoder:
                x = layer(x=x, mask=mask)

            return x

        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transformer Encoder block for ViT".title()
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
    parser.add_argument(
        "--num_layers",
        type=int,
        default=config()["ViT"]["num_layers"],
        help="Number of layers in the transformer encoder".capitalize(),
    )

    args = parser.parse_args()

    batch_size = config()["ViT"]["batch_size"]

    transformerEncoder = TransformerEncoder(
        dimension=args.dimension,
        nheads=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.num_layers,
        dropout=args.dropout,
        activation=args.activation,
        bias=True,
        epsilon=args.eps,
    )

    assert transformerEncoder(torch.randn(batch_size, 200, 512)).size() == (
        batch_size,
        200,
        args.dimension,
    ), "TransformerEncoder output size is incorrect"

    print("TransformerEncoder test passed")
