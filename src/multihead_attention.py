import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config
from scaled_dot_product import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dimension: int = 512,
        nheads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super(MultiHeadAttention, self).__init__()

        self.dimension = dimension
        self.nheads = nheads
        self.dropout = dropout
        self.bias = bias

        assert (
            self.dimension % self.nheads == 0
        ), "dimension must be divisible by nheads".capitalize()

        self.QKV = nn.Linear(
            in_features=self.dimension, out_features=3 * self.dimension, bias=self.bias
        )

        self.layers = nn.Linear(
            in_features=self.dimension, out_features=self.dimension, bias=self.bias
        )

    def forward(self, x: torch.Tensor, mask=None):
        if isinstance(x, torch.Tensor):
            QKV = self.QKV(x)

            self.query, self.key, self.value = torch.chunk(input=QKV, chunks=3, dim=-1)

            assert (
                self.query.size() == self.key.size() == self.value.size()
            ), "QKV must have the same size".capitalize()

            self.query = self.query.view(
                self.query.size(0),
                self.query.size(1),
                self.nheads,
                self.dimension // self.nheads,
            )

            self.key = self.key.view(
                self.key.size(0),
                self.key.size(1),
                self.nheads,
                self.dimension // self.nheads,
            )

            self.value = self.value.view(
                self.value.size(0),
                self.value.size(1),
                self.nheads,
                self.dimension // self.nheads,
            )

            self.query = self.query.permute(0, 2, 1, 3)
            self.key = self.key.permute(0, 2, 1, 3)
            self.value = self.value.permute(0, 2, 1, 3)

            self.attention = scaled_dot_product_attention(
                query=self.query,
                key=self.key,
                value=self.value,
                mask=mask,
            )

            self.attention = self.attention.view(
                self.attention.size(0),
                self.attention.size(2),
                self.attention.size(1) * self.attention.size(3),
            )

            assert (
                self.attention.size() == x.size()
            ), "Attention output size does not match input size"

            return self.layers(self.attention)

        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MultiHeadAttention Layer for the transformer".title()
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
        help="Number of heads in the multihead attention layer".capitalize(),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=config()["ViT"]["dropout"],
        help="Dropout rate for the multihead attention layer".capitalize(),
    )

    args = parser.parse_args()

    batch_size = config()["ViT"]["batch_size"]

    attention = MultiHeadAttention(
        dimension=args.dimension, nheads=args.nheads, dropout=args.dropout, bias=True
    )

    assert attention(torch.randn((batch_size, 200, args.dimension))).size() == (
        batch_size,
        200,
        args.dimension,
    ), "MultiHeadAttention Layer is not working properly".capitalize()
