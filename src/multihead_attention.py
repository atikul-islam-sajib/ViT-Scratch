import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


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

        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    attention = MultiHeadAttention(dimension=512, nheads=8, dropout=0.1, bias=True)
    print(attention(torch.randn((40, 200, 512))).size())
