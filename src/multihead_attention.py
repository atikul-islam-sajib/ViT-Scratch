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

    def forward(self, x: torch.Tensor, mask=None):
        if isinstance(x, torch.Tensor):
            pass


if __name__ == "__main__":
    pass
