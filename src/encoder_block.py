import sys
import torch
import argparse
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dimension: int = 512,
        nheads: int = 8,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
        activation: str = "relu",
    ):
        super(EncoderBlock, self).__init__()

        self.dimension = dimension
        self.nheads = nheads
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.activation = activation

    def forward(self, x: torch.Tensor, mask=None):
        if isinstance(x, torch.Tensor):
            pass
