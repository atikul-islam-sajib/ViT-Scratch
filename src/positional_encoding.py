import sys
import math
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")  # 40, 200, 512


class PositionalEncoding(nn.Module):
    def __init__(
        self, sequence_length: int = 200, dimension: int = 512, constant: int = 10000
    ):
        super(PositionalEncoding, self).__init__()

        self.sequence_length = sequence_length
        self.dimension = dimension
        self.constant = constant

        positional_encoding = torch.randn((self.sequence_length, self.dimension))

        for pos in range(self.sequence_length):
            for index in range(self.dimension):
                if index % 2 == 0:
                    positional_encoding[pos, index] = math.sin(
                        pos / (self.constant ** ((2 * index) / self.dimension))
                    )
                else:
                    positional_encoding[pos, index] = math.cos(
                        pos / (self.constant ** ((2 * index) / self.dimension))
                    )

        self.positional_encoding = nn.Parameter(positional_encoding, requires_grad=True)

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            return self.positional_encoding.unsqueeze(0)[:, : x.size(1), :]

        else:
            raise TypeError("Input must be a torch.Tensor")


if __name__ == "__main__":
    positional_encoding = PositionalEncoding(
        sequence_length=200, dimension=512, constant=10000
    )

    print(positional_encoding(torch.randn((400, 200, 512))).size())
