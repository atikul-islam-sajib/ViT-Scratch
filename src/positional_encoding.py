import sys
import math
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config


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
    parser = argparse.ArgumentParser(
        description="Positional Encoding for the transformer".title()
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=200,
        help="Length of the sequence".capitalize(),
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=config()["ViT"]["dimension"],
        help="Dimension of the positional encoding".capitalize(),
    )
    parser.add_argument(
        "--constant",
        type=int,
        default=10000,
        help="Constant used in the positional encoding".capitalize(),
    )

    args = parser.parse_args()

    positional_encoding = PositionalEncoding(
        sequence_length=args.sequence_length,
        dimension=args.dimension,
        constant=args.constant,
    )

    assert positional_encoding(
        torch.randn((40, args.sequence_length, args.dimension))
    ).size() == (
        1,
        args.sequence_length,
        args.dimension,
    )
