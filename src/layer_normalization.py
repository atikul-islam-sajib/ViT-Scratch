import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")  # 40, 200, 512

from utils import config


class LayerNormalization(nn.Module):
    def __init__(self, normalized_shape: int = 512, epsilon: float = 1e-05):
        super(LayerNormalization, self).__init__()

        self.normalized_shape = normalized_shape
        self.epsilon = epsilon

        self.gamma = nn.Parameter(torch.ones((normalized_shape,)))
        self.beta = nn.Parameter(torch.zeros((normalized_shape,)))

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            self.mean = torch.mean(input=x, dim=-1)
            self.variance = torch.var(input=x, dim=-1)

            self.mean = self.mean.unsqueeze(-1)
            self.variance = self.variance.unsqueeze(-1)

            return (
                self.gamma * (x - self.mean) / torch.sqrt(self.variance + self.epsilon)
                + self.beta
            )

        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Layer Normalization for the Transformer".title()
    )
    parser.add_argument(
        "--normalized_shape",
        type=int,
        default=config()["ViT"]["dimension"],
        help="The shape of the input tensor".capitalize(),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=config()["ViT"]["eps"],
        help="The epsilon value for the variance".capitalize(),
    )

    args = parser.parse_args()

    batch_size = config()["ViT"]["batch_size"]

    layer_norm = LayerNormalization(
        normalized_shape=args.normalized_shape, epsilon=args.epsilon
    )

    assert layer_norm(torch.rand((batch_size, 200, args.normalized_shape))).size() == (
        batch_size,
        200,
        args.normalized_shape,
    ), "Layer Normalization failed".capitalize()
