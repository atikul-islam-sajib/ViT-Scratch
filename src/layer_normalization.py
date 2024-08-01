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
    layer_norm = LayerNormalization(normalized_shape=512, epsilon=1e-05)

    print(layer_norm(torch.rand((40, 200, 512))).size())
