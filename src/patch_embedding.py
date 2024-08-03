import sys
import torch
import argparse
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        image_channels: int = 3,
        dimension: int = 512,
        patch_size: int = 16,
    ):
        super(PatchEmbedding, self).__init__()

        self.image_size = image_size
        self.image_channels = image_channels
        self.dimension = dimension
        self.patch_size = patch_size

        self.num_patches = (self.image_size // patch_size) ** 2

    def forward(self, x: torch.Tensor):
        pass
