import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from positional_encoding import PositionalEncoding


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        image_channels: int = 3,
        patch_size: int = 16,
    ):
        super(PatchEmbedding, self).__init__()

        self.image_size = image_size
        self.image_channels = image_channels
        self.patch_size = patch_size

        self.num_patches = (self.image_size // patch_size) ** 2
        self.num_dimension = (self.patch_size**2) * self.image_channels

        self.patcher = nn.Conv2d(
            in_channels=self.image_channels,
            out_channels=self.num_dimension,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=self.patch_size // self.patch_size,
        )

        self.positional_encoding = PositionalEncoding(
            sequence_length=self.num_patches + 1, dimension=self.num_dimension
        )

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            x = self.patcher(x)
            x = x.view(x.size(0), x.size(1), -1)
            x = x.permute(0, 2, 1)

            class_token = torch.randn(x.size(0), 1, x.size(2))

            x = torch.cat((class_token, x), dim=1)

            x = torch.add(x, self.positional_encoding(x))

            return x

        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    patch_embedding = PatchEmbedding(image_channels=3, image_size=224, patch_size=16)
    print(patch_embedding(torch.randn(64, 3, 224, 224)).size())
