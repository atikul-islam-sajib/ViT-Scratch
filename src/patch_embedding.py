import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config, device_init
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

        self.device = device_init(device=config()["trainer"]["device"])

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

            class_token = torch.randn(x.size(0), 1, x.size(2)).to(self.device)

            x = torch.cat((class_token, x), dim=1)

            x = torch.add(x, self.positional_encoding(x).to(self.device))

            return x

        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch Embedding for ViT".title())
    parser.add_argument(
        "--image_channels",
        type=int,
        default=config()["dataloader"]["channels"],
        help="Number of channels in the input image".capitalize(),
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=config()["dataloader"]["image_size"],
        help="Size of the input image".capitalize(),
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=config()["ViT"]["patch_size"],
        help="Size of the patch".capitalize(),
    )

    args = parser.parse_args()

    batch_size = config()["dataloader"]["batch_size"]

    patch_embedding = PatchEmbedding(
        image_channels=args.image_channels,
        image_size=args.image_size,
        patch_size=args.patch_size,
    )

    number_of_patches = (args.image_size // args.patch_size) ** 2
    number_of_dimension = (args.patch_size**2) * args.image_channels

    assert patch_embedding(
        torch.randn(batch_size, args.image_channels, args.image_size, args.image_size)
    ).size() == (batch_size, number_of_patches + 1, number_of_dimension)
