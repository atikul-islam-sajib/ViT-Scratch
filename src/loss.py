import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config


class CategoricalLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super(CategoricalLoss, self).__init__()

        self.reduction = reduction

        self.criterion = nn.CrossEntropyLoss(reduction=self.reduction)

    def forward(self, actual: torch.Tensor, predicted: torch.Tensor):
        if isinstance(actual, torch.Tensor) and isinstance(predicted, torch.Tensor):
            return self.criterion(actual, predicted)

        else:
            raise TypeError(
                "Both actual and predicted must be torch.Tensor".capitalize()
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loss function for the ViT".title())
    parser.add_argument(
        "--reduction",
        type=str,
        default="mean",
        help="The reduction method for the loss function".capitalize(),
    )
    args = parser.parse_args()

    loss = CategoricalLoss(reduction=args.reduction)

    batch_size = config()["dataloader"]["batch_size"]
    total_labels = len(config()["dataloader"]["labels"])

    actual = torch.randn((batch_size, total_labels))
    predicted = torch.randn((batch_size, total_labels))

    actual = torch.softmax(actual, dim=1)
    predicted = torch.softmax(predicted, dim=1)

    actual = torch.argmax(actual, dim=1).float()
    predicted = torch.argmax(predicted, dim=1).float()

    assert isinstance(
        loss(actual, predicted), torch.Tensor
    ), "Loss is not a torch.Tensor".capitalize()
