import os
import math
import sys
import torch
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append("./src/")

from ViT import ViT
from utils import load, config, device_init


class Tester:
    def __init__(
        self,
        image_channels: int = 3,
        image_size: int = 224,
        labels: int = 4,
        patch_size: int = 16,
        nheads: int = 8,
        num_encoder_layers: int = 8,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
        epsilon: float = 1e-5,
        activation: str = "relu",
        bias: bool = True,
        model=None,
        device: str = "cuda",
    ):

        self.image_channels = image_channels
        self.image_size = image_size
        self.labels = labels
        self.patch_size = patch_size
        self.nheads = nheads
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.epsilon = epsilon
        self.activation = activation
        self.bias = bias
        self.model = model
        self.device = device

        self.device = device_init(device=self.device)

    def select_best_model(self):
        self.ViT = ViT(
            image_channels=self.image_channels,
            image_size=self.image_size,
            labels=self.labels,
            patch_size=self.patch_size,
            nheads=self.nheads,
            num_encoder_layers=self.num_encoder_layers,
            dropout=self.dropout,
            dim_feedforward=self.dim_feedforward,
            epsilon=self.epsilon,
            activation=self.activation,
            bias=self.bias,
        ).to(self.device)

        if self.model is None:
            state_dict = torch.load(
                os.path.join(config()["path"]["BEST_MODEL_PATH"], "best_model.pth")
            )
            self.ViT.load_state_dict(state_dict["model"])
        else:
            self.ViT.load_state_dict(self.model)

        self.ViT.eval()

    def load_dataloader(self):
        return load(
            filename=os.path.join(
                config()["path"]["PROCESSED_DATA_PATH"], "valid_dataloader.pkl"
            )
        )

    def test(self):
        plt.figure(figsize=(20, 15))

        plt.suptitle("Model Evaluation of ViT", fontsize=20)

        dataloader = self.load_dataloader()

        target, label = next(iter(dataloader))
        target = target.to(self.device)
        label = label.to(self.device)

        try:
            self.select_best_model()
        except Exception as e:
            print("An error occurred while selecting the best model: ", e)

        predicted = self.ViT(target)
        predicted = torch.argmax(predicted, dim=1)

        number_of_rows = int(target.size(0) // math.sqrt(target.size(0)))
        number_of_columns = int(target.size(0) // number_of_rows)

        labels = config()["dataloader"]["labels"]

        for index, image in enumerate(target[: number_of_rows * number_of_columns]):
            image = image.permute(1, 2, 0).detach().cpu().numpy()
            predict = predicted[index].detach().cpu().numpy()
            actual = label[index].detach().cpu().numpy()

            image = (image - image.min()) / (image.max() - image.min())

            plt.subplot(number_of_rows, number_of_columns, index + 1)
            plt.imshow(image)
            plt.title(
                "Predicted: {}\nExpected: {}".format(
                    labels[predict].capitalize(), labels[actual].capitalize()
                )
            )
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(config()["path"]["OUTPUTS_PATH"], "test_image.png"))
        plt.show()

        print(
            "The test image has been saved in the outputs folder {}".format(
                config()["path"]["OUTPUTS_PATH"]
            ).capitalize()
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test for Vit Model".title())
    parser.add_argument(
        "--image_channels",
        type=int,
        default=config()["dataloader"]["channels"],
        help="Number of image channels".capitalize(),
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=config()["dataloader"]["image_size"],
        help="Image size".capitalize(),
    )
    parser.add_argument(
        "--labels",
        type=int,
        default=len(config()["dataloader"]["labels"]),
        help="Number of labels".capitalize(),
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=config()["ViT"]["patch_size"],
        help="patch_size size".capitalize(),
    )
    parser.add_argument(
        "--nheads",
        type=int,
        default=config()["ViT"]["nheads"],
        help="Number of heads".capitalize(),
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=config()["ViT"]["num_layers"],
        help="Number of encoder layers".capitalize(),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=config()["ViT"]["dropout"],
        help="Dropout rate".capitalize(),
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=config()["ViT"]["dim_feedforward"],
        help="Dimension of feedforward".capitalize(),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=config()["ViT"]["eps"],
        help="Epsilon".capitalize(),
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=config()["ViT"]["activation"],
        help="Activation function".capitalize(),
    )
    parser.add_argument(
        "--bias",
        type=bool,
        default=True,
        help="Bias".capitalize(),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config()["trainer"]["device"],
        help="Device".capitalize(),
    )
    args = parser.parse_args()

    tester = Tester(
        image_channels=args.image_channels,
        image_size=args.image_size,
        labels=args.labels,
        patch_size=args.patch_size,
        nheads=args.nheads,
        num_encoder_layers=args.num_encoder_layers,
        dropout=args.dropout,
        dim_feedforward=args.dim_feedforward,
        epsilon=args.epsilon,
        activation=args.activation,
        bias=args.bias,
        device=args.device,
    )

    tester.test()
