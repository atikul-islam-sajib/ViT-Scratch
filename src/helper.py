import os
import sys
import torch
import torch.optim as optim

sys.path.append("./src/")

from ViT import ViT
from utils import config, load
from loss import CategoricalLoss


def load_dataloader():
    if os.path.exists(config()["path"]["PROCESSED_DATA_PATH"]):
        train_dataloader = load(
            filename=os.path.join(
                config()["path"]["PROCESSED_DATA_PATH"], "train_dataloader.pkl"
            )
        )
        valid_dataloader = load(
            filename=os.path.join(
                config()["path"]["PROCESSED_DATA_PATH"], "valid_dataloader.pkl"
            )
        )

        return {
            "train_dataloader": train_dataloader,
            "valid_dataloader": valid_dataloader,
        }


def helpers(**kwargs):
    image_channels = kwargs["image_channels"]
    image_size = kwargs["image_size"]
    labels = kwargs["labels"]
    patch_size = kwargs["patch_size"]
    nheads = kwargs["nheads"]
    num_encoder_layers = kwargs["num_encoder_layers"]
    dropout = kwargs["dropout"]
    dim_feedforward = kwargs["dim_feedforward"]
    epsilon = kwargs["epsilon"]
    activation = kwargs["activation"]
    bias = kwargs["bias"]
    lr = kwargs["lr"]
    beta1 = kwargs["beta1"]
    beta2 = kwargs["beta2"]
    momentum = kwargs["momentum"]
    adam = kwargs["adam"]
    SGD = kwargs["SGD"]

    model = ViT(
        image_channels=image_channels,
        image_size=image_size,
        labels=labels,
        patch_size=patch_size,
        nheads=nheads,
        num_encoder_layers=num_encoder_layers,
        dropout=dropout,
        dim_feedforward=dim_feedforward,
        epsilon=epsilon,
        activation=activation,
        bias=bias,
    )

    if adam:
        optimizer = optim.Adam(params=model.parameters(), lr=lr, betas=(beta1, beta2))

    else:
        optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)

    try:
        dataloader = load_dataloader()
    except FileNotFoundError as e:
        print("An error is occurred: ", e)
    except Exception as e:
        print("An error is occurred: ", e)

    try:
        loss = CategoricalLoss(reduction="mean")
    except Exception as e:
        print("An error is occurred: ", e)

    return {
        "train_dataloader": dataloader["train_dataloader"],
        "valid_dataloader": dataloader["valid_dataloader"],
        "model": model,
        "optimizer": optimizer,
        "criterion": loss,
    }


if __name__ == "__main__":
    init = helpers(
        image_channels=config()["dataloader"]["channels"],
        image_size=config()["dataloader"]["image_size"],
        labels=len(config()["dataloader"]["labels"]),
        patch_size=config()["ViT"]["patch_size"],
        nheads=config()["ViT"]["nheads"],
        num_encoder_layers=config()["ViT"]["num_layers"],
        dropout=config()["ViT"]["dropout"],
        dim_feedforward=config()["ViT"]["dim_feedforward"],
        epsilon=config()["ViT"]["eps"],
        activation=config()["ViT"]["activation"],
        bias=True,
        lr=config()["trainer"]["lr"],
        beta1=config()["trainer"]["beta1"],
        beta2=config()["trainer"]["beta2"],
        momentum=config()["trainer"]["momentum"],
        adam=config()["trainer"]["adam"],
        SGD=config()["trainer"]["SGD"],
    )

    assert (
        init["train_dataloader"].__class__ == torch.utils.data.DataLoader
    ), "Dataloader is not a torch.utils.data.DataLoader".capitalize()
    assert (
        init["valid_dataloader"].__class__ == torch.utils.data.DataLoader
    ), "Dataloader is not a torch.utils.data.DataLoader".capitalize()

    assert init["model"].__class__ == ViT, "Model is not a ViT".capitalize()
    assert (
        init["optimizer"].__class__ == torch.optim.Adam
    ), "Optimizer is not a Adam".capitalize()
    assert (
        init["criterion"].__class__ == CategoricalLoss
    ), "Loss is not a CategoricalLoss".capitalize()
