import os
import sys
import torch
import mlflow
import dagshub
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR

sys.path.append(".src/")

from ViT import ViT
from helper import helpers
from utils import load, dump, config, device_init
from loss import CategoricalLoss


class Trainer:
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
        epochs: int = 100,
        lr: float = 0.0001,
        beta1: float = 0.5,
        beta2: float = 0.999,
        momentum: float = 0.9,
        step_size: int = 10,
        gamma: float = 0.85,
        threshold: int = 100,
        device: str = "cuda",
        adam: bool = True,
        SGD: bool = False,
        l1_regularization: bool = False,
        l2_regularization: bool = False,
        elasticnet_regularization: bool = False,
        mlflow: bool = False,
        verbose: bool = True,
        lr_scheduler: bool = False,
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
        self.epochs = epochs
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.momentum = momentum
        self.step_size = step_size
        self.gamma = gamma
        self.threshold = threshold
        self.device = device
        self.adam = adam
        self.SGD = SGD
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization
        self.elasticnet_regularization = elasticnet_regularization
        self.mlflow = mlflow
        self.verbose = verbose
        self.lr_scheduler = lr_scheduler

        self.init = helpers(
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
            lr=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
            momentum=self.momentum,
            adam=self.adam,
            SGD=self.SGD,
        )

        self.device = device_init(device=self.device)

        self.train_dataloader = self.init["train_dataloader"]
        self.valid_dataloader = self.init["valid_dataloader"]

        self.model = self.init["model"].to(self.device)

        self.optimizer = self.init["optimizer"]
        self.criterion = self.init["criterion"]

        assert (
            self.init["train_dataloader"].__class__ == torch.utils.data.DataLoader
        ), "Dataloader is not a torch.utils.data.DataLoader".capitalize()
        assert (
            self.init["valid_dataloader"].__class__ == torch.utils.data.DataLoader
        ), "Dataloader is not a torch.utils.data.DataLoader".capitalize()

        assert self.init["model"].__class__ == ViT, "Model is not a ViT".capitalize()
        assert (
            self.init["optimizer"].__class__ == torch.optim.Adam
        ), "Optimizer is not a Adam".capitalize()
        assert (
            self.init["criterion"].__class__ == CategoricalLoss
        ), "Loss is not a CategoricalLoss".capitalize()

        if self.mlflow:
            dagshub.init(
                repo_owner=config()["MLFLow"]["MLFLOW_USERNAME"],
                repo_name="ViT-Scratch",
                mlflow=True,
            )

        if self.lr_scheduler:
            self.scheduler = StepLR(
                optimizer=self.optimizer, step_size=self.step_size, gamma=self.gamma
            )

        self.loss = float("inf")

        self.total_train_loss = []
        self.total_valid_loss = []
        self.total_train_accuracy = []
        self.total_valid_accuracy = []

        self.model_history = {
            "train_loss": [],
            "valid_loss": [],
            "train_accuracy": [],
            "valid_accuracy": [],
        }

    def l1_loss(self, model: ViT):
        if isinstance(model, ViT):
            return sum(torch.norm(params, 1) for params in model.parameters())
        else:
            raise ValueError("Model is not a ViT".capitalize())

    def l2_loss(self, model: ViT):
        if isinstance(model, ViT):
            return sum(torch.norm(params, 2) for params in model.parameters())
        else:
            raise ValueError("Model is not a ViT".capitalize())

    def elasticnet_loss(self, model: ViT):
        if isinstance(model, ViT):
            return sum(
                torch.norm(params, 1) + torch.norm(params, 2)
                for params in model.parameters()
            )
        else:
            raise ValueError("Model is not a ViT".capitalize())

    def saved_checkpoints(self, **kwargs):
        epoch = kwargs["epoch"]
        train_loss = kwargs["train_loss"]

        if self.loss > train_loss:
            self.loss = train_loss

            torch.save(
                {
                    "model": self.model.state_dict(),
                    "epoch": epoch,
                    "loss": train_loss,
                },
                os.path.join(config()["path"]["BEST_MODEL_PATH"], "best_model.pth"),
            )

        torch.save(
            self.model.state_dict(),
            os.path.join(
                config()["path"]["TRAIN_MODELS_PATH"], "model{}.pth".format(epoch + 1)
            ),
        )

    def save_train_images(self, **kwargs):
        pass

    def display_progress(self, **kwargs):
        epoch = kwargs["epoch"]

        train_loss = kwargs["train_loss"]
        valid_loss = kwargs["valid_loss"]

        train_actual = kwargs["train_actual"]
        valid_actual = kwargs["valid_actual"]

        train_target = kwargs["train_target"]
        valid_target = kwargs["valid_target"]

        if self.verbose:
            print(
                "Epochs: [{}/{}] - train_loss: {:.4f} - valid_loss: {:.4f} - train_accuracy: {:.4f} - valid_accuracy: {:.4f}".format(
                    epoch + 1,
                    self.epochs,
                    train_loss,
                    valid_loss,
                    accuracy_score(train_actual, train_target),
                    accuracy_score(valid_actual, valid_target),
                )
            )

        else:
            print("Epochs: [{}/{}] is completed.".format(epoch + 1, self.epochs))

    def train(self):
        with mlflow.start_run(
            description="An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale".title()
        ) as run:
            for epoch in tqdm(range(self.epochs)):
                self.train_loss = list()
                self.valid_loss = list()

                self.train_actual = list()
                self.train_target = list()
                self.valid_actual = list()
                self.valid_target = list()

                for _, (X, y) in enumerate(self.train_dataloader):
                    X = X.to(self.device)
                    y = y.to(self.device)

                    self.optimizer.zero_grad()

                    predicted = self.model(X)

                    self.train_actual.extend(
                        torch.argmax(predicted, dim=1).cpu().detach().numpy()
                    )
                    self.train_target.extend(y.cpu().detach().numpy())

                    loss = self.criterion(predicted, y)

                    self.train_loss.append(loss.item())

                    if self.l1_regularization:
                        loss = 0.001 * self.l1_loss(model=self.model)
                    elif self.l2_regularization:
                        loss = 0.001 * self.l2_loss(model=self.model)
                    elif self.elasticnet_regularization:
                        loss = 0.001 * self.elasticnet_loss(model=self.model)

                    loss.backward()
                    self.optimizer.step()

                for _, (X, y) in enumerate(self.valid_dataloader):
                    X = X.to(self.device)
                    y = y.to(self.device)

                    predicted = self.model(X)
                    predicted = torch.argmax(input=predicted, dim=1)
                    predicted = predicted.float()

                    loss = self.criterion(predicted, y.float())

                    self.valid_actual.extend(predicted.cpu().detach().numpy())
                    self.valid_target.extend(y.cpu().detach().numpy())

                    self.valid_loss.append(loss.item())

                if self.lr_scheduler:
                    self.scheduler.step()

                self.display_progress(
                    epoch=epoch,
                    train_loss=np.mean(self.train_loss),
                    valid_loss=np.mean(self.valid_loss),
                    train_actual=self.train_actual,
                    train_target=self.train_target,
                    valid_actual=self.valid_actual,
                    valid_target=self.valid_target,
                )

                if epoch % self.threshold == 0:
                    self.saved_checkpoints(
                        epoch=epoch, train_loss=np.mean(self.train_loss)
                    )

                self.total_train_loss.append(np.mean(self.train_loss))
                self.total_valid_loss.append(np.mean(self.valid_loss))

                self.total_train_accuracy.append(
                    accuracy_score(self.train_target, self.train_actual)
                )
                self.total_valid_accuracy.append(
                    accuracy_score(self.valid_target, self.valid_actual)
                )

                mlflow.log_params(
                    {
                        "image_channels": self.image_channels,
                        "image_size": self.image_size,
                        "labels": self.labels,
                        "patch_size": self.patch_size,
                        "nheads": self.nheads,
                        "num_encoder_layers": self.num_encoder_layers,
                        "dropout": self.dropout,
                        "dim_feedforward": self.dim_feedforward,
                        "epsilon": self.epsilon,
                        "activation": self.activation,
                        "bias": self.bias,
                        "epochs": self.epochs,
                        "lr": self.lr,
                        "beta1": self.beta1,
                        "beta2": self.beta2,
                        "momentum": self.momentum,
                        "step_size": self.step_size,
                        "gamma": self.gamma,
                        "threshold": self.threshold,
                        "device": self.device,
                        "adam": self.adam,
                        "SGD": self.SGD,
                        "l1_regularization": self.l1_regularization,
                        "l2_regularization": self.l2_regularization,
                        "elasticnet_regularization": self.elasticnet_regularization,
                        "mlflow": self.mlflow,
                        "verbose": self.verbose,
                    }
                )

                mlflow.log_metric(
                    "train_loss", np.mean(self.train_loss), step=epoch + 1
                )
                mlflow.log_metric(
                    "valid_loss", np.mean(self.valid_loss), step=epoch + 1
                )

            try:
                pd.DataFrame(
                    {
                        "train_loss": self.total_train_loss,
                        "valid_loss": self.total_valid_loss,
                        "train_accuracy": self.total_train_accuracy,
                        "valid_accuracy": self.total_valid_accuracy,
                    }
                ).to_csv(os.path.join(config()["path"]["FILES_PATH"], "history.csv"))

                self.model_history["train_loss"].append(self.total_train_loss)
                self.model_history["valid_loss"].append(self.total_valid_loss)
                self.model_history["train_accuracy"].append(self.total_train_accuracy)
                self.model_history["valid_accuracy"].append(self.total_valid_accuracy)

                dump(
                    value=self.model_history,
                    filename=os.path.join(
                        config()["path"]["METRICS_PATH"], "history.pkl"
                    ),
                )
            except Exception as e:
                print("An error occurred: ", e)

            else:
                mlflow.pytorch.log_model(self.model, "model")

    @staticmethod
    def display_history():
        history = load(
            filename=os.path.join(config()["path"]["METRICS_PATH"], "history.pkl")
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].plot(history["train_loss"], label="Train Loss")
        axes[0].plot(history["valid_loss"], label="Valid Loss")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        axes[1].plot(history["train_accuracy"], label="Train Accuracy")
        axes[1].plot(history["valid_accuracy"], label="Valid Accuracy")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(config()["path"]["METRICS_PATH"], "history.png"))
        plt.show()

        print("The history has been saved to: ", config()["path"]["METRICS_PATH"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ViT model".title())
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
        "--epochs",
        type=int,
        default=config()["trainer"]["epochs"],
        help="Number of epochs".capitalize(),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config()["trainer"]["lr"],
        help="Learning rate".capitalize(),
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=config()["trainer"]["beta1"],
        help="Beta1".capitalize(),
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=config()["trainer"]["beta2"],
        help="Beta2".capitalize(),
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=config()["trainer"]["momentum"],
        help="Momentum".capitalize(),
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=config()["trainer"]["step_size"],
        help="Step size".capitalize(),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=config()["trainer"]["gamma"],
        help="Gamma".capitalize(),
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=config()["trainer"]["threshold"],
        help="Threshold".capitalize(),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config()["trainer"]["device"],
        help="Device initialization".capitalize(),
    )
    parser.add_argument(
        "--adam",
        type=bool,
        default=config()["trainer"]["adam"],
        help="Adam optimizer".capitalize(),
    )
    parser.add_argument(
        "--SGD",
        type=bool,
        default=config()["trainer"]["SGD"],
        help="SGD optimizer".capitalize(),
    )
    parser.add_argument(
        "--l1_regularization",
        type=bool,
        default=config()["trainer"]["l1_regularization"],
        help="L1 regularization".capitalize(),
    )
    parser.add_argument(
        "--l2_regularization",
        type=bool,
        default=config()["trainer"]["l2_regularization"],
        help="L2 regularization".capitalize(),
    )
    parser.add_argument(
        "--elasticnet_regularization",
        type=bool,
        default=config()["trainer"]["elasticnet_regularization"],
        help="Elasticnet regularization".capitalize(),
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=config()["trainer"]["verbose"],
        help="Verbose".capitalize(),
    )
    parser.add_argument(
        "--mlflow",
        type=bool,
        default=config()["trainer"]["mlflow"],
        help="MLflow".capitalize(),
    )

    args = parser.parse_args()

    trainer = Trainer(
        image_channels=args.image_channels,
        image_size=args.image_size,
        labels=args.labels,
        patch_size=args.patch_size,
        nheads=args.nheads,
        num_encoder_layers=args.num_encoder_layers,
        dropout=args.dropout,
        device=args.device,
        dim_feedforward=args.dim_feedforward,
        epsilon=args.epsilon,
        activation=args.activation,
        bias=args.bias,
        epochs=args.epochs,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        momentum=args.momentum,
        step_size=args.step_size,
        gamma=args.gamma,
        threshold=args.threshold,
        l1_regularization=args.l1_regularization,
        l2_regularization=args.l2_regularization,
        elasticnet_regularization=args.elasticnet_regularization,
        verbose=args.verbose,
        mlflow=args.mlflow,
    )

    trainer.train()

    Trainer.display_history()
