import sys
import torch
import argparse
import torch.nn as nn

sys.path.append(".src/")

from ViT import ViT
from helper import helpers
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
        verbose: bool = True,
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
        self.verbose = verbose

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

        self.train_dataloader = self.init["train_dataloader"]
        self.valid_dataloader = self.init["valid_dataloader"]
        self.model = self.init["model"]
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
        pass

    def save_train_images(self, **kwargs):
        pass

    def train(self):
        pass

    @staticmethod
    def display_history():
        pass


if __name__ == "__main__":
    trainer = Trainer()
