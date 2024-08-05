import sys
import torch
import argparse
import torch.nn as nn


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

    def l1_loss(self, model):
        pass

    def l2_loss(self, model):
        pass

    def elasticnet_loss(self, model):
        pass

    def saved_checkpoints(self, **kwargs):
        pass

    def save_train_images(self, **kwargs):
        pass

    def train(self):
        pass

    @staticmethod
    def display_history():
        pass
