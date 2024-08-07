import sys
import torch
import argparse

sys.path.append("./src/")

from utils import config
from dataloader import Loader
from trainer import Trainer
from tester import Tester


def cli():
    parser = argparse.ArgumentParser(description="CLI for ViT model".title())
    parser.add_argument(
        "--image_path",
        type=str,
        default=config()["dataloader"]["image_path"],
        help="Path to the image dataset".capitalize(),
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=config()["dataloader"]["channels"],
        help="Number of channels in the image".capitalize(),
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=config()["dataloader"]["image_size"],
        help="Size of the image".capitalize(),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config()["dataloader"]["batch_size"],
        help="Batch size for the dataloader".capitalize(),
    )
    parser.add_argument(
        "--split_size",
        type=float,
        default=config()["dataloader"]["split_size"],
        help="Split size for the dataloader".capitalize(),
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
    parser.add_argument("--train", action="store_true", help="Train".capitalize())
    parser.add_argument("--test", action="store_true", help="Test".capitalize())

    args = parser.parse_args()

    loader = Loader(
        image_path=args.image_path,
        image_channels=args.channels,
        image_size=args.image_size,
        batch_size=args.batch_size,
        split_size=args.split_size,
    )

    trainer = Trainer(
        image_channels=args.channels,
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

    tester = Tester(
        image_channels=args.channels,
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

    if args.train:
        # loader.unzip_folder()
        loader.create_dataloader()

        try:
            Loader.display_images()
        except FileNotFoundError as e:
            print("An error is occcured", e)
        except Exception as e:
            print("An error is occcured", e)

        try:
            Loader.dataset_details()
        except FileNotFoundError as e:
            print("An error is occcured", e)
        except Exception as e:
            print("An error is occcured", e)

        try:
            trainer.train()
        except Exception as e:
            print("An error is occcured", e)

        else:
            Trainer.display_history()

    else:
        tester.test()


if __name__ == "__main__":
    cli()
