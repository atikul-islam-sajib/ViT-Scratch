import sys
import yaml
import joblib
import torch


def dump(value=None, filename=None):
    if (value is not None) and (filename is not None):
        joblib.dump(value=value, filename=filename)

    else:
        raise ValueError("Both value and filename must be provided".capitalize())


def load(filename=None):
    if filename is not None:
        return joblib.load(filename=filename)

    else:
        raise ValueError("Filename must be provided".capitalize())


def device_init(device="cuda"):
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    elif device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    else:
        return torch.device("cpu")


def config():
    with open("./config.yml", "r") as file:
        return yaml.safe_load(file)
