import sys
import zipfile
import cv2
import torch

sys.path.append("./src/")


class Loader:
    def __init__(
        self,
        image_path: str = None,
        image_channels: int = 3,
        image_size: int = 128,
        batch_size: int = 64,
        split_size: float = 0.25,
    ):
        self.image_path = image_path
        self.image_channels = image_channels
        self.image_size = image_size
        self.batch_size = batch_size
        self.split_size = split_size

    def unzip_folder(self):
        with zipfile.ZipFile(self.image_path, "r") as zip:
            zip.extractall()
