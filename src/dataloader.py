import os
import sys
import zipfile
import cv2
import torch

sys.path.append("./src/")

from utils import config


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
        raw_data_path = config()["path"]["RAW_DATA_PATH"]

        if os.path.exists(raw_data_path):
            with zipfile.ZipFile(self.image_path, "r") as zip:
                zip.extractall(raw_data_path)

        else:
            raise FileNotFoundError("RAW Path not found".capitalize())


if __name__ == "__main__":
    loader = Loader(image_path="/Users/shahmuhammadraditrahman/Desktop/dataset.zip")
    loader.unzip_folder()
