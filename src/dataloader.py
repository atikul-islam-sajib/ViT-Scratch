import os
import sys
import zipfile
import cv2
import torch
from torchvision import transforms

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

        self.X = list()
        self.Y = list()

        try:
            self.CONFIG = config()
        except Exception as e:
            print("An error occurred while loading config file: ", e)
        else:
            self.RAW_DATA_PATH = self.CONFIG["path"]["RAW_DATA_PATH"]
            self.PROCESSED_DATA_PATH = self.CONFIG["path"]["PROCESSED_DATA_PATH"]

    def transforms(self):
        return transforms.Compose(
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.CenterCrop((self.image_size, self.image_size)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        )

    def unzip_folder(self):
        if os.path.exists(self.RAW_DATA_PATH):
            with zipfile.ZipFile(self.image_path, "r") as zip:
                zip.extractall(self.RAW_DATA_PATH)

        else:
            raise FileNotFoundError("RAW Path not found".capitalize())

    def extract_features(self):
        self.directory = os.path.join(self.RAW_DATA_PATH, "dataset")
        self.categories = config()["dataloader"]["labels"]

        for category in self.categories:
            image_path = os.path.join(self.directory, category)

            for image in os.listdir(image_path):
                image = os.path.join(image_path, image)

                if image is not None:
                    image = cv2.imread(image)
                    image = cv2.cvtColor(image, cv2.GRAY2BGR)


if __name__ == "__main__":
    loader = Loader(image_path="/Users/shahmuhammadraditrahman/Desktop/dataset.zip")
    # loader.unzip_folder()
    loader.extract_features()
