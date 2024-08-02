import os
import sys
import zipfile
import cv2
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

sys.path.append("./src/")

from utils import config, dump


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

    def dataset_split(self, X: list, y: list):
        if isinstance(X, list) and isinstance(y, list):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.split_size, random_state=42
            )

            return {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            }

        else:
            raise TypeError("X and y must be list".capitalize())

    def transforms(self):
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.CenterCrop((self.image_size, self.image_size)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
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

        for category in tqdm(self.categories):
            image_path = os.path.join(self.directory, category)

            for image in os.listdir(image_path):
                image = os.path.join(image_path, image)

                if (image is not None) and (image.endswith((".jpg", ".png", ".jpeg"))):
                    image = cv2.imread(image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    image = self.transforms()(Image.fromarray(image))
                    label = self.categories.index(category)

                    self.X.append(image)
                    self.Y.append(label)

                else:
                    print("Image not found".capitalize())

        assert len(self.X) == len(
            self.Y
        ), "Image size and Label size not equal".capitalize()

        try:
            dataset = self.dataset_split(X=self.X, y=self.Y)
        except TypeError as e:
            print("An error occured: ", e)
        except Exception as e:
            print("An error occured: ", e)

        else:
            return dataset

    def create_dataloader(self):
        dataset = self.extract_features()

        train_dataloader = DataLoader(
            dataset=list(zip(dataset["X_train"], dataset["y_train"])),
            batch_size=self.batch_size,
            shuffle=True,
        )

        valid_dataloader = DataLoader(
            dataset=list(zip(dataset["X_test"], dataset["y_test"])),
            batch_size=self.batch_size,
            shuffle=True,
        )

        for value, filename in [
            (train_dataloader, "train_dataloader"),
            (valid_dataloader, "valid_dataloader"),
        ]:
            dump(
                value=value,
                filename=os.path.join(
                    self.PROCESSED_DATA_PATH, "{}.pkl".format(filename)
                ),
            )

        print("Dataloader is saved in the folder {}".format(self.PROCESSED_DATA_PATH))


if __name__ == "__main__":
    loader = Loader(image_path="/Users/shahmuhammadraditrahman/Desktop/dataset.zip")
    # loader.unzip_folder()
    # loader.extract_features()
    loader.create_dataloader()
