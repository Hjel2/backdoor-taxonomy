import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize
from torchvision.datasets import CIFAR10
import pytorch_lightning as pl

__all__ = [
    "Cifar10Data",
    "test_data",
]


class Cifar10Data(pl.LightningDataModule):
    def __init__(self, data_dir = '/local/scratch/hjel2/data' if torch.cuda.is_available() else '~/Documents/Code/data', batch_size = 128, num_workers = 8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        μ = (0.4914, 0.4822, 0.4465)
        σ = (0.2023, 0.1994, 0.2010)

        self.transform_train = Compose(
            [
                RandomCrop(32, padding = 4),
                RandomHorizontalFlip(),
                ToTensor(),
                # Normalize(μ, σ),
            ]
        )
        self.transform_test = Compose(
            [
                ToTensor(),
                # Normalize(μ, σ),
            ]
        )

    def setup(self, stage: str) -> None:
        self.train = CIFAR10(
            self.data_dir,
            train=True,
            transform=self.transform_train
        )

        self.test = CIFAR10(
            self.data_dir,
            train=False,
            transform=self.transform_test
        )

    def train_dataloader(self):
        return DataLoader(
            dataset = self.train,
            batch_size = 128,
            shuffle = True,
            drop_last = True,
            pin_memory = True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset = self.test,
            batch_size = 128,
            pin_memory = True,
        )

    def val_dataloader(self):
        return self.test_dataloader()


def test_data():
    data_dir = '/local/scratch/hjel2/data' if torch.cuda.is_available() else '~/Documents/Code/data'
    CIFAR10(
        data_dir,
        train=False,
        transform=ToTensor()
    )
