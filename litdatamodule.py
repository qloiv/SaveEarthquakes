from pytorch_lightning import LightningDataModule
from torch.nn import (
    CrossEntropyLoss,
)

from datasets import SeismoDataset, DataLoader


class LitDataModule(LightningDataModule):
    def __init__(self, catalog_path, hdf5_path):
        super().__init__()
        self.criterion = CrossEntropyLoss()
        self.catalog_path = catalog_path
        self.hdf5_path = hdf5_path

    # TODO some dataset logic can be put into the datamodule, ie extracting data from the catalogue

    def train_dataloader(self):
        batch_size = 64
        num_workers = 4
        shuffle = True
        test_run = False

        if test_run:
            num_workers = 1

        training_data = SeismoDataset(
            catalog_path=self.catalog_path,
            hdf5_path=self.hdf5_path,
            split="TRAIN",
            test_run=test_run,
        )
        training_loader = DataLoader(
            training_data,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )
        return training_loader

    def val_dataloader(self):
        batch_size = 64
        num_workers = 4
        test_run = False
        validation_data = SeismoDataset(
            catalog_path=self.catalog_path,
            hdf5_path=self.hdf5_path,
            split="DEV",
            test_run=test_run,
        )

        validation_loader = DataLoader(
            validation_data,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        return validation_loader

    def test_dataloader(self):
        batch_size = 64
        num_workers = 4
        test_run = False
        if test_run:
            num_workers = 1
        test_data = SeismoDataset(
            catalog_path=self.catalog_path,
            hdf5_path=self.hdf5_path,
            split="TEST",
            test_run=test_run,
        )

        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

        return test_loader
