from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from datasets_detection import DetectionDataset


class LitDataModule(LightningDataModule):
    def __init__(self, catalog_path, hdf5_path, batch_size):
        super().__init__()
        self.catalog_path = catalog_path
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size

    # TODO some dataset logic can be put into the datamodule, ie extracting data from the catalogue

    def train_dataloader(self):
        num_workers = 4
        shuffle = True
        test_run = False

        if test_run:
            num_workers = 1

        training_data = DetectionDataset(
            catalog_path=self.catalog_path,
            hdf5_path=self.hdf5_path,
            split="TRAIN",
        )
        training_loader = DataLoader(
            training_data,
            batch_size=self.batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )
        return training_loader

    def val_dataloader(self):
        num_workers = 4
        validation_data = DetectionDataset(
            catalog_path=self.catalog_path,
            hdf5_path=self.hdf5_path,
            split="DEV",
        )

        validation_loader = DataLoader(
            validation_data,
            batch_size=self.batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        return validation_loader

    def test_dataloader(self):
        num_workers = 4
        test_run = False
        if test_run:
            num_workers = 1
        test_data = DetectionDataset(
            catalog_path=self.catalog_path,
            hdf5_path=self.hdf5_path,
            split="TEST",
        )

        test_loader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

        return test_loader
