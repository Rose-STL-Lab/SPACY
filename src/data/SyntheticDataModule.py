from typing import List, Dict, Any, Tuple
import torch
import lightning as L
import os
from src.data.SyntheticDataConfig import SyntheticDataConfig
from src.data.SyntheticDataset import SyntheticDataset
from src.utils.files import generate_datafolder_name
from src.utils.pylogger import RankedLogger
import torch.nn.functional as F
from torch.utils.data import DataLoader

log = RankedLogger(__name__, rank_zero_only=True)


class SyntheticDataModule(L.LightningDataModule):

    def __init__(self,
                 data_dir: str,
                 data_config: SyntheticDataConfig,
                 train_ids: List[int],
                 val_ids: List[int],
                 batch_size: int,
                 num_workers: int = 0,
                 pin_memory: bool = False
                 ):
        """
        Reads synthetic data generated
        :param data_dir: A str pointing to the location where data is stored.
        :param train_data_config: SyntheticDataConfig specifying the parameters for the training data
        :param val_data_config: SyntheticDataConfig specifying the parameters for the validation data
        :param test_data_config: SyntheticDataConfig specifying the parameters for the test data
        :param train_ids: Range of instance ids to use for training. The same instance ids are used with all settings.
        :param val_ids: Range of instance ids to use for validation. The same instance ids are used with all settings.
        :param batch_size: The batch size to use during training and validation.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
         """

        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.data_config = data_config
        self.train_ids = list(range(train_ids[0], train_ids[1]))
        self.val_ids = list(range(val_ids[0], val_ids[1]))
        self.test_ids = self.train_ids + self.val_ids

    def prepare_data(self) -> None:
        """This function verifies if synthetic data matching the description has been generated.

        Returns:
            None
        """
        log.info("Verifying if synthetic data folders exist...")

        folder_name = generate_datafolder_name(self.data_config)
        assert os.path.exists(os.path.join(
            self.data_dir, folder_name)), f"Setting {folder_name} does not exist!"

        # check if all files exist in the folder
        for id in self.test_ids:
            assert os.path.exists(os.path.join(
                self.data_dir, folder_name, f"{id}.pt")), f"File {id}.pt does not exist in {folder_name}"

        log.info("Verified that all required data files are present!")

    def setup(self, stage: str):
        """Instantiates the correct dataset depending on the stage of the problem
        
        Args:
            stage (str): The stage of the problem. Can be one of "fit", "test", or "predict".
        """
        if stage == "fit":
            # load the train and validation dataset
            self.train_dataset = SyntheticDataset(self.data_dir,
                                                  self.data_config,
                                                  self.train_ids)
            self.val_dataset = SyntheticDataset(self.data_dir,
                                                self.data_config,
                                                self.val_ids)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_dataset = SyntheticDataset(self.data_dir,
                                                 self.data_config,
                                                 self.test_ids)

        if stage == "predict":
            raise NotImplementedError

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        Returns:
            DataLoader: The train dataloader.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        Returns:
            DataLoader: The validation dataloader.
        """
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        Returns:
            DataLoader: The test dataloader.
        """
        return DataLoader(
            dataset=self.test_dataset,
            # batch_size=len(self.test_dataset),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False
        )
