from torch.utils.data import Dataset
from src.data.SSTDataConfig import SSTDataConfig
from src.utils.files import generate_datafolder_name
from typing import List
import torch
import os


class SSTDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 data_config: SSTDataConfig,
                 ids: List[int]):
        """
        Initializes sst dataset class.

        :param data_dir: Path to the data files
        :param data_config: Object of type DataConfig
        :param ids: List of ids to include
        """

        self.data_dir = data_dir
        self.data_config = data_config
        self.ids = ids
        self.num_samples = len(self.ids)
        self.folder_path = os.path.join(
            self.data_dir, generate_datafolder_name(data_config))

    def __len__(self):
        """ Returns the number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(self, idx):
        """ Get the data sample at index idx.

        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]: A tuple containing the data sample,
            and empty lists for the other return values.
        """
        # obtain the correct data config
        sample_id = self.ids[idx]
        # open the corresponding file
        X = torch.load(os.path.join(self.folder_path, f"{sample_id}.pt"))
        if self.data_config.num_variates > 1:
            X = X.squeeze()

        return X.float(), [], [], []
