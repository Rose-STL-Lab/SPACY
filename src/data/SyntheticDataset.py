from torch.utils.data import Dataset
from src.data.SyntheticDataConfig import SyntheticDataConfig
from src.utils.files import generate_datafolder_name
from typing import List
import torch
import os


class SyntheticDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 data_config: SyntheticDataConfig,
                 ids: List[int]):
        """Initializes synthetic dataset class.

        Args:
            data_dir (str): Path to the data files.
            data_config (SyntheticDataConfig): Object of type SyntheticDataConfig.
            ids (List[int]): List of ids to include.
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
        G = torch.load(os.path.join(self.folder_path, f"graph.pt"))

        ########################################################################
        # Deprecated: this is never used in experiments in the paper (SPACY does perform well on these data)
        if self.data_config.model == "savar":
            mode_weights = torch.load(os.path.join(
                self.folder_path, f"mode_weights.pt"))
            mapping = torch.load(os.path.join(
                self.folder_path, f"spatial_mapping.pt")) 
            v,n,_,_ = mode_weights.shape
            mode_weights = mode_weights.view(v,n,-1)
            mode_inv = torch.linalg.pinv(mode_weights)
            Z_gt = torch.einsum('vtl, vln->tn', X, mode_inv)
            spatial_factors = mode_weights
        elif self.data_config.model == "varimax_spacy": 
            Z_gt = torch.load(os.path.join(self.folder_path, f"{sample_id}_latent.pt"))
            spatial_factors = torch.zeros((1))
        elif self.data_config.model == "cdsd":
            Z_gt = torch.load(os.path.join(self.folder_path, f"cdsd/{sample_id}.pt"))
            spatial_factors = torch.zeros((1))
        ########################################################################
        else:
            Z_gt = torch.load(os.path.join(self.folder_path, f"{sample_id}_latent.pt"))
            spatial_factors = torch.load(os.path.join(
                self.folder_path, f"spatial_factors.pt"))
                        
        return X.float(), G.float(), Z_gt.float(), spatial_factors.float()
