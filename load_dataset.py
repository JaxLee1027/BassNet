import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class SpectrogramDataset(Dataset):
    def __init__(self, base_dir, indices):
        """
        Args:
            base_dir (string): paths of dataset
            indices (list of int): List of folder numbers this dataset should contain
        """
        self.base_dir = base_dir
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the real folder number
        folder_idx = self.indices[idx]
        folder_name = f"{folder_idx:06d}"
        
        # Constructing file paths
        high_freq_path = os.path.join(self.base_dir, folder_name, 'high_freq_data.npy')
        low_freq_path = os.path.join(self.base_dir, folder_name, 'low_freq_data.npy')
        
        # Load .npy files
        high_freq_data = np.load(high_freq_path)
        low_freq_data = np.load(low_freq_path)
        # Convert to PyTorch Tensors
        # ResNet requires a "channel" dimension, so we need to add a dimension (C, H, W) to the front
        # We have a single channel here, so C=1       

        high_freq_tensor = torch.from_numpy(high_freq_data).float().unsqueeze(0)
        low_freq_tensor = torch.from_numpy(low_freq_data).float().unsqueeze(0)
        
        return high_freq_tensor, low_freq_tensor