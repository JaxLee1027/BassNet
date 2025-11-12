import os
import torch
import numpy as np
from torch.utils.data import Dataset

class HybridDataset(Dataset):
    """
    A PyTorch Dataset class to load hybrid audio datasets.
    It loads three files per sample:
    1. low_freq_wave.npy (Target Y, 1D Waveform)
    2. high_freq_wave.npy (Input X1, 1D Waveform)
    3. high_freq_spec.npy (Input X2, 2D Spectrogram)
    """
    def __init__(self, base_dir, indices):
        """
        Args:
            base_dir (str): Base directory containing sample folders.
            indices (list): List of folder indices to load.
        """
        self.base_dir = base_dir
        self.indices = indices

        # Define the filenames we expect to find in each folder
        self.target_wave_file = "low_freq_wave.npy"
        self.input_wave_file = "high_freq_wave.npy"
        self.input_spec_file = "high_freq_spec.npy"

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.
        Args: 
            idx (int): Index of the sample to fetch.
        Returns:
            tuple: (low_freq_wave, high_freq_wave, high_freq_spec)
                    - input_wave (Tensor): Shape (1, num_samples)
                    - target_wave (Tensor): Shape (1, num_samples)
                    - input_spec (Tensor): Shape (1, num_freq_bins, time_frames)
        """
        # 1. Get the folder index and construct the path
        folder_idx = self.indices[idx]
        folder_name = f"{folder_idx:06d}"
        folder_path = os.path.join(self.base_dir, folder_name)

        # 2. Define the full paths to the three .npy files
        target_wave_path = os.path.join(folder_path, self.target_wave_file)
        input_wave_path = os.path.join(folder_path, self.input_wave_file)
        input_spec_path = os.path.join(folder_path, self.input_spec_file)

        try:
            # 3. Load the .npy files
            target_wave = np.load(target_wave_path)
            input_wave = np.load(input_wave_path)
            input_spec = np.load(input_spec_path)

            # 4. Convert to PyTorch tensors and add channel dimension
            target_wave_tensor = torch.from_numpy(target_wave).float()  
            input_wave_tensor = torch.from_numpy(input_wave).float()
            input_spec_tensor = torch.from_numpy(input_spec).float()

            # 5. Add channel dimension
            # PyTorch's Conv1D/2D layers expect data in (Batch, Channels,...) format.
            # (L,) -> (1, L)
            # (H, W) -> (1, H, W)
            target_wave_tensor = target_wave_tensor.unsqueeze(0)  # Shape: (1, 19200)
            input_wave_tensor = input_wave_tensor.unsqueeze(0)    # Shape: (1, 19200)
            input_spec_tensor = input_spec_tensor.unsqueeze(0)    # Shape: (1, 1025, 38)

            return input_wave_tensor, input_spec_tensor, target_wave_tensor
        
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing file in folder {folder_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading data from folder {folder_path}: {e}")
