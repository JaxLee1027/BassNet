import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from tqdm import tqdm

from model import UNet
from load_dataset import SpectrogramDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = '/home/jiayangli/Downloads/raf_dataset/archived/EmptyRoom/data'
MODEL_PATH = "unet_spectrogram_model.pth"
BATCH_SIZE = 16
OUTPUT_DIR = "evaluation_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE = 48000
N_FFT = 2048
HOP_LENGTH = 512
SPLIT_FREQ_HZ = 16000

# Pre-calculate y-axis frequency bins for plotting
print("Calculating frequency bins...")
FREQS = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)
SPLIT_INDEX = np.argmin(np.abs(FREQS - SPLIT_FREQ_HZ))
LOW_FREQ_COORDS = FREQS[:SPLIT_INDEX] # 0-16kHz
print(f"Plotting with {len(LOW_FREQ_COORDS)} frequency bins up to ~{FREQS[SPLIT_INDEX]:.0f} Hz")


all_indices = list(range(47484))
train_split = int(0.8 * len(all_indices))
val_split = int(0.9 * len(all_indices))

train_indices = all_indices[:train_split]
val_indices = all_indices[train_split:val_split]
test_indices = all_indices[val_split:]
test_dataset = SpectrogramDataset(BASE_DIR, test_indices)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = UNet(in_channels=1, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

criterion = nn.MSELoss()
test_loss = 0.0

print("--- Starting Evaluation on Test Set ---")

with torch.no_grad():
    for high_freq, low_freq in tqdm(test_loader, desc="Calculating Test Loss"):
        high_freq = high_freq.to(DEVICE)
        low_freq = low_freq.to(DEVICE)

        outputs = model(high_freq)

        if outputs.shape != low_freq.shape:
            outputs = torch.nn.functional.interpolate(outputs, size=low_freq.shape[2:], mode='bilinear', align_corners=False)
        loss = criterion(outputs, low_freq)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f"Quantitative Evaluation:")
print(f"Average Test MSE Loss: {avg_test_loss:.6f}")

print("Generating visual comparison plots...")
with torch.no_grad():
    for i in range(5):
        high_freq, low_freq_truth = test_dataset[i]
        print(f"Loading Test Sample {i}: Ground Truth .npy shape is: {low_freq_truth.shape}")
        high_freq_input = high_freq.unsqueeze(0).to(DEVICE)
        low_freq_pred = model(high_freq_input)
        if low_freq_pred.shape != low_freq_truth.unsqueeze(0).shape:
            low_freq_pred = torch.nn.functional.interpolate(low_freq_pred, size=low_freq_truth.shape[1:], mode='bilinear', align_corners=False)
        truth_np = low_freq_truth.squeeze(0).cpu().numpy()
        pred_np = low_freq_pred.squeeze(0).squeeze(0).cpu().numpy()

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle(f'Sample {i} Spectrogram Comparison', fontsize=16)

        librosa.display.specshow(truth_np, ax=ax1, y_coords=LOW_FREQ_COORDS, y_axis='hz', x_axis='time', sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
        ax1.set_title('Ground Truth Low-Frequency Spectrogram')
        # ax1.set_ylim(0, 16000)

        librosa.display.specshow(pred_np, ax=ax2, y_coords=LOW_FREQ_COORDS, y_axis='hz', x_axis='time', sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
        ax2.set_title('Predicted Low-Frequency Spectrogram')
        # ax2.set_ylim(0, 16000)

        error_map = np.abs(truth_np - pred_np)
        img = librosa.display.specshow(error_map, ax=ax3, y_coords=LOW_FREQ_COORDS, y_axis='hz', x_axis='time', sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
        # ax3.set_ylim(0, 16000)
        fig.colorbar(img, ax=ax3, format="%+2.0f dB")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'sample_{i}_comparison.png'))
        plt.close()

print(f"Visual comparison plots saved in '{OUTPUT_DIR}' directory.")
