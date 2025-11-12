import os
import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from tqdm import tqdm

# --- Import custom modules ---
from load_dataset import HybridDataset
from model import HybridUNet

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Paths
BASE_DIR = '/home/jiayangli/Downloads/raf_dataset/archived/EmptyRoom/data'
OUTPUT_DIR = './evaluation_outputs'
MODEL_PATH = './checkpoints/best_model.pth'
SPLIT_FILE = 'dataset_split.npy'

# Parameters
SAMPLE_RATE = 48000
TOTAL_SAMPELS = 47484  # Total number of folders/samples
VAL_SPLIT = int(0.9 * TOTAL_SAMPELS)

NUM_SAMPLES_TO_EVALUATE = 10 

def plot_waveforms(target, prediction, input_high, path, sample_idx):
    """
    Plots and saves waveforms of target, prediction, and input high-frequency signals.
    """
    fig, ax = plt.subplots(3, 1, figsize=(15, 10), sharex=True, sharey=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot Target (Ground Truth)
    librosa.display.waveshow(target, sr=SAMPLE_RATE, ax=ax[0], color='blue', label='Target (Low Freq)')
    ax[0].set_title(f"Sample {sample_idx}: Ground Truth (Target Low Freq)")
    ax[0].legend()
    
    # Plot Prediction
    librosa.display.waveshow(prediction, sr=SAMPLE_RATE, ax=ax[1], color='green', label='Prediction (Low Freq)')
    ax[1].set_title("Model Prediction")
    ax[1].legend()

    # Plot Input (High Freq)
    librosa.display.waveshow(input_high, sr=SAMPLE_RATE, ax=ax[2], color='red', label='Input (High Freq)')
    ax[2].set_title("Model Input (High Freq)")
    ax[2].legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)

def main():
    print("--- Starting Evaluation ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -- 1. Load Model --
    print(f"Loading model from {MODEL_PATH}...")
    model = HybridUNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")

    # -- 2. Load Dataset --
    if not os.path.exists(SPLIT_FILE):
        print(f"Split file {SPLIT_FILE} not found. Creating new split...")
        return
    
    print(f"Loading split indices from {SPLIT_FILE}...")
    all_indices = np.load(SPLIT_FILE)
    test_indices = all_indices[VAL_SPLIT:]
    
    np.random.shuffle(test_indices)
    selected_indices = test_indices[:NUM_SAMPLES_TO_EVALUATE]
    print(f"Selected {NUM_SAMPLES_TO_EVALUATE} samples for evaluation.")

    # -- 3. Evaluate Model --
    eval_dataset = HybridDataset(BASE_DIR, selected_indices)

    for i in tqdm(range(len(eval_dataset)), desc="Evaluating Model"):
        input_wave, input_spec, target_wave = eval_dataset[i]

        sample_idx = selected_indices[i]
        
        # --- 4. Model Inference ---
        # Add batch dimension (B=1) and move to device
        input_wave = input_wave.unsqueeze(0).to(DEVICE)
        input_spec = input_spec.unsqueeze(0).to(DEVICE)

        # Forward pass
        with torch.no_grad():
            pred_wave = model(input_wave, input_spec)

        # --- 5. Post-processing ---
        pred_np = pred_wave.squeeze().cpu().numpy()
        target_np = target_wave.squeeze().cpu().numpy()
        input_high_np = input_wave.squeeze().cpu().numpy()

        # --- 6. Save Results ---
        # Create a dedicated folder for this sample
        sample_output_dir = os.path.join(OUTPUT_DIR, f"sample_{sample_idx:06d}")
        os.makedirs(sample_output_dir, exist_ok=True)

        # Save audio files
        sf.write(os.path.join(sample_output_dir, "target_low.wav"), target_np, SAMPLE_RATE)
        sf.write(os.path.join(sample_output_dir, "prediction_low.wav"), pred_np, SAMPLE_RATE)
        sf.write(os.path.join(sample_output_dir, "input_high.wav"), input_high_np, SAMPLE_RATE)

        # Save the comparison plot
        plot_path = os.path.join(sample_output_dir, "comparison_plot.png")
        plot_waveforms(target_np, pred_np, input_high_np, plot_path, sample_idx)

    print("Evaluation complete.")
    print(f"Find all evaluation results in: {OUTPUT_DIR}")

if __name__ == "__main__":
    plt.switch_backend('Agg')
    main()
        