import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys

# --- Import our custom modules ---
from load_dataset import HybridDataset
from model import HybridUNet
from loss import MultiResolutionSTFTLoss
from metrics_cuda import metric_cal

# --- 1. CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Paths
BASE_DIR = '/home/jiayangli/Downloads/raf_dataset/archived/EmptyRoom/data'
MODEL_PATH = "./checkpoints/best_model.pth" # Path to your best saved model
SPLIT_FILE = "dataset_split.npy"

# Parameters
BATCH_SIZE = 16       # Use the same batch size as validation
TOTAL_SAMPLES = 47484 
VAL_SPLIT = int(0.9 * TOTAL_SAMPLES) # Test set starts at the 90% mark
SAMPLE_RATE = 48000

# --- 2. TEST FUNCTION ---
def run_test(model, loader, criterion):
    """
    Runs a full pass over the test dataset to calculate the average loss.
    """
    model.eval() # Set model to evaluation mode
    total_test_loss = 0.0
    t60_errors = []
    edt_errors = []
    c50_errors = []
    angle_errors = []
    # We don't need gradients for testing
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Running Test")
        
        for i, (input_wave, input_spec, target_wave) in enumerate(progress_bar):
            # Move all data to the selected device
            input_wave = input_wave.to(DEVICE)
            input_spec = input_spec.to(DEVICE)
            target_wave = target_wave.to(DEVICE) # <-- Target also moved to device
            
            # 1. Forward pass
            # The model expects (你是说高低频谱evaluate么B, C, L) and (B, C, H, W)
            pred_wave = model(input_wave, input_spec) # (B, 1, 19200)
            
            # 2. Calculate loss
            # The criterion also expects batches: (B, C, L)
            loss = criterion(pred_wave, target_wave)
            target_np = target_wave.squeeze(1).cpu().numpy()
            pred_np = pred_wave.squeeze(1).cpu().numpy()

            # 3. Calculate metrics
            angle_error, amp_error, env_error, t60_error, edt_error, c50_error, multi_stft_loss, ori_energy, pred_energy = metric_cal(target_np, pred_np, fs=SAMPLE_RATE, device=DEVICE, input="numpy")
            t60_errors.append(t60_error.item())
            edt_errors.append(edt_error.item())
            c50_errors.append(c50_error.item())
            angle_errors.append(angle_error)
            # 3. Logging
            total_test_loss += loss.item()
            progress_bar.set_postfix(avg_loss=total_test_loss / (i + 1))
    # Calculate the average metrics over the entire test set        
    avg_t60_error = np.mean(t60_errors)
    avg_edt_error = np.mean(edt_errors)
    avg_c50_error = np.mean(c50_errors)
    avg_angle_error = np.mean(angle_errors)
    # Return the final average loss over the entire test set
    avg_loss = total_test_loss / len(loader)
    return avg_loss, avg_t60_error, avg_edt_error, avg_c50_error, avg_angle_error

# --- 3. MAIN EXECUTION ---
def main():
    print("--- Starting Final Model Test ---")
    
    # --- 1. Load Test Set Indices ---
    if not os.path.exists(SPLIT_FILE):
        print(f"Error: Split file '{SPLIT_FILE}' not found. Please run train.py first.")
        sys.exit(1)
        
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Please run train.py first.")
        sys.exit(1)
        
    print("Loading dataset split...")
    all_indices = np.load(SPLIT_FILE)
    test_indices = all_indices[VAL_SPLIT:] # The final 10%
    
    test_dataset = HybridDataset(BASE_DIR, test_indices)
    
    print(f"Loaded {len(test_dataset)} samples for the test set.")
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, # We never shuffle the test set
        num_workers=4, 
        pin_memory=True
    )
    
    # --- 2. Initialize Model and Criterion ---
    print(f"Loading best model from {MODEL_PATH}...")
    
    model = HybridUNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    # We must use the *exact same* loss function we trained with
    criterion = MultiResolutionSTFTLoss().to(DEVICE)
    
    # --- 3. Run the Test ---
    (avg_test_loss, avg_t60_error, avg_edt_error, avg_c50_error, avg_angle_error) = run_test(model, test_loader, criterion)
    
    print("\n--- Test Complete ---")
    print(f"Total test samples evaluated: {len(test_dataset)}")
    print(f"Final Average Multi-Resolution STFT Loss on Test Set: {avg_test_loss:.6f}")
    print(f"Final Average T60 Error on Test Set: {avg_t60_error*100:.1f}")
    print(f"Final Average EDT Error on Test Set: {avg_edt_error*1000:.1f}")
    print(f"Final Average C50 Error on Test Set: {avg_c50_error:.1f}")
    # print(f"Final Average Angle Error on Test Set: {avg_angle_error:.1f}")

if __name__ == '__main__':
    main()