import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np

# --- Import custom modules ---
from load_dataset import HybridDataset
from model import HybridUNet
from loss import MultiResolutionSTFTLoss

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Paths
BASE_DIR = '/home/jiayangli/Downloads/raf_dataset/archived/EmptyRoom/data'
SAVE_DIR = './checkpoints'
LOG_DIR = 'runs/hybrid_model_v1'

# Training
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 2e-4
TOTAL_SAMPELS = 47484  # Total number of folders/samples

# Dataset splits
TRAIN_SPLIT = int(0.8 * TOTAL_SAMPELS)
VAL_SPLIT = int(0.9 * TOTAL_SAMPELS)

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Training loop ---
def train_loop(model, loader, criterion, opitmizer, epoch, writer):
    """
    Runs one epoch of training.
    """
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(loader, desc=f"Epoch {1+epoch}/{NUM_EPOCHS} [Train]")

    for i, (input_wave, input_spec, target_wave) in enumerate(progress_bar):
        input_wave = input_wave.to(DEVICE)
        input_spec = input_spec.to(DEVICE)
        target_wave = target_wave.to(DEVICE)

        # Zero gradients
        opitmizer.zero_grad()

        # Forward pass
        # Our model takes two inputs and produces one output
        output_wave = model(input_wave, input_spec)

        # Compute loss
        # MR-STFT loss compares output_wave and target_wave
        loss = criterion(output_wave, target_wave)

        # Backward pass and optimization step
        loss.backward()
        opitmizer.step()
         
        # Accumulate loss
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item(), avg_loss=total_loss/(i+1))

    avg_epoch_loss = total_loss / len(loader)
    writer.add_scalar('Loss/Train', avg_epoch_loss, epoch)
    return avg_epoch_loss

# --- Validation loop ---
def validate_loop(model, loader, criterion, epoch, writer):
    """
    Runs one epoch of validation.
    """
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        progress_bar = tqdm(loader, desc=f"Epoch {1+epoch}/{NUM_EPOCHS} [Val]")
        
        for i, (input_wave, input_spec, target_wave) in enumerate(progress_bar):
            input_wave = input_wave.to(DEVICE)
            input_spec = input_spec.to(DEVICE)
            target_wave = target_wave.to(DEVICE)

            # Forward pass
            pred_wave = model(input_wave, input_spec)

            # Compute loss
            loss = criterion(pred_wave, target_wave)

            # Accumulate loss
            total_val_loss += loss.item()
            progress_bar.set_postfix(val_loss=loss.item(), avg_val_loss=total_val_loss/(i+1))

    avg_loss = total_val_loss / len(loader)
    writer.add_scalar('Loss/Val', avg_loss, epoch)
    return avg_loss

# --- Main execution ---
def main():
    # Setup Tensorboard
    writer = SummaryWriter(LOG_DIR)

    # Create dataset and dataloaders
    print("Loading datasets...")
    SPLIT_FILE = "dataset_split.npy"
    RANDOM_SEED = 42
    if not os.path.exists(SPLIT_FILE):
        print(f"Split file {SPLIT_FILE} not found. Creating new split...")
        print(f"Using random seed: {RANDOM_SEED}")

        # Create the full index list
        all_indices = np.arange(TOTAL_SAMPELS)

        # Shuffle indices
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(all_indices)

        # Save the split indices
        np.save(SPLIT_FILE, all_indices)
        print(f"Saved split indices to {SPLIT_FILE}")
    else:
        print(f"Loading split indices from {SPLIT_FILE}...")
        # Load the previously shuffled indices
        all_indices = np.load(SPLIT_FILE)

    train_indices = all_indices[:TRAIN_SPLIT]
    val_indices = all_indices[TRAIN_SPLIT:VAL_SPLIT]

    train_dataset = HybridDataset(BASE_DIR, train_indices)
    val_dataset = HybridDataset(BASE_DIR, val_indices)

    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model, loss function, optimizer, and scheduler
    print("Initializing model...")

    model = HybridUNet().to(DEVICE)
    criterion = MultiResolutionSTFTLoss().to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.8, 0.99))

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    best_val_loss = float('inf')

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        # Run one epoch of training
        train_loss = train_loop(model, train_loader, criterion, optimizer, epoch, writer)
        # Run one epoch of validation
        val_loss = validate_loop(model, val_loader, criterion, epoch, writer)
        # Log learning rate
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        # Step the scheduler
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # --- Save Checkpoints ---
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'latest_model.pth'))

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
            print(f"Saved new best model with val loss: {best_val_loss:.6f}")

    writer.close()
    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Find best model in: {os.path.join(SAVE_DIR, 'best_model.pth')}")
    print(f"Find Tensorboard logs in: {LOG_DIR}")

if __name__ == '__main__':
    main()
