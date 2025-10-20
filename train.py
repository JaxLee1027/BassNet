import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import time
from torch.utils.tensorboard import SummaryWriter
import torchvision

from model import UNet
from load_dataset import SpectrogramDataset

def normalize_for_vis(tensor, vmin=-80, vmax=0):
    """Normalize a tensor for visualization purposes."""
    tensor = torch.clamp(tensor, vmin, vmax)
    tensor = (tensor - vmin) / (vmax - vmin)
    return tensor


if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="Train U-Net on spectrograms.")
    
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    
    parser.add_argument('--batch_size', '-b', type=int, default=16,
                        help='Batch size for training (default: 16)')
    
    parser.add_argument('--num_epochs', '-e', type=int, default=25,
                        help='Number of training epochs (default: 25)')
    
    args = parser.parse_args()  

    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs

    BASE_DIR = '/home/jiayangli/Downloads/raf_dataset/archived/EmptyRoom/data'
    MODEL_SAVE_PATH = "unet_spectrogram_model.pth"


    # Initialize TensorBoard writer
    log_dir_name = f"unet_lr{LEARNING_RATE}_bs{BATCH_SIZE}_ep{NUM_EPOCHS}_{int(time.time())}"
    writer = SummaryWriter(f"runs/{log_dir_name}")
    print(f"TensorBoard logs will be saved to runs/{log_dir_name}")


    print(f"Using device: {DEVICE}")
    all_indices = list(range(47484))
    train_split = int(0.8 * len(all_indices))
    val_split = int(0.9 * len(all_indices))

    train_indices = all_indices[:train_split]
    val_indices = all_indices[train_split:val_split]
    test_indices = all_indices[val_split:]

    print(f"Total samples: {len(all_indices)}")
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    print(f"Testing samples: {len(test_indices)}")

    train_dataset = SpectrogramDataset(base_dir=BASE_DIR, indices=train_indices)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_dataset = SpectrogramDataset(base_dir=BASE_DIR, indices=val_indices)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_dataset = SpectrogramDataset(base_dir=BASE_DIR, indices=test_indices)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Visualization of a fixed batch
    fixed_vis_batch = next(iter(test_loader))
    fixed_high_freq, fixed_low_freq = fixed_vis_batch
    fixed_high_freq = fixed_high_freq.to(DEVICE)

    # Save the true low-frequency spectrograms for comparison
    truth_grid = torchvision.utils.make_grid(normalize_for_vis(fixed_low_freq)[:8], nrow=4)
    writer.add_image('Validation/Ground_Truth', truth_grid, 0)

    # --- Model, Loss, Optimizer ---
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    print("Model architecture loaded from model.py: UNet")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')

    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=True)
        for high_freq, low_freq in loop:
            high_freq = high_freq.to(DEVICE)
            low_freq = low_freq.to(DEVICE)

            # Forward pass
            outputs = model(high_freq)

            # Safe check: ensure output and target dimensions match before loss calculation
            if outputs.shape != low_freq.shape:
                outputs = nn.functional.interpolate(outputs, size=low_freq.shape[2:], mode='bilinear', align_corners=False)
            
            # Calculate loss
            loss = criterion(outputs, low_freq)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for high_freq, low_freq in val_loader:
                high_freq = high_freq.to(DEVICE)
                low_freq = low_freq.to(DEVICE)

                outputs = model(high_freq)

                # Safe check: ensure output and target dimensions match before loss calculation
                if outputs.shape != low_freq.shape:
                    outputs = nn.functional.interpolate(outputs, size=low_freq.shape[2:], mode='bilinear', align_corners=False)

                loss = criterion(outputs, low_freq)
                val_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Log losses to TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch+1)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch+1)

        # Visualization on fixed batch
        model.eval()
        with torch.no_grad():
            vis_outputs = model(fixed_high_freq)
            if vis_outputs.shape != fixed_low_freq.shape:
                vis_outputs = nn.functional.interpolate(vis_outputs, size=fixed_low_freq.shape[2:], mode='bilinear', align_corners=False)
            vis_grid = torchvision.utils.make_grid(normalize_for_vis(vis_outputs.cpu())[:8], nrow=4)
            writer.add_image('Validation/Prediction', vis_grid, epoch+1)
            
        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved with val loss: {best_val_loss:.4f}")

    # Save the trained model
    print(f"Model saved to {MODEL_SAVE_PATH}")
    writer.close()
    print("TensorBoard logs saved. Run 'tensorboard --logdir=runs' to view them.")
