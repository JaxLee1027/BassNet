import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Reusable 1D Convolutional Block ---
class Downsample1DBlock(nn.Module):
    """
    A 1D convolutional block used in the U-Net encoder.
    It consists of Conv1D -> BatchNorm1D -> LeakyReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - stride) // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.lrelu(self.bn(self.conv(x)))

class Upsample1DBlock(nn.Module):
    """
    A 1D transposed convolutional block used in the U-Net decoder.
    It consists of ConvTranspose1D -> BatchNorm1D -> ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.tconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - stride) // 2, output_padding=stride - 1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.lrelu(self.bn(self.tconv(x)))

# --- The Auxiliary 2D Spectrogram Encoder ---
class SpecEncoder2D(nn.Module):
    """
    Encodes the 2D spectrogram input into a 1D feature vector that can be fused with the 1D U-Net bottleneck.

    Input: (B, 1, 1025, 38)
    Output: (B, feature_channels, 3)
    """
    def __init__(self, out_time_dim=3, out_channels=512):
        super().__init__()
        self.out_time_dim = out_time_dim

        # A simple 2D CNN stack
        # Input: (B, 1, 1025, 38)
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(2, 2), padding=1),  # (B, 32, 513, 19)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=1),  # (B, 64, 257, 10)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 2), padding=1),  # (B, 128, 129, 5)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 2), padding=1),  # (B, 256, 65, 3)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        # At this point, we have (B, 256, 65, 3)
        # We need to get it to (B, out_channels, out_time_dim)

        # 1. Flatten the frequency dimension (H)
        # (B, 256, 65, 3) -> (B, 256 * 65, 3) = (B, 16640, 3)
        self.flattened_channels = 256 * 65

        # 2. Project it down to the desired channel size
        # We use a 1*1 Conv (acting on time) which is equivalent to a linear layer applied to each time frame
        self.projection = nn.Conv1d(self.flattened_channels, out_channels, kernel_size=1)

    def forward(self, x_spec):
        # x_spec: (B, 1, 1025, 38)
        x = self.conv_stack(x_spec)
        # x shape: (B, 256, 65, 3)
        
        # Flatten H and C dimensions together
        # (B, 256, 65, 3) -> (B, 256*65, 3)
        batch_size = x.size(0)
        x_flat = x.view(batch_size, self.flattened_channels, self.out_time_dim)

        # Project to desired out_channels
        x_projected = self.projection(x_flat)

        return x_projected  # Shape: (B, out_channels, out_time_dim)

class HybridWaveUNet(nn.Module):
    