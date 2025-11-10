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
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.lrelu(self.bn(self.conv(x)))

class Upsample1DBlock(nn.Module):
    """
    A 1D transposed convolutional block used in the U-Net decoder.
    It consists of ConvTranspose1D -> BatchNorm1D -> LeakyReLU.
    
    NOW INCLUDES 'output_padding' TO FIX ASYMMETRY.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0): 
        super().__init__()
        self.tconv = nn.ConvTranspose1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding=(kernel_size - 1) // 2,
            output_padding=output_padding 
        )
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

class HybridUNet(nn.Module):
    """
    The main model. Combines a 1D U-Net with a 2D spectrogram encoder.
    Input X1: x_wave (B, 1, 19200) - High-frequency waveform
    Input X2: x_spec (B, 1, 1025, 38) - High-frequency spectrogram
    Output Y: y_wave (B, 1, 19200) - Reconstructed low-frequency waveform
    """

    def __init__(self, spec_out_channels=512, spec_out_time_dim=3):
        super().__init__()

        # --- 1D U-Net Encoder ---
        # L0: Input (B, 1, 19200)
        self.enc1 = Downsample1DBlock(1, 32, kernel_size=15, stride=4)   # (B, 32, 4800)
        self.enc2 = Downsample1DBlock(32, 64, kernel_size=15, stride=4)  # (B, 64, 1200)
        self.enc3 = Downsample1DBlock(64, 128, kernel_size=15, stride=4) # (B, 128, 300)
        self.enc4 = Downsample1DBlock(128, 256, kernel_size=15, stride=4) # (B, 256, 75)
        self.enc5 = Downsample1DBlock(256, 512, kernel_size=15, stride=5) # (B, 512, 15)
        self.enc6 = Downsample1DBlock(512, 1024, kernel_size=15, stride=5) # (B, 1024, 3) - Bottleneck

        # --- 2D Spectrogram Encoder ---
        self.spec_encoder = SpecEncoder2D(out_channels=spec_out_channels, out_time_dim=spec_out_time_dim)

        # --- 1D U-Net Decoder ---
        # The first decoder block's in_channels must accept the fused features
        fused_channels = 1024 + spec_out_channels
        self.dec1 = Upsample1DBlock(fused_channels, 512, kernel_size=15, stride=5, output_padding=4)  # (B, 512, 15)
        self.dec2 = Upsample1DBlock(512 + 512, 256, kernel_size=15, stride=5, output_padding=4)  # (B, 256, 75)
        self.dec3 = Upsample1DBlock(256 + 256, 128, kernel_size=15, stride=4, output_padding=3) # (B, 128, 300)
        self.dec4 = Upsample1DBlock(128 + 128, 64, kernel_size=15, stride=4, output_padding=3)  # (B, 64, 1200)
        self.dec5 = Upsample1DBlock(64 + 64, 32, kernel_size=15, stride=4, output_padding=3)    # (B, 32, 4800)
        self.dec6 = Upsample1DBlock(32 + 32, 16, kernel_size=15, stride=4, output_padding=3)    # (B, 16, 19200)

        # Final output layer to project back to 1 channel
        self.out_conv = nn.Conv1d(16, 1, kernel_size=7, padding=3)

    def forward(self, x_wave, x_spec):
        """
        The main forward pass.
        Args:
            x_wave (Tensor): High-frequency waveform input (B, 1, 19200)
            x_spec (Tensor): High-frequency spectrogram input (B, 1, 1025, 38)
        """
        # --- 1. 1D Encoder Pass ---
        # We save all intermediate outputs for skip connections
        s1 = self.enc1(x_wave)
        # print("s1.shape:", s1.shape)
        s2 = self.enc2(s1)
        # print("s2.shape:", s2.shape)
        s3 = self.enc3(s2)
        # print("s3.shape:", s3.shape)
        s4 = self.enc4(s3)
        # print("s4.shape:", s4.shape)
        s5 = self.enc5(s4)
        # print("s5.shape:", s5.shape)
        wave_bottleneck = self.enc6(s5)  # (B, 1024, 3)
        # print("wave_bottleneck.shape:", wave_bottleneck.shape)
        # --- 2. 2D Encoder Pass ---
        spec_bottleneck = self.spec_encoder(x_spec)  # (B, spec_out_channels, 3)

        # --- 3. Fuse Bottlenecks ---
        fused_bottleneck = torch.cat([wave_bottleneck, spec_bottleneck], dim=1)
        # print("fused_bottleneck.shape:", fused_bottleneck.shape)
        # fused_bottleneck: (B, 1024 + spec_out_channels, 3)

        # --- 4. 1D Decoder Pass with Skip Connections ---
        # We concatenate the output of the decoder block with the skip connection from the corresponsing encoder block
        d1 = self.dec1(fused_bottleneck)
        # print("d1.shape:", d1.shape)
        d1_skip = torch.cat([d1, s5], dim=1) # (B, 512 + 512, 15)

        d2 = self.dec2(d1_skip)
        d2_skip = torch.cat([d2, s4], dim=1) # (B, 256 + 256, 75)

        d3 = self.dec3(d2_skip)
        d3_skip = torch.cat([d3, s3], dim=1) # (B, 128 + 128, 300)

        d4 = self.dec4(d3_skip)
        d4_skip = torch.cat([d4, s2], dim=1) # (B, 64 + 64, 1200)

        d5 = self.dec5(d4_skip)
        d5_skip = torch.cat([d5, s1], dim=1) # (B, 32 + 32, 4800)

        d6 = self.dec6(d5_skip)  # (B, 16, 19200)

        # --- 5. Final Output Layer ---
        out_wave = self.out_conv(d6)  # (B, 1, 19200)

        # Apply tanh activation to keep output in [-1, 1]
        out_wave = torch.tanh(out_wave)
        return out_wave