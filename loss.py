import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

"""
This script is adapted from the official HiFi-GAN repository:
https://github.com/jik876/hifi-gan
And the Parallel WaveGAN repository:
https://github.com/kan-bayashi/ParallelWaveGAN

We define two loss components, as described in the HiFi-GAN paper:
1. Spectral Convergence Loss (L1 on the linear spectrogram)
2. Log STFT Magnitude Loss (L2/MSE on the log-magnitude spectrogram)
"""
def spectral_convergence_loss(x_mag, y_mag):
    """
    Spectral Convergence Loss component.
    Computes the L1 norm of the difference, normalized by the L1 norm of the target.
    """
    # L1 norm of the difference
    l1_diff = F.l1_loss(x_mag, y_mag, reduction='none')

    # L1 norm of the target
    l1_norm_gt = F.l1_loss(y_mag, torch.zeros_like(y_mag), reduction='none')

    # Compute the loss
    loss = torch.norm(l1_diff, p=1, dim=(1, 2)) / (torch.norm(l1_norm_gt, p=1, dim=(1, 2)) + 1e-8) # Avoid division by zero
    return loss.mean()

def log_stft_magnitude_loss(x_mag, y_mag):
    """
    Log STFT Magnitude Loss component.
    Computes the Mean Squared Error (L2 loss) between the log-magnitude spectrograms.
    """
    # Compute log-magnitude spectrograms
    log_x_mag = torch.log(x_mag + 1e-8)  # Avoid log(0)
    log_y_mag = torch.log(y_mag + 1e-8)

    # Compute MSE loss
    loss = F.mse_loss(log_x_mag, log_y_mag)
    return loss

class STFTLoss(nn.Module):
    """
    A single STFT loss module.
    This computes the STFT and then calculates the two loss components.
    """
    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # Create and register a Hann window buffer
        # This ensures the window tensor is moved to the correct device along with the model
        self.register_buffer('window', torch.hann_window(win_length), persistent=False)
    
    def forward(self, x, y):
        """
        Args:
            x (Tensor): Generated waveform (B, 1, L)
            y (Tensor): Target waveform (B, 1, L)
        """
        # Squeeze channel dimension for STFT (B, 1, L) -> (B, L)
        x = x.squeeze(1)
        y = y.squeeze(1)

        # Compute STFT
        # torch.stft returns a complex tensor of shape (B, freq_bins, time_frames)
        # We set center=True to be consistent with librosa's default behavior
        x_spec = torch.stft(x,
                            n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            win_length=self.win_length,
                            window=self.window,
                            center=True,
                            return_complex=True)
        y_spec = torch.stft(y,
                            n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            win_length=self.win_length,
                            window=self.window,
                            center=True,
                            return_complex=True)

        # Get the magnitude
        x_mag = torch.abs(x_spec)
        y_mag = torch.abs(y_spec)

        # Compute the two loss components
        sc_loss = spectral_convergence_loss(x_mag, y_mag)
        mag_loss = log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss + mag_loss

class MultiResolutionSTFTLoss(nn.Module):
    """
    The main Multi-Resolution STFT Loss module.
    This class creates multiple STFTLoss instances with different parameters.
    """
    def __init__(self, fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240]):
        """
        Initializes the Multi-Resolution STFT Loss.
        The default parameters are taken from the HiFi-GAN paper.
        """
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths), "STFT parameter lists must have the same length."

        self.stft_losses = nn.ModuleList()
        for n_fft, hop_length, win_length in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses.append(STFTLoss(n_fft, hop_length, win_length))
    
    def forward(self, x, y):
        """
        Computes the Multi-Resolution STFT Loss.
        Args:
            x (Tensor): Generated waveform (B, 1, L)
            y (Tensor): Target waveform (B, 1, L)
        Returns:
            Tensor: The averaged loss over all resolutions.
        """
        total_loss = 0.0
        for stft_loss in self.stft_losses:
            total_loss += stft_loss(x, y)
        return total_loss / len(self.stft_losses)