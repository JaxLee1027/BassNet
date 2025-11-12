import torch
import torch.nn.functional as F
import numpy as np
# from torchaudio.transforms import MultiResolutionSTFTLoss
from auraloss.freq import MultiResolutionSTFTLoss
from scipy.signal import hilbert


def metric_cal(ori_ir_np, pred_ir_np, fs=48000, window=32, device="cuda", input="numpy", eps=1e-13):

    if input == "numpy":
        if ori_ir_np.ndim == 1:
            ori_ir_np = ori_ir_np[np.newaxis, :]
        if pred_ir_np.ndim == 1:
            pred_ir_np = pred_ir_np[np.newaxis, :]
    
        # Convert to torch tensors on GPU (no gradients)
        ori_ir = torch.from_numpy(ori_ir_np).float().to(device)
        pred_ir = torch.from_numpy(pred_ir_np).float().to(device)

    elif input == "torch":
        if ori_ir_np.ndim == 1:
            ori_ir_np = ori_ir_np.unsqueeze(0)
        if pred_ir_np.ndim == 1:
            pred_ir_np = pred_ir_np.unsqueeze(0)
        ori_ir = ori_ir_np.float().to(device)
        pred_ir = pred_ir_np.float().to(device)
        ori_ir_np = ori_ir.cpu().numpy()
        pred_ir_np = pred_ir.cpu().numpy()

    else:
        raise ValueError(f"Invalid input type: {input}")

    with torch.no_grad():
        # ---- Multi-Resolution STFT Loss ----
        multi_stft = MultiResolutionSTFTLoss(w_lin_mag=1,
                                             fft_sizes=[512, 256, 128],
                                             win_lengths=[300, 150, 75],
                                             hop_sizes=[60, 30, 8]).to(device)
        multi_stft_loss = multi_stft(ori_ir.unsqueeze(1), pred_ir.unsqueeze(1))

        # ---- FFT-based metrics ----
        fft_ori = torch.fft.rfft(ori_ir, dim=-1)
        fft_pred = torch.fft.rfft(pred_ir, dim=-1)

        angle_ori = torch.angle(fft_ori)
        angle_pred = torch.angle(fft_pred)
        angle_error = torch.mean(torch.abs(torch.cos(angle_ori) - torch.cos(angle_pred)) +
                                 torch.abs(torch.sin(angle_ori) - torch.sin(angle_pred))).item()

        # Smoothed amplitude error
        amp_ori = torch.abs(fft_ori)
        amp_pred = torch.abs(fft_pred)
        kernel = torch.ones(1, 1, window, device=device) / window
        amp_ori_smoothed = F.conv1d(amp_ori.unsqueeze(1), kernel, padding=window // 2).squeeze(1)
        amp_pred_smoothed = F.conv1d(amp_pred.unsqueeze(1), kernel, padding=window // 2).squeeze(1)
        amp_error = torch.mean(torch.abs(amp_ori_smoothed - amp_pred_smoothed) / (amp_ori_smoothed + 1e-6)).item()

        # ---- Envelope error (Hilbert still uses CPU) ----
        ori_env = np.abs(hilbert(ori_ir_np, axis=-1))
        pred_env = np.abs(hilbert(pred_ir_np, axis=-1))
        env_error = np.mean(np.abs(ori_env - pred_env) / (np.max(ori_env, axis=1, keepdims=True) + 1e-6))

        # ---- Energy decay (reverse cumulative sum) ----
        def compute_energy_trend(x):
            x_rev = torch.flip(x ** 2, dims=[-1])
            cumsum = torch.cumsum(x_rev + eps, dim=-1)
            log_energy = 10.0 * torch.log10(torch.flip(cumsum, dims=[-1]))
            log_energy = log_energy - log_energy[:, :1].clone()
            return log_energy
        
        ori_energy = compute_energy_trend(ori_ir)
        pred_energy = compute_energy_trend(pred_ir)

        # ---- T60 & EDT (compute on CPU for now) ----
        # ori_energy_np = ori_energy.cpu().numpy()
        # pred_energy_np = pred_energy.cpu().numpy()
        ori_t60, ori_edt = t60_EDT_cal_torch(ori_energy, fs)
        pred_t60, pred_edt = t60_EDT_cal_torch(pred_energy, fs)
        t60_error = torch.mean(torch.abs(ori_t60 - pred_t60) / (ori_t60 + 1e-6))
        edt_error = torch.mean(torch.abs(ori_edt - pred_edt))

        # ---- C50 error ----
        base_sample = 0
        samples_50ms = int(0.05 * fs) + base_sample
        early_ori = torch.sum(ori_ir[:, base_sample:samples_50ms] ** 2, dim=-1)
        late_ori = torch.sum(ori_ir[:, samples_50ms:] ** 2, dim=-1)
        early_pred = torch.sum(pred_ir[:, base_sample:samples_50ms] ** 2, dim=-1)
        late_pred = torch.sum(pred_ir[:, samples_50ms:] ** 2, dim=-1)
        C50_ori = 10.0 * torch.log10((early_ori + eps) / (late_ori + eps))
        C50_pred = 10.0 * torch.log10((early_pred + eps) / (late_pred + eps))
        C50_error = torch.mean(torch.abs(C50_ori - C50_pred))


    return (angle_error, amp_error, env_error, t60_error.cpu(), edt_error.cpu(),
            C50_error.cpu(), multi_stft_loss.cpu(), ori_energy.cpu().numpy(), pred_energy.cpu().numpy())

def t60_EDT_cal_torch(energy_db: torch.Tensor, fs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-friendly version of T60 and EDT calculation in PyTorch.
    
    Args:
        energy_db (torch.Tensor): [B, T] log-energy decay curve (in dB), already normalized.
        fs (int): sampling rate.
        
    Returns:
        t60 (torch.Tensor): [B] RT60 per sample (in seconds).
        edt (torch.Tensor): [B] EDT per sample (in seconds).
    """
    B, T = energy_db.shape
    device = energy_db.device
    time = torch.arange(T, device=device).float() / fs  # [T]
    time = time.expand(B, -1)  # [B, T]

    def fit_decay(decay_curve, target_db_start, target_db_end):
        """
        Fit decay between target_db_start and target_db_end, return slope.
        """
        # Create a mask for each sample where the dB is within the range
        mask = (decay_curve <= target_db_start) & (decay_curve >= target_db_end)  # [B, T]
        eps = 1e-6

        # Count valid points per sample
        valid_counts = mask.sum(dim=1).clamp(min=1)  # [B]
        mask_f = mask.float()

        # Linear regression: fit line to decay vs. time using least squares
        x = time * mask_f  # [B, T]
        y = decay_curve * mask_f  # [B, T]

        mean_x = x.sum(dim=1) / valid_counts  # [B]
        mean_y = y.sum(dim=1) / valid_counts  # [B]

        x_centered = x - mean_x.unsqueeze(1)
        y_centered = y - mean_y.unsqueeze(1)

        slope = (x_centered * y_centered * mask_f).sum(dim=1) / (
            (x_centered**2 * mask_f).sum(dim=1) + eps
        )  # [B]

        return slope

    with torch.no_grad():
        # T60: fit between 0 and -60 dB
        slope_t60 = fit_decay(energy_db, target_db_start=-5, target_db_end=-25.0)
        t60 = -60.0 / (slope_t60 + 1e-6)

        # EDT: fit between 0 and -10 dB
        slope_edt = fit_decay(energy_db, target_db_start=0.0, target_db_end=-10.0)
        edt = -60.0 / (slope_edt + 1e-6)

    return t60, edt