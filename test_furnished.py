import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys
import shutil  # <--- 新增
import librosa # <--- 新增
import librosa.display # <--- 新增
import matplotlib.pyplot as plt # <--- 新增
import soundfile as sf # <--- 新增

# --- Import our custom modules ---
from load_dataset import HybridDataset
from model import HybridUNet
from loss import MultiResolutionSTFTLoss 
try:
    from metrics_cuda import metric_cal
except ImportError:
    print("\n--- 警告 ---")
    print("未找到 'metrics_cuda.py'。将仅计算 MR-STFT 损失。")
    metric_cal = None 
except ModuleNotFoundError: # 捕获 auraloss/scipy 未安装
    print("\n--- 警告 ---")
    print("未找到 'auraloss' 或 'scipy'。将仅计算 MR-STFT 损失。")
    metric_cal = None

# --- 1. CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ▼▼▼ 确保这些设置正确 ▼▼▼
BASE_DIR = '/home/jiayangli/Downloads/raf_dataset/archived/FurnishedRoom/data'
MODEL_PATH = "./checkpoints/best_model.pth" 
TOTAL_SAMPLES = 39133 # 替换为您的 FurnishedRoom 样本总数
# ▲▲▲ 确保这些设置正确 ▲▲▲

# Parameters
BATCH_SIZE = 16       
SAMPLE_RATE = 48000 
TEST_PERCENTAGE = 0.1 
RANDOM_SEED = 42      

# --- 新增: 可视化配置 ---
NUM_SAMPLES_TO_VISUALIZE = 10 # 我们将保存前 10 个样本的图
OUTPUT_DIR = "test_outputs_furnished" # 保存所有 .wav 和 .png 的主目录
CENTRAL_PLOT_DIR = os.path.join(OUTPUT_DIR, "all_plots") # 集中存放PNG的文件夹
# --- 结束新增 ---


# --- 新增: 复制 plot_waveforms 辅助函数 ---
def plot_waveforms(target, prediction, input_high, path, sample_idx_str):
    """
    Plots the target (overlapped with prediction) and input waveforms.
    """
    fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True, 
                           gridspec_kw={'height_ratios': [2, 1]})
    
    # 子图 1: 重叠比较
    librosa.display.waveshow(target, sr=SAMPLE_RATE, ax=ax[0], color='blue', alpha=1.0, label='Target (Low Freq)')
    librosa.display.waveshow(prediction, sr=SAMPLE_RATE, ax=ax[0], color='green', alpha=0.7, label='Prediction (Low Freq)')
    ax[0].set_title(f"Sample {sample_idx_str}: Overlapped Comparison (Target vs. Prediction)")
    ax[0].legend()

    # 子图 2: 输入参考
    librosa.display.waveshow(input_high, sr=SAMPLE_RATE, ax=ax[1], color='red', label='Input (High Freq)')
    ax[1].set_title("Model Input (High Freq)")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
# --- 结束新增 ---


# --- 2. TEST FUNCTION (有修改) ---
def run_test(model, loader, criterion, test_indices_list): # <--- 修改: 接收索引列表
    model.eval() 
    
    loss_list_original = []
    t60_errors, edt_errors, c50_errors, angle_errors = [], [], [], []
    
    # <--- 新增: 创建输出目录 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CENTRAL_PLOT_DIR, exist_ok=True)
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Running Test on FurnishedRoom (10% Sample)")
        
        for i, (input_wave, input_spec, target_wave) in enumerate(progress_bar):
            input_wave = input_wave.to(DEVICE)
            input_spec = input_spec.to(DEVICE)
            target_wave = target_wave.to(DEVICE)
            
            pred_wave = model(input_wave, input_spec)
            
            # --- 1. 计算定量指标 (所有批次) ---
            loss_orig = criterion(pred_wave, target_wave)
            loss_list_original.append(loss_orig.item())
            
            if metric_cal is not None: 
                target_np_batch = target_wave.squeeze(1).cpu().numpy()
                pred_np_batch = pred_wave.squeeze(1).cpu().numpy()
                
                try:
                    (angle_error, _, _, t60_error, edt_error, 
                     C50_error, _, _, _) = metric_cal(
                                                target_np_batch, pred_np_batch, fs=SAMPLE_RATE, 
                                                device=DEVICE, input="numpy"
                                            )
                    t60_errors.append(t60_error.item())
                    edt_errors.append(edt_error.item())
                    c50_errors.append(C50_error.item())
                    angle_errors.append(angle_error)
                except Exception as e:
                    pass # 静默处理错误
            
            # --- 2. 保存定性可视化 (仅前几个样本) ---
            # i 是批次索引。我们只可视化第一个批次 (i=0) 中的前 N 个样本
            if i == 0 and (NUM_SAMPLES_TO_VISUALIZE > 0):
                print(f"\nSaving {NUM_SAMPLES_TO_VISUALIZE} visualization samples...")
                
                # 确保我们不会超出批次大小
                num_to_save = min(NUM_SAMPLES_TO_VISUALIZE, BATCH_SIZE)
                
                for j in range(num_to_save):
                    # 获取此样本的 6 位数文件夹 ID
                    sample_idx = test_indices_list[i * BATCH_SIZE + j]
                    sample_idx_str = f"{sample_idx:06d}"
                    
                    # 从批次中提取单个样本
                    pred_np_vis = pred_wave[j].squeeze().cpu().numpy()
                    target_np_vis = target_wave[j].squeeze().cpu().numpy()
                    input_np_vis = input_wave[j].squeeze().cpu().numpy()
                    
                    # 创建单独的文件夹
                    sample_output_dir = os.path.join(OUTPUT_DIR, sample_idx_str)
                    os.makedirs(sample_output_dir, exist_ok=True)
                    
                    # 保存 .wav 文件
                    sf.write(os.path.join(sample_output_dir, "predicted_low.wav"), pred_np_vis, SAMPLE_RATE)
                    sf.write(os.path.join(sample_output_dir, "target_low.wav"), target_np_vis, SAMPLE_RATE)
                    sf.write(os.path.join(sample_output_dir, "input_high.wav"), input_np_vis, SAMPLE_RATE)
                    
                    # 保存 .png (在子文件夹中)
                    plot_path = os.path.join(sample_output_dir, "comparison_plot.png")
                    plot_waveforms(target_np_vis, pred_np_vis, input_np_vis, plot_path, sample_idx_str)
                    
                    # 复制 .png (到集中文件夹)
                    central_plot_name = f"{sample_idx_str}_comparison_plot.png"
                    central_plot_path = os.path.join(CENTRAL_PLOT_DIR, central_plot_name)
                    try:
                        shutil.copyfile(plot_path, central_plot_path)
                    except IOError as e:
                        print(f"Warning: Could not copy plot. {e}")
            
            progress_bar.set_postfix(
                orig_loss=np.mean(loss_list_original),
                t60_err=np.mean(t60_errors) if t60_errors else "N/A"
            )

    # ... (计算平均值的逻辑保持不变) ...
    avg_loss_original = np.mean(loss_list_original)
    avg_t60_error = np.mean(t60_errors) if t60_errors else np.nan
    avg_edt_error = np.mean(edt_errors) if edt_errors else np.nan
    avg_c50_error = np.mean(c50_errors) if c50_errors else np.nan
    avg_angle_error = np.mean(angle_errors) if angle_errors else np.nan
    
    return avg_loss_original, avg_t60_error, avg_edt_error, avg_c50_error, avg_angle_error

# --- 3. MAIN EXECUTION (有修改) ---
def main():
    print("--- Starting Final Model Test (on 10% of FurnishedRoom) ---")
    
    # ... (文件检查保持不变) ...
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Please run train.py first.")
        sys.exit(1)
        
    print(f"Testing on {TEST_PERCENTAGE*100}% of {TOTAL_SAMPLES} samples from {BASE_DIR}")
    
    # ▼▼▼ 修改: 我们需要这个列表来命名文件夹 ▼▼▼
    print("Creating reproducible random 10% split...")
    all_indices = np.arange(TOTAL_SAMPLES)
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(all_indices)
    num_test_samples = int(TOTAL_SAMPLES * TEST_PERCENTAGE)
    
    test_indices_list = all_indices[:num_test_samples] # <--- 修改: 保存为列表
    
    test_dataset = HybridDataset(BASE_DIR, test_indices_list) # <--- 修改
    print(f"Loaded {len(test_dataset)} samples for the test set.")
    # ▲▲▲ 结束修改 ▲▲▲
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, # <-- 我们已经打乱了 test_indices_list，所以这里不需要
        num_workers=4, 
        pin_memory=True
    )
    
    # --- 2. 初始化模型和损失函数 ---
    print(f"Loading best model from {MODEL_PATH} (Trained on EmptyRoom)...")
    model = HybridUNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    criterion = MultiResolutionSTFTLoss().to(DEVICE)
    
    # --- 3. 运行测试 ---
    (avg_loss, avg_t60, 
     avg_edt, avg_c50, avg_angle) = run_test(model, test_loader, criterion, test_indices_list) # <--- 修改
    
    # --- 4. 打印所有结果 ---
    print("\n--- Test on 10% FurnishedRoom Complete ---")
    print(f"Total test samples evaluated: {len(test_dataset)}")
    
    # <--- 新增: 报告可视化结果 ---
    print(f"\n--- Qualitative Visualization ---")
    print(f"Saved {NUM_SAMPLES_TO_VISUALIZE} visualization samples (audio & plots) to:")
    print(f"  {OUTPUT_DIR}/<sample_id>/")
    print(f"  All plots collected in: {CENTRAL_PLOT_DIR}/")
    # <--- 结束新增 ---
    
    print("\n--- 1. Primary Training Metric ---")
    print(f"  Final Avg. MR-STFT Loss (from loss.py): {avg_loss:.6f}")
    
    print("\n--- 2. Acoustic & Freq Metrics (from metrics_cuda.py) ---")
    print(f"  Final Avg. Phase Error (Abs):           {avg_angle:.6f}")
    print(f"  Final Avg. T60 Error (Abs):           {avg_t60:.6f}")
    print(f"  Final Avg. EDT Error (Abs):           {avg_edt:.6f}")
    print(f"  Final Avg. C50 Error (Abs, dB):         {avg_c50:.6f}")


if __name__ == '__main__':
    # <--- 新增: 运行前的检查 ---
    try:
        plt.switch_backend('Agg') 
    except ImportError:
        print("Warning: matplotlib not found. Cannot generate plots.")
        sys.exit(1)
        
    if not 'sf' in globals():
        print("Error: The 'soundfile' library is required to save .wav files.")
        print("Please install it: pip install soundfile")
        sys.exit(1)
    
    main()
    # <--- 结束新增 ---