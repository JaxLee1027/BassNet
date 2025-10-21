import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sys

# --- 1. 配置 ---
# 包含 '000036' 子文件夹的根目录
BASE_DIR = '/home/jiayangli/Downloads/raf_dataset/archived/EmptyRoom/data'

# ▼▼▼ 目标文件夹 ▼▼▼
TARGET_FOLDER_NAME = '042735'

# 输入文件名
LOW_FREQ_FILENAME = 'low_freq_data.npy'
HIGH_FREQ_FILENAME = 'high_freq_data.npy'

# 新的输出文件名
LOW_FREQ_OUTPUT_IMG = 'low_freq_spectrogram.png'
HIGH_FREQ_OUTPUT_IMG = 'high_freq_spectrogram.png'

# --- 2. 频谱图参数 (必须与 convert_spectrogram.py 保持一致) ---
SAMPLE_RATE = 48000
N_FFT = 2048
HOP_LENGTH = 512
SPLIT_FREQ_HZ = 16000

# --- 3. 预先计算频率坐标轴 (关键！) ---
print("Calculating frequency bins...")
FREQS = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)
# 找到分割点
SPLIT_INDEX = np.argmin(np.abs(FREQS - SPLIT_FREQ_HZ))

# 为两个图分别创建坐标
LOW_FREQ_COORDS = FREQS[:SPLIT_INDEX]
HIGH_FREQ_COORDS = FREQS[SPLIT_INDEX:]
print(f"Data split at index {SPLIT_INDEX} (~{FREQS[SPLIT_INDEX]:.0f} Hz)")

def process_folder(folder_path):
    """
    在单个文件夹中加载 .npy 文件, 并为每个文件保存一张图像。
    """
    # 定义所有文件的路径
    low_freq_path = os.path.join(folder_path, LOW_FREQ_FILENAME)
    high_freq_path = os.path.join(folder_path, HIGH_FREQ_FILENAME)
    low_output_path = os.path.join(folder_path, LOW_FREQ_OUTPUT_IMG)
    high_output_path = os.path.join(folder_path, HIGH_FREQ_OUTPUT_IMG)

    try:
        # --- 检查文件 ---
        if not (os.path.exists(low_freq_path) and os.path.exists(high_freq_path)):
            print(f"Error: Input files not found in {folder_path}")
            return False

        # --- 绘制并保存 低频图 ---
        print(f"Processing {LOW_FREQ_FILENAME}...")
        low_data = np.load(low_freq_path)
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(low_data, 
                                 y_coords=LOW_FREQ_COORDS,
                                 y_axis='hz', 
                                 x_axis='time',
                                 sr=SAMPLE_RATE, 
                                 hop_length=HOP_LENGTH)
        
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Low Freq (0 - {SPLIT_FREQ_HZ/1000:.0f}kHz): {os.path.basename(folder_path)}')
        plt.tight_layout()
        plt.savefig(low_output_path)
        plt.close()
        print(f"Saved: {low_output_path}")

        # --- 绘制并保存 高频图 ---
        print(f"Processing {HIGH_FREQ_FILENAME}...")
        high_data = np.load(high_freq_path)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(high_data, 
                                 y_coords=HIGH_FREQ_COORDS,
                                 y_axis='hz', 
                                 x_axis='time',
                                 sr=SAMPLE_RATE, 
                                 hop_length=HOP_LENGTH)
        
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'High Freq ({SPLIT_FREQ_HZ/1000:.0f}kHz+): {os.path.basename(folder_path)}')
        plt.tight_layout()
        plt.savefig(high_output_path)
        plt.close()
        print(f"Saved: {high_output_path}")
        
        return True

    except Exception as e:
        print(f"Error processing {folder_path}: {e}")
        return False

def main():
    print(f"--- Starting Single Folder Visualization Test ---")
    print(f"Target Folder: {TARGET_FOLDER_NAME}")
    
    # 构建目标文件夹的完整路径
    folder_path = os.path.join(BASE_DIR, TARGET_FOLDER_NAME)
    
    # 检查目标文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"ERROR: Target folder not found at: '{folder_path}'")
        sys.exit(1) # 退出脚本
        
    # 处理这一个文件夹
    success = process_folder(folder_path)
    
    if success:
        print("\n--- Visualization complete! ---")
    else:
        print("\n--- Visualization failed. ---")

if __name__ == '__main__':
    main()