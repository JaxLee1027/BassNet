import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# --- 1. CONFIGURATION ---
BASE_DIR = '/home/jiayangli/Downloads/raf_dataset/archived/EmptyRoom/data'
TARGET_FOLDER_NAME = '031585' # Using the folder from your example image
AUDIO_FILENAME = 'rir.wav'

# --- 2. SPECTROGRAM PARAMETERS ---
SAMPLE_RATE = 48000
N_FFT = 2048
HOP_LENGTH = 512
FIG_SIZE = (10, 4)
CMAP = 'magma' # 'magma' is also a great choice

# --- 3. SPLIT PARAMETERS ---
SPLIT_FREQ_HZ = 16000 # The frequency to split at (16 kHz)

def visualize_split():
    """
    Loads an audio file, calculates its spectrogram, splits it into
    low and high frequency parts, and saves visualizations of all three.
    """
    print("--- Starting Spectrogram Splitting Visualization ---")
    
    audio_path = os.path.join(BASE_DIR, TARGET_FOLDER_NAME, AUDIO_FILENAME)
    
    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found at '{audio_path}'.")
        return

    try:
        # --- A. Load and Calculate Full Spectrogram ---
        print("[*] Loading audio and calculating full spectrogram...")
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
        D = librosa.amplitude_to_db(np.abs(S), ref=np.max)

        # --- B. Calculate the Split Index ---
        # Get the frequencies for each row in the spectrogram matrix
        freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
        # Find the index of the frequency bin closest to our split frequency
        split_index = np.argmin(np.abs(freqs - SPLIT_FREQ_HZ))
        print(f"[*] Split frequency: {SPLIT_FREQ_HZ} Hz corresponds to array index: {split_index}")

        # --- C. Split the Data Matrix ---
        print("[*] Splitting the spectrogram matrix into high and low parts...")
        low_freq_data = D[:split_index, :]
        high_freq_data = D[split_index:, :]
        
        # --- D. Visualize and Save ---

        # 1. Original spectrogram with a split line
        output_path_line = os.path.join(BASE_DIR, TARGET_FOLDER_NAME, 'spectrogram_with_split_line.png')
        print(f"[*] Saving original spectrogram with split line to {output_path_line}")
        plt.figure(figsize=FIG_SIZE)
        librosa.display.specshow(D, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='linear')
        plt.colorbar(format='%+2.0f dB')
        # Draw a horizontal red dashed line at 16kHz
        plt.axhline(y=SPLIT_FREQ_HZ, color='r', linestyle='--') 
        plt.title(f'Full Spectrogram with {SPLIT_FREQ_HZ/1000} kHz Split Line')
        plt.tight_layout()
        plt.savefig(output_path_line)
        plt.close()

        # 2. Low-frequency part (your target Y)
        output_path_low = os.path.join(BASE_DIR, TARGET_FOLDER_NAME, 'spectrogram_low_freq_part.png')
        print(f"[*] Saving low-frequency part to {output_path_low}")
        plt.figure(figsize=FIG_SIZE)
        librosa.display.specshow(low_freq_data, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='hz', y_coords = freqs[:split_index])
        plt.colorbar(format='%+2.0f dB')
        plt.title('Low-Frequency Part (0 - 16 kHz)')
        plt.tight_layout()
        plt.savefig(output_path_low)
        plt.close()

        # 3. High-frequency part (your input X)
        output_path_high = os.path.join(BASE_DIR, TARGET_FOLDER_NAME, 'spectrogram_high_freq_part.png')
        print(f"[*] Saving high-frequency part to {output_path_high}")
        plt.figure(figsize=FIG_SIZE)
        # We need to specify the correct frequency offset for the y-axis
        librosa.display.specshow(high_freq_data, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='hz', y_coords=freqs[split_index:])
        plt.colorbar(format='%+2.0f dB')
        plt.title('High-Frequency Part (16 kHz and above)')
        plt.tight_layout()
        plt.savefig(output_path_high)
        plt.close()

        print("\n--- Visualization Complete! ---")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == '__main__':
    visualize_split()
