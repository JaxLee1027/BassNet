import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --- 1. CONFIGURATION (NO NEED TO CHANGE BASED ON LAST INFO) ---
# The root directory where the dataset is located.
BASE_DIR = '/home/jiayangli/Downloads/raf_dataset/archived/EmptyRoom/data'

# Folder range to process.
START_FOLDER = 0
END_FOLDER = 47483

# --- 2. FILENAMES (UNCHANGED) ---
AUDIO_FILENAME = 'rir.wav'
# Output for visualization
VISUAL_OUTPUT_FILENAME = 'spectrogram.png'
# Outputs for model training data
HIGH_FREQ_OUTPUT_FILENAME = 'high_freq_data.npy'
LOW_FREQ_OUTPUT_FILENAME = 'low_freq_data.npy'

# --- 3. PROCESSING PARAMETERS ---
SAMPLE_RATE = 48000
N_FFT = 2048
HOP_LENGTH = 512
SPLIT_FREQ_HZ = 16000 # The frequency to split at (16 kHz)

# --- 4. MAIN PROCESSING LOGIC ---

def process_all_files():
    """
    Iterates through all specified folders, processes each rir.wav,
    and saves a visual spectrogram (PNG) and the split numerical data (NPY).
    """
    print("--- Starting Full Dataset Processing ---")
    print(f"This script will generate 3 files per audio source:")
    print(f"  - {VISUAL_OUTPUT_FILENAME} (for visual check)")
    print(f"  - {HIGH_FREQ_OUTPUT_FILENAME} (model input X)")
    print(f"  - {LOW_FREQ_OUTPUT_FILENAME} (model target Y)")

    folder_range = range(START_FOLDER, END_FOLDER + 1)
    for i in tqdm(folder_range, desc="Processing Audio Files"):
        folder_name = f"{i:06d}"
        folder_path = os.path.join(BASE_DIR, folder_name)
        
        # Define all paths
        audio_path = os.path.join(folder_path, AUDIO_FILENAME)
        visual_output_path = os.path.join(folder_path, VISUAL_OUTPUT_FILENAME)
        high_freq_output_path = os.path.join(folder_path, HIGH_FREQ_OUTPUT_FILENAME)
        low_freq_output_path = os.path.join(folder_path, LOW_FREQ_OUTPUT_FILENAME)

        # --- Resumability Check ---
        # If all three output files already exist, skip this folder.
        if (os.path.exists(visual_output_path) and
            os.path.exists(high_freq_output_path) and
            os.path.exists(low_freq_output_path)):
            continue
            
        # Check if source audio exists
        if not os.path.exists(audio_path):
            continue

        try:
            # --- A. Load and Calculate Full Spectrogram ---
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
            D = librosa.amplitude_to_db(np.abs(S), ref=np.max)

            # --- B. Save the Full Visual Spectrogram (PNG) ---
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(D, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='linear')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Spectrogram: {folder_name}/{AUDIO_FILENAME}')
            plt.tight_layout()
            plt.savefig(visual_output_path)
            plt.close() # Always close plots to free memory in a long loop

            # --- C. Split the Numerical Data ---
            # Get the frequency values for each row in the spectrogram matrix
            freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
            # Find the index of the frequency bin closest to our split frequency
            split_index = np.argmin(np.abs(freqs - SPLIT_FREQ_HZ))
            
            # Slice the matrix
            low_freq_data = D[:split_index, :]
            high_freq_data = D[split_index:, :]

            # --- D. Save the Split Data Matrices (NPY) ---
            np.save(low_freq_output_path, low_freq_data)
            np.save(high_freq_output_path, high_freq_data)

        except Exception as e:
            print(f"An error occurred while processing {audio_path}: {e}")

    print("\n--- All files processed successfully! ---")

if __name__ == '__main__':
    # Verify that the base directory exists
    if not os.path.isdir(BASE_DIR) or 'path/to' in BASE_DIR:
        print(f"ERROR: The 'BASE_DIR' is not set correctly. Please edit the script.")
        print(f"Current BASE_DIR: '{BASE_DIR}'")
    else:
        process_all_files()