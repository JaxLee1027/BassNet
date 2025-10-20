import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# --- 1. CONFIGURATION ---

# The root directory where the dataset is located.
# This has been set to the path you provided.
BASE_DIR = '/home/jiayangli/Downloads/raf_dataset/archived/EmptyRoom/data'

# The specific directory to test.
TARGET_FOLDER_NAME = '000000'

# Input and output filenames.
AUDIO_FILENAME = 'rir.wav'
# Use a different name to identify this as a test output.
OUTPUT_FILENAME = 'spectrogram_TEST.png' 

# --- 2. SPECTROGRAM PARAMETERS ---
N_FFT = 2048
HOP_LENGTH = 512
FIG_SIZE = (10, 4)
CMAP = 'viridis'

# --- 3. MAIN TEST LOGIC ---

def run_test():
    """
    Processes the audio file in a single specified directory 
    and generates a spectrogram.
    """
    print("--- Starting Single File Test ---")

    # Construct the full paths.
    target_folder_path = os.path.join(BASE_DIR, TARGET_FOLDER_NAME)
    audio_path = os.path.join(target_folder_path, AUDIO_FILENAME)
    output_path = os.path.join(target_folder_path, OUTPUT_FILENAME)

    print(f"[*] Target directory: {target_folder_path}")
    print(f"[*] Looking for audio file: {audio_path}")

    # --- Sanity Checks ---
    # Check if the audio file exists before proceeding.
    if not os.path.exists(audio_path):
        print(f"\nERROR: Audio file not found at '{audio_path}'.")
        print("Please check the following:")
        print(f"1. If 'BASE_DIR' ('{BASE_DIR}') is set correctly.")
        print(f"2. If the directory '{TARGET_FOLDER_NAME}' exists inside the base directory.")
        print(f"3. If the file '{AUDIO_FILENAME}' is inside that directory.")
        return
        
    print(f"[+] Audio file found successfully!")

    try:
        # --- Core Processing Steps ---
        print("[*] Loading audio file...")
        # sr=None preserves the original sample rate.
        y, sr = librosa.load(audio_path, sr=None)
        print(f"[+] Audio loaded. Detected sample rate: {sr} Hz")

        print("[*] Calculating STFT...")
        S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
        
        print("[*] Converting amplitude to Decibels (dB)...")
        D = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        
        print("[*] Plotting spectrogram...")
        plt.figure(figsize=FIG_SIZE)
        librosa.display.specshow(D, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='linear')
        
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram: {TARGET_FOLDER_NAME}/{AUDIO_FILENAME}')
        plt.tight_layout()
        
        print(f"[*] Saving image to: {output_path}")
        plt.savefig(output_path)
        # Close the plot to free up memory.
        plt.close()

        print("\n--- Test Successful! ---")
        print(f"Spectrogram saved as '{OUTPUT_FILENAME}' in the directory '{TARGET_FOLDER_NAME}'. Please check the file.")

    except Exception as e:
        print(f"\nAn unexpected error occurred during processing: {e}")
        print("--- Test Failed ---")

if __name__ == '__main__':
    run_test()
