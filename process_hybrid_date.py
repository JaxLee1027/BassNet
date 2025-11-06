import os
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from tqdm import tqdm
import warnings

# --- Configuration ---
BASE_DIR = '/home/jiayangli/Downloads/raf_dataset/archived/EmptyRoom/data'
START_FOLDER = 0
END_FOLDER = 47483 # Process the entire dataset

# --- File names ---
# Source audio files
AUDIO_FILENAME = 'rir.wav'
# Output filenames
LOW_FREQ_WAVE_OUT = 'low_freq_wave.npy'
HIGH_FREQ_WAVE_OUT = 'high_freq_wave.npy'
HIGH_FREQ_SPEC_OUT = 'high_freq_spec.npy'
WAVEFORM_PLOT_OUT = 'filtered_waveforms.png'

# --- Processing parameters ---
SAMPLE_RATE = 48000
DURATION_TO_KEEP = 0.4 # The 0.4 second truncation
SPLIT_FREQ_HZ = 16000 # Frequency to split low and high
FILTER_ORDER = 5 # Order of the Butterworth filter

# --- Spectrogram parameters ---
N_FFT = 2048
HOP_LENGTH = 512

# --- Butterworth filter design ---
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = lfilter(b, a, data)
    return y

# --- Main processing loop ---
def process_folder(folder_path, num_samples_to_keep):
    """
    Process a single folder: loads, truncates, filters,
    generates spectrogram, and saves all 3 .npy files.
    """
    # Define all file paths
    audio_path = os.path.join(folder_path, AUDIO_FILENAME)
    low_freq_wave_path = os.path.join(folder_path, LOW_FREQ_WAVE_OUT)
    high_freq_wave_path = os.path.join(folder_path, HIGH_FREQ_WAVE_OUT)
    high_freq_spec_path = os.path.join(folder_path, HIGH_FREQ_SPEC_OUT)
    plot_path = os.path.join(folder_path, WAVEFORM_PLOT_OUT)

    # --- Resumability check ---
    if (os.path.exists(low_freq_wave_path) and
        os.path.exists(high_freq_wave_path) and
        os.path.exists(high_freq_spec_path)):
        print(f"Skipping {folder_path}, already processed.")
        return
    
    # Check if source audio exists
    if not os.path.exists(audio_path):
        warnings.warn(f"Audio file not found: {audio_path}. Skipping.")
        return
    
    try:
        # --- Load and truncate audio ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Load audio
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        
        # Truncate to the first DURATION_TO_KEEP seconds
        y = y[:num_samples_to_keep]

        # Pad with zeros if shorter than required
        if len(y) < num_samples_to_keep:
            y = np.pad(y, (0, num_samples_to_keep - len(y)), 'constant')
        
        # --- Filter waveforms (Inputs X1 and Target Y) ---
        # Target Y: Low-frequency waveform
        y_low = butter_lowpass_filter(y, SPLIT_FREQ_HZ, SAMPLE_RATE, FILTER_ORDER)
        # Input X1: High-frequency waveform
        y_high = butter_highpass_filter(y, SPLIT_FREQ_HZ, SAMPLE_RATE, FILTER_ORDER)

        # --- Generate spectrogram (Input X2) ---
        S_high = librosa.stft(y_high, n_fft=N_FFT, hop_length=HOP_LENGTH)
        # We use np.abs(), so we are still only feeding magnitude to the 2D encoder
        D_high = librosa.amplitude_to_db(np.abs(S_high), ref=np.max)

        # --- Save the 3 data files ---
        np.save(low_freq_wave_path, y_low.astype(np.float32))
        np.save(high_freq_wave_path, y_high.astype(np.float32))
        np.save(high_freq_spec_path, D_high.astype(np.float32))

        # --- Save a verification plot ---
        # Set to False to disable
        SAVE_PLOT = True
        if SAVE_PLOT and not os.path.exists(plot_path):
            fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            librosa.display.waveshow(y_low, sr=SAMPLE_RATE, ax=ax[0], alpha=0.8, label='Low-Freq (<16kHz)')
            ax[0].legend()
            librosa.display.waveshow(y_high, sr=SAMPLE_RATE, ax=ax[1], color='r', alpha=0.8, label='High-Freq (>16kHz)')
            ax[1].legend()
            plt.suptitle('Filtered Waveforms: {os.path.basename(folder_path)}')
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close(fig)
    
    except Exception as e:
        warnings.warn(f"Error processing {folder_path}: {e}")
        if os.path.exists(low_freq_wave_path):
            os.remove(low_freq_wave_path)
        if os.path.exists(high_freq_wave_path):
            os.remove(high_freq_wave_path)
        if os.path.exists(high_freq_spec_path):
            os.remove(high_freq_spec_path)

# --- Main execution ---
if __name__ == '__main__':
    print("--- Starting Hybrid Dataset Processing ---")
    print(f"This script will generate 3 files per audio source:")
    print(f"  - {LOW_FREQ_WAVE_OUT} (Target Y, 1D Wave)")
    print(f"  - {HIGH_FREQ_WAVE_OUT} (Input X1, 1D Wave)")
    print(f"  - {HIGH_FREQ_SPEC_OUT} (Input X2, 2D Spec)")

    num_samples_to_keep = int(DURATION_TO_KEEP * SAMPLE_RATE)
    print(f"Each audio will be truncated/padded to {DURATION_TO_KEEP} seconds ({num_samples_to_keep} samples).")
    
    if not os.path.exists(BASE_DIR):
        raise FileNotFoundError(f"Base directory not found: {BASE_DIR}")
        sys.exit(1)
    
    folder_range = range(START_FOLDER, END_FOLDER+1)

    # Process each folder
    for i in tqdm(folder_range, desc="Processing Audio Files"): 
        folder_name = f"{i:06d}"
        folder_path = os.path.join(BASE_DIR, folder_name)
        
        if os.path.isdir(folder_path):
            process_folder(folder_path, num_samples_to_keep)
        
    print("\n--- All files processed successfully! ---")