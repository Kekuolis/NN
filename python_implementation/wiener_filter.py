import os
import re
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import wiener
from glob import glob

# === Your settings ===
noisy_input_folder = "/home/kek/Documents/rudens/praktika/prof_praktika/network/irasai/TEST/NOISY/"
noisy_regex = r"([LR]_[A-Z]{2}_[FM]\d+_[A-Z]{2}\d{3}_a\d{4}_\d+db\.wav)"

output_folder = "./denoised_output_wiener"
os.makedirs(output_folder, exist_ok=True)

# === Helper: find files matching the regex ===
def get_matched_file_paths(folder, pattern):
    matched_files = []
    regex = re.compile(pattern)
    for file_path in glob(os.path.join(folder, "*.wav")):
        if regex.search(os.path.basename(file_path)):
            matched_files.append(file_path)
    return matched_files

# === Wiener denoising ===
def wiener_denoise(input_wav_path, output_wav_path):
    sample_rate, data = wav.read(input_wav_path)

    # Normalize if data is int16
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0

    # If stereo, process each channel
    if len(data.shape) == 2:
        denoised = np.stack([wiener(data[:, ch]) for ch in range(data.shape[1])], axis=1)
    else:
        denoised = wiener(data)

    # Rescale and convert back to int16
    denoised = np.clip(denoised * 32768.0, -32768, 32767).astype(np.int16)

    wav.write(output_wav_path, sample_rate, denoised)

# === Process all matched noisy files ===
noisy_files = get_matched_file_paths(noisy_input_folder, noisy_regex)

for file_path in noisy_files:
    filename = os.path.basename(file_path)
    output_path = os.path.join(output_folder, filename)
    print(f"Denoising {filename}...")
    wiener_denoise(file_path, output_path)

print("Denoising complete.")
