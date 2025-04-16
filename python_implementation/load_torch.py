import os
import re
import numpy as np
import librosa
import torch
import torch.nn as nn
import soundfile as sf  # to save denoised audio

#############################
# 1. Audio Loader Functions #
#############################

def load_audio(file_path, sr=16000, fixed_length=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    audio = np.array(audio, dtype=np.float32)
    if len(audio) < fixed_length:
        audio = np.pad(audio, (0, fixed_length - len(audio)), mode='constant')
    else:
        audio = audio[:fixed_length]
    return audio

def get_matched_file_paths(base_path, regex_pattern):
    matched_files = []
    for filename in os.listdir(base_path):
        if re.match(regex_pattern, filename):
            matched_files.append(os.path.join(base_path, filename))
        else:
            print(f"[SKIP] {filename} does not match the pattern.")
    matched_files.sort()
    print(f"[INFO] Total matched files in {base_path}: {len(matched_files)}")
    return matched_files

#############################
# 2. PyTorch Model          #
#############################
class ConvDenoisingAutoencoder(nn.Module):
    def __init__(self, input_channels=1, feature_dim=16):
        super(ConvDenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16),             # Match index 1
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),             # Match index 4
            nn.ReLU(),
            nn.Conv1d(32, feature_dim, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(feature_dim),    # Match index 7
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(feature_dim, 32, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.BatchNorm1d(32),             # Match index 1
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.BatchNorm1d(16),             # Match index 4
            nn.ReLU(),
            nn.ConvTranspose1d(16, input_channels, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # x = x.unsqueeze(1)  # [B, 1, L]
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)  # [B, 1, L']

        # If the output is not exactly the same length, crop or pad
        if decoded.size(-1) > x.size(-1):
            decoded = decoded[..., :x.size(-1)]
        elif decoded.size(-1) < x.size(-1):
            pad_len = x.size(-1) - decoded.size(-1)
            decoded = F.pad(decoded, (0, pad_len))  # pad at the end

        return decoded.squeeze(1)  # [B, L]


#############################
# 3. Denoising Function     #
#############################

def denoise_audio(model, audio, device):
    """
    Uses the provided model to denoise the given audio signal.
    
    Args:
        model (torch.nn.Module): The loaded denoising model.
        audio (np.array): The noisy audio signal.
        device (torch.device): Device to run the model on.
        
    Returns:
        np.array: The denoised audio signal.
    """
   # Convert the audio to a tensor, add batch and channel dimensions: [1, 1, L]
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        model.eval()
        denoised_tensor = model(audio_tensor)
    
    # Remove channel and batch dimensions: [1, 1, L] -> [L]
    denoised_audio = denoised_tensor.squeeze(0).squeeze(0).cpu().numpy()
    return denoised_audio

#############################
# 4. Main                   #
#############################

if __name__ == "__main__":
    # Define parameters
    input_dim = 16000          # length of the audio signal (samples)
    hidden_dim = 1024          # hidden layer size (must match training)
    sr = 16000                 # sampling rate
    fixed_length = 16000      # fixed length for audio signals
    
    # Path to the saved model parameters.
    saved_model_path = "denoising_autoencoder_final.pt"
    
    # The folder containing new noisy audio files for denoising
    noisy_input_folder = "/home/kek/Documents/rudens/praktika/prof_praktika/network/irasai/TEST/NOISY/"
    # Regex for matching noisy files. Adjust as needed.
    noisy_regex = r"([LR]_[A-Z]{2}_[FM]\d+_[A-Z]{2}\d{3}_a\d{4}_\d+db\.wav)"
    
    # Folder where denoised output will be saved.
    output_folder = "./denoised_output"
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the model architecture and state dictionary.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the convolutional model
    model = ConvDenoisingAutoencoder(input_channels=1, feature_dim=16).to(device)
    model.load_state_dict(torch.load(saved_model_path, map_location=device))
    print(f"[INFO] Loaded ConvDenoisingAutoencoder model from {saved_model_path}")


    # Find all the noisy files in the designated folder.
    noisy_files = get_matched_file_paths(noisy_input_folder, noisy_regex)
    
    # Process each noisy file.
    for file_path in noisy_files:
        print(f"[INFO] Denoising file: {os.path.basename(file_path)}")
        
        # Load the noisy audio.
        noisy_audio = load_audio(file_path, sr=sr, fixed_length=fixed_length)
        
        # Denoise the audio using the pre-trained model.
        denoised_audio = denoise_audio(model, noisy_audio, device)
        
        # Construct an output filename. For example, append '_denoised' to the name.
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        output_filepath = os.path.join(output_folder, f"{base_filename}_denoised.wav")
        
        # Save the denoised audio.
        sf.write(output_filepath, denoised_audio, sr)
        print(f"[INFO] Denoised audio saved to: {output_filepath}")
