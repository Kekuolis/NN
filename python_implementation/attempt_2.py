import os
import re
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim

#############################
# 1. Data Loader Functions  #
#############################

def get_matched_file_paths(base_path, regex_pattern):
    matched_files = []
    for filename in os.listdir(base_path):
        if re.match(regex_pattern, filename):
            matched_files.append(os.path.join(base_path, filename))
        else:
            print(f"[SKIP] {filename} does not match.")
    matched_files.sort()
    print(f"[INFO] Total matched files in {base_path}: {len(matched_files)}")
    return matched_files

def load_dataset(clean_regex, noisy_regex, clean_base, noisy_base):
    clean_paths = get_matched_file_paths(clean_base, clean_regex)
    noisy_paths = get_matched_file_paths(noisy_base, noisy_regex)
    
    noisy_dict = {}
    for path in noisy_paths:
        base_noisy = os.path.splitext(os.path.basename(path))[0]
        noisy_prefix = base_noisy.rsplit('_', 1)[0]
        noisy_dict.setdefault(noisy_prefix, []).append(path)
    
    dataset = []
    for clean_path in clean_paths:
        clean_prefix = os.path.splitext(os.path.basename(clean_path))[0]
        if clean_prefix in noisy_dict and len(noisy_dict[clean_prefix]) == 8:
            dataset.append((clean_path, noisy_dict[clean_prefix]))
        else:
            print(f"[WARNING] Clean file {clean_path} does not have 8 corresponding noisy files.")
    print(f"[INFO] Total paired dataset items: {len(dataset)}")
    return dataset

#############################
# 2. Audio Loader
#############################

def load_audio(file_path, sr=16000, fixed_length=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    audio = np.array(audio, dtype=np.float32)
    if len(audio) < fixed_length:
        audio = np.pad(audio, (0, fixed_length - len(audio)), mode='constant')
    else:
        audio = audio[:fixed_length]
    return audio

#############################
# 3. PyTorch Model
#############################


class ConvDenoisingAutoencoder(nn.Module):
    def __init__(self, input_channels=1, feature_dim=16):
        super(ConvDenoisingAutoencoder, self).__init__()
        # Encoder: progressively reduce temporal dimension while increasing features
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=15, stride=2, padding=7),  # [B, 16, L/2]
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7),              # [B, 32, L/4]
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, feature_dim, kernel_size=15, stride=2, padding=7),       # [B, feature_dim, L/8]
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )
        # Decoder: gradually upsample back to original signal length
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(feature_dim, 32, kernel_size=15, stride=2, padding=7, output_padding=1),  # [B, 32, L/4]
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=15, stride=2, padding=7, output_padding=1),           # [B, 16, L/2]
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.ConvTranspose1d(16, input_channels, kernel_size=15, stride=2, padding=7, output_padding=1), # [B, 1, L]
            nn.Tanh()  # Assuming normalized audio within (-1, 1)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, L]
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)  # [B, 1, L']

        # If the output is not exactly the same length, crop or pad
        if decoded.size(-1) > x.size(-1):
            decoded = decoded[..., :x.size(-1)]
        elif decoded.size(-1) < x.size(-1):
            pad_len = x.size(-1) - decoded.size(-1)
            decoded = F.pad(decoded, (0, pad_len))  # pad at the end

        return decoded.squeeze(1)  # [B, L]


# Improved training loop with scheduler and enhanced logging

def train_denoising_autoencoder(dataset, model, optimizer, num_epochs=10, sr=16000, fixed_length=16000):
    criterion = nn.MSELoss()
    # Create a learning rate scheduler for dynamic adjustment
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0.0
        sample_count = 0
        print(f"[INFO] Epoch {epoch+1}/{num_epochs} starting...")

        for idx, (clean_path, noisy_paths) in enumerate(dataset):
            # Load the clean reference audio
            clean_audio = load_audio(clean_path, sr=sr, fixed_length=fixed_length)
            clean_tensor = torch.from_numpy(clean_audio).float()

            for noisy_path in noisy_paths:
                # Load the noisy audio and convert to tensor
                noisy_audio = load_audio(noisy_path, sr=sr, fixed_length=fixed_length)
                noisy_tensor = torch.from_numpy(noisy_audio).float()
                
                # Expand dimensions to create batch size of 1 (if using batch normalization)
                clean_tensor_batch = clean_tensor.unsqueeze(0)
                noisy_tensor_batch = noisy_tensor.unsqueeze(0)

                optimizer.zero_grad()
                reconstruction = model(noisy_tensor_batch)
                loss = criterion(reconstruction, clean_tensor_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                sample_count += 1

                print(f"  [TRACE] Sample {idx+1}, Loss: {loss.item():.6f}")

        avg_loss = total_loss / sample_count if sample_count > 0 else float('inf')
        print(f"[INFO] Epoch {epoch+1} completed. Average Loss: {avg_loss:.6f}")
        
        # Adjust learning rate based on loss plateauing
        scheduler.step(avg_loss)
        
        # Optiona
#############################
# 5. Save Model Parameters
#############################

def save_model_parameters(model, filepath):
    """
    Saves the model parameters to a file.
    
    Args:
        model (torch.nn.Module): The model to be saved.
        filepath (str): Path where the state dictionary will be saved.
    """
    torch.save(model.state_dict(), filepath)
    print(f"[INFO] Model parameters saved to {filepath}")

#############################
# 6. Main
#############################

if __name__ == "__main__":
    clean_base = "./../irasai/TRAIN/"
    noisy_base = "./../irasai/TRAIN/noisy"

    clean_regex = r"(^.+_.+_.+_.+_a\d{4}\.wav)"
    noisy_regex = r"([LR]_[A-Z]{2}_[FM]\d+_[A-Z]{2}\d{3}_a\d{4}_\d+db\.wav)"

    dataset = load_dataset(clean_regex, noisy_regex, clean_base, noisy_base)

    input_dim = 16000
    hidden_dim = 1024
    num_epochs = 20

    model = ConvDenoisingAutoencoder(input_channels=1, feature_dim=16)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    train_denoising_autoencoder(dataset, model, optimizer, num_epochs=num_epochs,
                                sr=16000, fixed_length=input_dim)
    
    # Save the final model parameters after training
    save_model_parameters(model, "denoising_autoencoder_final.pt")
