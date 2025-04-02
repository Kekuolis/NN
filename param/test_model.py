import dynet as dy
import numpy as np
import scipy.io.wavfile as wav

def apply_dynet_model(model_path, input_wav, output_wav):
    # Load the trained DyNet model
    model = dy.Model()

    parameters = dy.()

    # Read the input WAV file
    sample_rate, audio_data = wav.read(input_wav)
    
    # Normalize the audio data
    audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data))
    
    # Process the audio using the trained model
    dy.renew_cg()  # Renew computation graph
    input_vector = dy.inputVector(audio_data.tolist())  # Convert to DyNet input format
    output_vector = param_collection * input_vector  # Apply model (assuming simple transformation)
    
    # Convert back to numpy
    processed_audio = output_vector.npvalue()
    
    # Denormalize back to original format
    processed_audio = (processed_audio * 32767).astype(np.int16)
    
    # Save the modified audio as a new WAV file
    wav.write(output_wav, sample_rate, processed_audio)
    
    print(f"Processed audio saved as {output_wav}")

# Example Usage
apply_dynet_model("./params.model", "/home/kek/Documents/rudens/praktika/prof_praktika/network/irasai/L_RA_M4_01_10dB.wav", "output.wav")
