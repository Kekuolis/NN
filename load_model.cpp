#include "load_model.h"
#include "load_files.h"
#include "cnn.h"
#include "dynet/io.h"
#include "dynet/model.h"
#include <vector>

using namespace dynet;

#include <fstream>
#include <vector>
#include <iostream>

void writeWavFile(const std::string& filename, const soundData& sound) {
    const header& hdr = sound.headerData;
    std::ofstream outFile(filename, std::ios::binary);
    
    if (!outFile) {
        std::cerr << "Failed to open file for writing: " << filename << "\n";
        return;
    }

    // Write WAV header
    outFile.write(hdr.riffHeader, 4);
    outFile.write(reinterpret_cast<const char*>(&hdr.wavSize), sizeof(hdr.wavSize));
    outFile.write(hdr.waveHeader, 4);
    outFile.write(hdr.fmtHeader, 4);
    outFile.write(reinterpret_cast<const char*>(&hdr.fmtChunkSize), sizeof(hdr.fmtChunkSize));
    outFile.write(reinterpret_cast<const char*>(&hdr.audioFormat), sizeof(hdr.audioFormat));
    outFile.write(reinterpret_cast<const char*>(&hdr.numChannels), sizeof(hdr.numChannels));
    outFile.write(reinterpret_cast<const char*>(&hdr.sampleRate), sizeof(hdr.sampleRate));
    outFile.write(reinterpret_cast<const char*>(&hdr.byteRate), sizeof(hdr.byteRate));
    outFile.write(reinterpret_cast<const char*>(&hdr.blockAlign), sizeof(hdr.blockAlign));
    outFile.write(reinterpret_cast<const char*>(&hdr.bitsPerSample), sizeof(hdr.bitsPerSample));
    outFile.write(hdr.dataHeader, 4);
    outFile.write(reinterpret_cast<const char*>(&hdr.dataSize), sizeof(hdr.dataSize));

    // Write audio data
    if (hdr.numChannels == 1) {
        // Mono
        for (int sample : sound.monoSound) {
            short s = static_cast<short>(sample);
            outFile.write(reinterpret_cast<const char*>(&s), sizeof(short));
        }
    } else {
        std::cerr << "Unsupported number of channels: " << hdr.numChannels << "\n";
    }

    outFile.close();
    std::cout << "WAV file written to: " << filename << "\n";
}


void load_model(uint batch_size, std::string filepath) {
  // (1) Re-create your parameter collection: add parameters with the same dimensions
  ParameterCollection pc;
  pc.add_parameters({2});
  std::vector<Parameter> parameters;
  Parameter tmp;
  LookupParameter l_param;{
    TextFileLoader loader("/home/kek/Documents/rudens/praktika/prof_praktika/network/param/params.model");
    parameters.push_back(loader.load_param(pc, "/_0"));
    parameters.push_back(loader.load_param(pc, "/_1"));
  }

  Speech_Denoising_Model loaded_params(pc);
  soundData outputFile = loaded_params.use_model(pc,parameters, filepath, batch_size);
  
  writeWavFile("/home/kek/Documents/rudens/praktika/prof_praktika/network/param/output_file.wav", outputFile);
}

void generate_denoised_files(std::regex reg_noisy,  const std::string base_path, uint batch_size) {
    std::vector<std::string> paths_noisy = get_matched_file_paths(base_path, reg_noisy);
  
  
    ParameterCollection pc;
      pc.add_parameters({2});
      std::vector<Parameter> parameters;
      LookupParameter l_param;{ // load model params
        TextFileLoader loader("/home/kek/Documents/rudens/praktika/prof_praktika/network/param/params.model");
        parameters.push_back(loader.load_param(pc, "/_0"));
        parameters.push_back(loader.load_param(pc, "/_1"));
    }
    Speech_Denoising_Model use_model(pc);
    for (int i =0; i < paths_noisy.size(); i++) {
      std::cout << "Processing file: " << paths_noisy[i] << std::endl;
      soundData tmp = use_model.use_model(pc,parameters, paths_noisy[i], batch_size);
      std::filesystem::path p( paths_noisy[i]);

      writeWavFile("/home/kek/Documents/rudens/praktika/prof_praktika/network/irasai/TEST/DENOISED/" + p.filename().string(), tmp);
    }
    
  }