#include "load_model.h"
#include "cnn.h"
#include "dynet/io.h"
#include "dynet/model.h"
#include "load_files.h"
#include <vector>

using namespace dynet;

#include <fstream>
#include <iostream>
#include <vector>

void writeWavFile(const std::string &filename, const soundData &sound) {
  const header &hdr = sound.headerData;
  std::ofstream outFile(filename, std::ios::binary);

  if (!outFile) {
    std::cerr << "Failed to open file for writing: " << filename << "\n";
    return;
  }

  // Write WAV header
  outFile.write(hdr.riffHeader, 4);
  outFile.write(reinterpret_cast<const char *>(&hdr.wavSize),
                sizeof(hdr.wavSize));
  outFile.write(hdr.waveHeader, 4);
  outFile.write(hdr.fmtHeader, 4);
  outFile.write(reinterpret_cast<const char *>(&hdr.fmtChunkSize),
                sizeof(hdr.fmtChunkSize));
  outFile.write(reinterpret_cast<const char *>(&hdr.audioFormat),
                sizeof(hdr.audioFormat));
  outFile.write(reinterpret_cast<const char *>(&hdr.numChannels),
                sizeof(hdr.numChannels));
  outFile.write(reinterpret_cast<const char *>(&hdr.sampleRate),
                sizeof(hdr.sampleRate));
  outFile.write(reinterpret_cast<const char *>(&hdr.byteRate),
                sizeof(hdr.byteRate));
  outFile.write(reinterpret_cast<const char *>(&hdr.blockAlign),
                sizeof(hdr.blockAlign));
  outFile.write(reinterpret_cast<const char *>(&hdr.bitsPerSample),
                sizeof(hdr.bitsPerSample));
  outFile.write(hdr.dataHeader, 4);
  outFile.write(reinterpret_cast<const char *>(&hdr.dataSize),
                sizeof(hdr.dataSize));

  // Write audio data
  if (hdr.numChannels == 1) {
    // Mono
    for (int sample : sound.monoSound) {
      short s = static_cast<short>(sample);
      outFile.write(reinterpret_cast<const char *>(&s), sizeof(short));
    }
  } else {
    std::cerr << "Unsupported number of channels: " << hdr.numChannels << "\n";
  }

  outFile.close();
  std::cout << "WAV file written to: " << filename << "\n";
}

void generate_denoised_files(std::regex reg_noisy, const std::string base_path,
                             uint batch_size) {
  std::vector<std::string> paths_noisy =
      get_matched_file_paths(base_path, reg_noisy);

  ParameterCollection pc;
  // pc.add_parameters({2});
  
  LookupParameter p_c;
  Parameter p_R;
  Parameter p_bias;

  TextFileLoader loader("/home/kek/Documents/rudens/praktika/prof_praktika/"
    "network/param/params.model");

  loader.populate(pc, "/_0");
  loader.populate(pc, "/_1");
  loader.populate(pc, "/_2");
  p_R = loader.load_param(pc, "/_1");
  p_bias = loader.load_param(pc, "/_2");
  p_c = loader.load_lookup_param(pc, "/_0");
  Speech_Denoising_Model use_model(pc);
  for (int i = 0; i < paths_noisy.size(); i++) {
    std::cout << "Processing file: " << paths_noisy[i] << std::endl;
    soundData tmp =
        use_model.use_model(p_c, p_R, p_bias, paths_noisy[i], batch_size);
    std::filesystem::path p(paths_noisy[i]);

  //   writeWavFile("/home/kek/Documents/rudens/praktika/prof_praktika/network/"
  //                "irasai/TEST/DENOISED/" +
  //                    p.filename().string(),
  //                tmp);
  }
}