#include "cnn.h"
#include "load_model.h"
#include "segment_data.h"
#include "wav.h"
#include <iostream>
#include <vector>

// need to fix clean data mapping, this kinda works?
// so what the  do i do here?
//
// make dynamic file loading - this is for the end of the project
// create a function to output clean sound files - i want to do this now.
// write a function that can consume params only and run the model on data
// write the report - lol
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Add necessary includes for dynet and your custom headers

int main(int argc, char **argv) {
  dynet::initialize(argc, argv);

  const std::string basePath =
      "/home/kek/Documents/rudens/praktika/prof_praktika/network/irasai/";
  const std::string modelSavePath = "/home/kek/Documents/rudens/praktika/"
                                    "prof_praktika/network/param/params.model";
  const std::string testWavPath = basePath + "L_RA_M4_01_10dB.wav";

  ParameterCollection pc;
  Speech_Denoising_Model model(pc);

  std::vector<soundRealDataClean> trainingDataClean;
  std::vector<soundRealDataNoisy> trainingDataNoisy;

  const std::vector<std::string> cleanDataPaths = {
      "L_RA_M4_01.wav", "L_RA_M4_02.wav", "L_RA_M5_01.wav", "L_RA_M5_02.wav",
      "R_RD_F3_01.wav", "R_RD_M4_01.wav", "R_RD_F3_02.wav", "R_RD_F3_03.wav"};

  const std::vector<std::string> noisyDataPrefixes = {
      "L_RA_M4_01_", "L_RA_M4_02_", "L_RA_M5_01_", "L_RA_M5_02_",
      "R_RD_F3_01_", "R_RD_M4_01_", "R_RD_F3_02_", "R_RD_F3_03_"};

  if (cleanDataPaths.size() != noisyDataPrefixes.size()) {
    std::cerr << "Mismatch between clean and noisy data paths!" << std::endl;
    return 1;
  }

  for (size_t i = 0; i < cleanDataPaths.size(); ++i) {
    const std::string cleanFilePath = basePath + cleanDataPaths[i];

    if (!std::filesystem::exists(cleanFilePath)) {
      std::cerr << "File not found: " << cleanFilePath << std::endl;
      continue;
    }

    soundData dataClean = readWav(cleanFilePath);
    std::vector<soundData> segmentsClean = segment_data(dataClean);

    for (const auto &segment : segmentsClean) {
      soundRealDataClean cleanSegment;
      cleanSegment.clean_sound = vecToReal<int>(segment.monoSound);
      trainingDataClean.push_back(std::move(cleanSegment));
    }

    std::vector<soundRealDataNoisy> noisySegments =
        batch_noisy_data(noisyDataPrefixes[i]);
    trainingDataNoisy.insert(trainingDataNoisy.end(),
                             std::make_move_iterator(noisySegments.begin()),
                             std::make_move_iterator(noisySegments.end()));

    std::cout << "Loaded " << noisySegments.size()
              << " noisy segments for: " << noisyDataPrefixes[i] << std::endl;
  }

  const uint32_t batchSize = 1;

  if (false) {
    load_model(trainingDataNoisy, batchSize, testWavPath);
    return 0;
  }

  std::cout << "Total noisy training segments: " << trainingDataNoisy.size()
            << std::endl;

  const auto startTime = std::chrono::high_resolution_clock::now();
  model.train(trainingDataNoisy, trainingDataClean, pc, 0.01, batchSize);
  const auto endTime = std::chrono::high_resolution_clock::now();

  TextFileSaver saver(modelSavePath);
  saver.save(pc);

  const std::chrono::duration<double> trainingDuration = endTime - startTime;
  std::cout << "Training time: " << trainingDuration.count() << " seconds"
            << std::endl;

  return 0;
}
