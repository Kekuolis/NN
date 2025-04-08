#include "cnn.h"
#include "load_model.h"
#include "segment_data.h"
#include "wav.h"
#include <iostream>
#include <vector>
#include "load_files.h"
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

  std::vector<SoundRealDataClean> trainingDataClean = load_files<CleanTag>();
  std::vector<SoundRealDataNoisy> trainingDataNoisy = load_files<NoisyTag>(false, std::regex(R"([LR]_[A-Z]{2}_[FM]\d+_[A-Z]{2}\d{3}_a\d{4}_\d+db\.wav)"));

  const uint32_t batchSize = 1;
  // Check for training/using the trained models params for audio output
  if (true) {
    load_model(batchSize, testWavPath);
    return 0;
  }

  std::cout << "Total noisy training segments: " << trainingDataNoisy.size()
            << std::endl;

  const auto startTime = std::chrono::high_resolution_clock::now();
  model.train(trainingDataNoisy, trainingDataClean, pc, 0.01, batchSize, 8);
  const auto endTime = std::chrono::high_resolution_clock::now();

  TextFileSaver saver(modelSavePath);
  saver.save(pc);

  const std::chrono::duration<double> trainingDuration = endTime - startTime;
  std::cout << "Training time: " << trainingDuration.count() << " seconds"
            << std::endl;

  return 0;
}
