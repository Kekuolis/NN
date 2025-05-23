#include "cnn.h"
#include "load_files.h"
#include "load_model.h"
#include "segment_data.h"
#include <iostream>
#include <vector>
// Model works
// Loading params works
// Added dynamic file loading
// conversion back to sound file works
//
// I can add more layers
// I can add more filters
// I can add more parameters
// I can add multithreading???
// I can add loss output
// I can add
//
// write the report - lol

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  dynet::initialize(argc, argv);

  const std::string base_path =
      "/home/kek/Documents/rudens/praktika/prof_praktika/network/irasai/";
  const std::string model_save_path =
      "/home/kek/Documents/rudens/praktika/"
      "prof_praktika/network/param/params.model";
  const std::string test_wav_file_path = base_path + "L_RA_M4_01_10dB.wav";

  ParameterCollection pc;
  Speech_Denoising_Model model(pc);

  const uint32_t batchSize = 8;
  // this is for loading params
  // Check for "--load" flag in command-line arguments
  bool load_params = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--load") {
      load_params = true;
      break;
    }
  }

  if (load_params) {
    generate_denoised_files(
        std::regex(R"([LR]_[A-Z]{2}_[FM]\d+_[A-Z]{2}\d{3}_a\d{4}_\d+db\.wav)"),
        base_path + "TEST/NOISY/", batchSize);
    return 0;
  }

  const std::vector<SoundRealDataClean> training_data_clean =
      load_files<CleanTag>();
  const std::vector<SoundRealDataNoisy> training_data_noisy =
      load_files<NoisyTag>(
          false,
          std::regex(
              R"([LR]_[A-Z]{2}_[FM]\d+_[A-Z]{2}\d{3}_a\d{4}_\d+db\.wav)"));

  // Check for training/using the trained models params for audio output

  std::cout << "Total noisy training segments: " << training_data_noisy.size()
            << std::endl;

  const auto startTime = std::chrono::high_resolution_clock::now();

  model.train(training_data_noisy, training_data_clean, pc, 0.01, batchSize, 8);
  const auto endTime = std::chrono::high_resolution_clock::now();

  TextFileSaver saver(model_save_path);
  saver.save(pc);

  const std::chrono::duration<double> trainingDuration = endTime - startTime;
  std::cout << "Training time: " << trainingDuration.count() << " seconds"
            << std::endl;

  return 0;
}
