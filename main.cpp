#include "cnn.h"
#include "load_model.h"
#include "segment_data.h"
#include "wav.h"
#include <iostream>
#include <iterator>
#include <vector>
#include "load_files.h"
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
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>


// Add necessary includes for dynet and your custom headers

int main(int argc, char **argv) {
  dynet::initialize(argc, argv);

  const std::string base_path =
      "/home/kek/Documents/rudens/praktika/prof_praktika/network/irasai/";
  const std::string model_save_path = "/home/kek/Documents/rudens/praktika/"
                                    "prof_praktika/network/param/params.model";
  const std::string test_wav_file_path = base_path + "L_RA_M4_01_10dB.wav";

  ParameterCollection pc;
  Speech_Denoising_Model model(pc);
  
  const uint32_t batchSize = 1;
  // this is for loading params
  if (false) {
  // Re-create your parameter collection: add parameters with the same dimensions

    ParameterCollection pc;
    pc.add_parameters({2});
    std::vector<Parameter> parameters;
    LookupParameter l_param;{ // load model params
      TextFileLoader loader("/home/kek/Documents/rudens/praktika/prof_praktika/network/param/params.model");
      parameters.push_back(loader.load_param(pc, "/_0"));
      parameters.push_back(loader.load_param(pc, "/_1"));
    }
    generate_denoised_files(
        std::regex(R"([LR]_[A-Z]{2}_[FM]\d+_[A-Z]{2}\d{3}_a\d{4}_\d+db\.wav)"),
        base_path + "TEST/NOISY/", batchSize);
    return 0;
  }
  std::vector<SoundRealDataClean> trainingDataClean = load_files<CleanTag>();
  std::vector<SoundRealDataNoisy> trainingDataNoisy = load_files<NoisyTag>(false, std::regex(R"([LR]_[A-Z]{2}_[FM]\d+_[A-Z]{2}\d{3}_a\d{4}_\d+db\.wav)"));

  // Check for training/using the trained models params for audio output

  std::cout << "Total noisy training segments: " << trainingDataNoisy.size()
            << std::endl;

  const auto startTime = std::chrono::high_resolution_clock::now();
  model.train(trainingDataNoisy, trainingDataClean, pc, 0.01, batchSize, 8);
  const auto endTime = std::chrono::high_resolution_clock::now();

  TextFileSaver saver(model_save_path);
  saver.save(pc);

  const std::chrono::duration<double> trainingDuration = endTime - startTime;
  std::cout << "Training time: " << trainingDuration.count() << " seconds"
            << std::endl;

  return 0;
}
