#include "load_files.h"
#include "cnn.h"
#include "segment_data.h"

#include <filesystem>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "wav.h"

namespace fs = std::filesystem;

std::vector<std::string>
get_matched_file_paths(const std::string &basePath,
                       const std::regex &regexPattern) {
  std::vector<fs::directory_entry> entries;

  std::cout << "[INFO] Scanning directory: " << basePath << std::endl;

  try {
    for (const auto &entry : fs::directory_iterator(basePath)) {
      if (entry.is_regular_file()) {
        std::string fileName = entry.path().filename().string();
        // std::cout << "[DEBUG] Found file: " << fileName << std::endl;

        if (std::regex_match(fileName, regexPattern)) {
          //   std::cout << "[MATCH] " << fileName << " matches pattern." <<
          //   std::endl;
          entries.push_back(entry);
        } else {
          std::cout << "[SKIP] " << fileName << " does not match." << std::endl;
        }
      }
    }

    std::sort(entries.begin(), entries.end(),
              [](const fs::directory_entry &a, const fs::directory_entry &b) {
                return a.path().filename().string() <
                       b.path().filename().string();
              });

  } catch (const fs::filesystem_error &e) {
    std::cerr << "[ERROR] Filesystem error: " << e.what() << std::endl;
  }

  std::vector<std::string> matched_paths;
  for (const auto &entry : entries) {
    std::string fullPath = entry.path().string();
    matched_paths.push_back(fullPath);
    // std::cout << "[RESULT] Matched path: " << fullPath << std::endl;
  }

  std::cout << "[INFO] Total matched files: " << matched_paths.size()
            << std::endl;
  return matched_paths;
}
// This is a load files for training function
template <typename T>
std::vector<SoundRealData<T>> load_files(bool clean_or_noisy,
                                         std::regex pattern) {

  // Determine the base path using a ternary operator
  const char *base_path = clean_or_noisy
                              ? "/home/kek/Documents/rudens/praktika/"
                                "prof_praktika/network/irasai/TRAIN/"
                              : "/home/kek/Documents/rudens/praktika/"
                                "prof_praktika/network/irasai/TRAIN/noisy/";

  std::vector<std::string> data_paths;

  // Get the matched file paths based on the pattern
  data_paths = get_matched_file_paths(base_path, pattern);

  std::vector<SoundRealData<T>> training_data;

  // Iterate through the matched file paths
  for (int i = 0; i < data_paths.size(); i++) {
    const std::string filePath = data_paths[i];
    std::cout << "Processing file: " << filePath << std::endl;

    // Check if the file exists
    if (!std::filesystem::exists(filePath)) {
      std::cerr << "File not found: " << filePath << std::endl;
      continue;
    }

    // Read and process the file
    soundData data = readWav(filePath);
    std::vector<soundData> segments = segment_data(data);

    for (const auto &segment : segments) {
      SoundRealData<T> segment_data;
      segment_data.sound = vecToReal<int>(segment.monoSound);
      training_data.push_back(std::move(segment_data));
    }
  }

  // Return the processed data
  return training_data;
}

template std::vector<SoundRealData<NoisyTag>> load_files<NoisyTag>(bool,
                                                                   std::regex);
template std::vector<SoundRealData<CleanTag>> load_files<CleanTag>(bool,
                                                                   std::regex);

std::vector<SoundRealDataNoisy> load_single_file(std::string path) {

  std::vector<SoundRealDataNoisy> tmp;

  soundData data_noisy = readWav(path);
  std::vector<soundData> segments_noisy = segment_data(data_noisy);
  for (const auto &segment : segments_noisy) {
    SoundRealDataNoisy noisy_segment;
    noisy_segment.sound = vecToReal<int>(segment.monoSound);
    tmp.push_back(std::move(noisy_segment));
  }
  return tmp;
}
