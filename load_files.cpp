#include "load_files.h"
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

namespace fs = std::filesystem;

std::vector<std::string>
get_matched_file_paths(const std::string &basePath,
                       const std::regex &regexPattern) {
  std::vector<fs::directory_entry> entries;
  std::regex pattern(regexPattern);

  std::cout << "[INFO] Scanning directory: " << basePath << std::endl;

  try {
    for (const auto &entry : fs::directory_iterator(basePath)) {
      if (entry.is_regular_file()) {
        std::string fileName = entry.path().filename().string();
        // std::cout << "[DEBUG] Found file: " << fileName << std::endl;

        if (std::regex_match(fileName, pattern)) {
        //   std::cout << "[MATCH] " << fileName << " matches pattern." << std::endl;
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

  std::cout << "[INFO] Total matched files: " << matched_paths.size() << std::endl;
  return matched_paths;
}

template <typename T>
std::vector<SoundRealData<T>>
load_files(bool clean_or_noisy,
           std::regex pattern) {

  if (clean_or_noisy) {
    const std::string base_path = "/home/kek/Documents/rudens/praktika/"
                                  "prof_praktika/network/irasai/TRAIN/";

    std::vector<std::string> clean_data_paths;

    clean_data_paths = get_matched_file_paths(base_path, pattern);

    std::vector<SoundRealData<T>> training_data_clean;

    for (int i = 0; i < clean_data_paths.size(); i++) {
      const std::string cleanFilePath = clean_data_paths[i];
      std::cout << "Processing file: " << cleanFilePath << std::endl;

      if (!std::filesystem::exists(clean_data_paths[i])) {
        std::cerr << "File not found: " << cleanFilePath << std::endl;
        continue;
      }
      soundData dataClean = readWav(cleanFilePath);
      std::vector<soundData> segmentsClean = segment_data(dataClean);

      for (const auto &segment : segmentsClean) {
        SoundRealData<T> clean_segment;
        clean_segment.sound = vecToReal<int>(segment.monoSound);
        training_data_clean.push_back(std::move(clean_segment));
      }
    }
    return training_data_clean;
  } else {
    const std::string base_path = "/home/kek/Documents/rudens/praktika/"
                                  "prof_praktika/network/irasai/TRAIN/noisy/";
    std::vector<std::string> noisy_data_paths;
    noisy_data_paths = get_matched_file_paths(base_path, pattern);
    std::vector<SoundRealData<T>> training_data_noisy;

    for (auto i : noisy_data_paths) {
      soundData data_noisy = readWav(i);
      std::vector<soundData> segments_noisy = segment_data(data_noisy);
      for (const auto segment : segments_noisy) {
        SoundRealData<T> noisy_segment;
        noisy_segment.sound = vecToReal<int>(segment.monoSound);
        training_data_noisy.push_back(std::move(noisy_segment));
      }
    }
    return training_data_noisy;
  }
}

template std::vector<SoundRealData<NoisyTag>> load_files<NoisyTag>(bool, std::regex);
template std::vector<SoundRealData<CleanTag>> load_files<CleanTag>(bool, std::regex);

std::vector<SoundRealDataNoisy> load_single_file(std::string path){
    
    std::vector<SoundRealDataNoisy> training_data_noisy;
    
    soundData data_noisy = readWav(path);
    std::vector<soundData> segments_noisy = segment_data(data_noisy);
    for (const auto segment : segments_noisy) {
    SoundRealDataNoisy noisy_segment;
    noisy_segment.sound = vecToReal<int>(segment.monoSound);
    training_data_noisy.push_back(std::move(noisy_segment));
    }
    return training_data_noisy;
}