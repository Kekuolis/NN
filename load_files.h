#pragma once
#include <regex>
#include "segment_data.h"

template <typename T>
std::vector<SoundRealData<T>>
 load_files(bool clean_or_noisy = true,
    std::regex pattern = std::regex(R"(^.+_.+_.+_.+_a\d{4}\.wav$)"));

std::vector<SoundRealDataNoisy> load_single_file(std::string path);