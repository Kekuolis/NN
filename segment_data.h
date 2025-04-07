#pragma once
// #include "cnn.h"
#include "wav.h"
#include "dynet/tensor.h"


struct soundRealDataClean {
    std::vector<dynet::real> clean_sound;
    std::vector<dynet::real> file_number;
};
struct soundRealDataNoisy {
std::vector<dynet::real> noisy_sound;
std::vector<dynet::real> file_number;
};

std::vector<soundData> segment_data(const soundData &data);

// template <typename T>
// std::vector<real> vecToReal(const std::vector<T> &input);
template <typename T>
std::vector<dynet::real> vecToReal(const std::vector<T> &input) {
    std::vector<dynet::real> tmp;
    tmp.reserve(input.size());  // Optimize memory allocation
    for (const auto& val : input) {
        tmp.push_back(static_cast<dynet::real>(val));
    }
    return tmp;
}
template <typename T>
std::vector<T> realToVec(std::vector<dynet::real> &input);
std::vector<soundRealDataNoisy> batch_noisy_data(std::string prefix = "L_RA_M4_01_", std::string suffix = "dB.wav");