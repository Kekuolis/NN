#pragma once
// #include "cnn.h"
#include "wav.h"
#include "dynet/tensor.h"


template <typename SoundType>
struct SoundRealData {
    std::vector<dynet::real> sound;
    int file_number;
    int file_segment_count;
    soundData sound_data;
};
using SoundRealDataClean = SoundRealData<struct CleanTag>;
using SoundRealDataNoisy = SoundRealData<struct NoisyTag>;

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
template <typename T>
std::vector<T> batch_noisy_data(std::string prefix = "L_RA_M4_01_", std::string suffix = "dB.wav");