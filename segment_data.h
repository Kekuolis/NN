
#include "wav.h"
#include "cnn.h"

std::vector<soundData> segment_data(const soundData &data);

// template <typename T>
// std::vector<real> vecToReal(const std::vector<T> &input);
template <typename T>
std::vector<real> vecToReal(const std::vector<T> &input) {
    std::vector<real> tmp;
    tmp.reserve(input.size());  // Optimize memory allocation
    for (const auto& val : input) {
        tmp.push_back(static_cast<real>(val));
    }
    return tmp;
}
template <typename T>
std::vector<T> realToVec(std::vector<real> &input);
std::vector<soundRealDataNoisy> batch_noisy_data(std::string prefix = "L_RA_M4_01_", std::string suffix = "dB.wav");