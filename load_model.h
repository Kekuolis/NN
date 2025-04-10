#pragma once
#include "dynet/rnn-state-machine.h"
#include "dynet/tensor.h"
#include "wav.h"
#include "segment_data.h"
#include "cnn.h"
#include "regex"
#include <dynet/dict.h>
#include <dynet/expr.h>
#include <dynet/lstm.h>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

// std::string noisy_training_data_location =  "/home/kek/Documents/rudens/praktika/prof_praktika/network/irasai/TRAIN/noisy/";

void load_model(uint batch_size = 8, std::string filepath = "/home/kek/Documents/rudens/praktika/prof_praktika/network/irasai/L_RA_M4_01_10dB.wav");
void generate_denoised_files(
    std::regex reg_noisy = std::regex(R"([LR]_[A-Z]{2}_[FM]\d+_[A-Z]{2}\d{3}_a\d{4}_\d+db\.wav)"),
    const std::string base_path = "/home/kek/Documents/rudens/praktika/prof_praktika/network/irasai/TRAIN/noisy/",
    uint batch_size = 8);