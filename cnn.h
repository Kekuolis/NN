#pragma once
#include "dynet/rnn-state-machine.h"

#include <chrono>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/io.h"
#include "dynet/model.h"
#include "dynet/rnn.h"
#include "dynet/tensor.h"
#include "dynet/training.h"
#include <dynet/dict.h>
#include <dynet/expr.h>
#include <dynet/lstm.h>

#include "segment_data.h"
#include "wav.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

using namespace dynet;
class Speech_Denoising_Model {
public:
  Speech_Denoising_Model(ParameterCollection &pc, unsigned filter_width = 3,
                         unsigned out_channels = 16, uint HIDDEN_SIZE = 8) {
    // Initialize parameters with given dimensions.
    p_conv = pc.add_parameters({1, filter_width, 1, out_channels});
    p_reconstruct = pc.add_parameters({1, 1, out_channels, 1});
  }

  // Converts vector of real to vector of int.
  std::vector<int> realToVec(const std::vector<real> &input) const {
    std::vector<int> tmp;
    tmp.reserve(input.size());
    for (const auto &val : input) {
      tmp.push_back(static_cast<int>(val));
    }
    return tmp;
  }

  // Build an input expression for the computation graph.
  Expression createInputExpression(ComputationGraph &cg,
                                   const std::vector<real> &batch,
                                   unsigned width, unsigned batch_size) const {
    Dim inputDim({1, width, 1}, batch_size);
    return input(cg, inputDim, batch);
  }

  // Apply convolution operation using a given filter.
  Expression applyConvolution(ComputationGraph &cg, Expression input_expr,
                              const Parameter &filter) const {
    Expression conv_filter = parameter(cg, filter);
    return conv2d(input_expr, conv_filter, {1, 1}, false);
  }

  // Compute mean squared error between predicted and target expressions.
  Expression computeMSE(Expression predicted, Expression target, unsigned width,
                        unsigned batch_size) const {
    Expression diff = predicted - target;
    Expression sq = square(diff);
    return sum_batches(sum_elems(sq)) / (width * batch_size);
  }

  // Build the computation graph for a training instance.
  Expression buildGraph(ComputationGraph &cg, std::vector<real> &noisy_batch,
                        std::vector<real> &clean_batch,
                        unsigned batch_size) const {
    if (noisy_batch.size() != clean_batch.size()) {
      throw std::runtime_error(
          "Noisy and clean batches must have the same size.");
    }
    unsigned width = static_cast<unsigned>(noisy_batch.size() / batch_size);

    Expression input_expr =
        createInputExpression(cg, noisy_batch, width, batch_size);

    // Apply convolution layers
    Expression conv_out = applyConvolution(cg, input_expr, p_conv);
    conv_out = applyConvolution(cg, conv_out, p_reconstruct);

    Expression target_expr =
        createInputExpression(cg, clean_batch, width, batch_size);

    return computeMSE(conv_out, target_expr, width, batch_size);
  }

  // Build the model for inference.
  Expression load_model(ComputationGraph &cg, std::vector<real> &noisy_batch,
                        uint batch_size) const {
    unsigned width = static_cast<unsigned>(noisy_batch.size() / batch_size);
    Expression input_expr =
        createInputExpression(cg, noisy_batch, width, batch_size);
    Expression conv_out = applyConvolution(cg, input_expr, p_conv);
    conv_out = applyConvolution(cg, conv_out, p_reconstruct);
    return conv_out;
  }

  // Train the model over multiple epochs using the provided training segments.
  void train(const std::vector<SoundRealDataNoisy> &dataSegmentsNoisy,
             const std::vector<SoundRealDataClean> &dataSegmentsClean,
             ParameterCollection &pc, float learning_rate = 0.01,
             unsigned batch_size = 8, int noisy_data_file_count = 8) {

    if (dataSegmentsNoisy.empty() || dataSegmentsClean.empty()) {
      std::cerr << "Training data is empty." << std::endl;
      return;
    }

    SimpleSGDTrainer trainer(pc, learning_rate);
    auto start = std::chrono::high_resolution_clock::now();

    size_t num_segments = dataSegmentsNoisy.size();
    size_t num_clean_segments = dataSegmentsClean.size();
    float loss = 0.0f;

    std::cout << "Noisy batch size: " << num_segments << std::endl;
    std::cout << "Noisy batch size trimmed: "
              << num_segments / static_cast<size_t>(noisy_data_file_count)
              << std::endl;
    std::cout << "Clean batch size: " << num_clean_segments << std::endl;

    if (!dataSegmentsNoisy.empty() &&
        !dataSegmentsNoisy.back().sound.empty())
      std::cout << "Single Noisy batch size: "
                << dataSegmentsNoisy.back().sound.size() << std::endl;
    if (!dataSegmentsClean.empty() &&
        !dataSegmentsClean.back().sound.empty())
      std::cout << "Single Clean batch size: "
                << dataSegmentsClean.back().sound.size() << std::endl;

    // Process training mini-batches.
    for (size_t seg_start = 0; seg_start < num_segments;
         seg_start += batch_size) {
      ComputationGraph cg;
      std::vector<real> noisy_batch;
      std::vector<real> clean_batch;
      unsigned current_batch_size = 0;

      size_t seg_end = std::min(seg_start + batch_size, num_segments);

      // Collect noisy data for the batch.
      for (size_t seg = seg_start; seg < seg_end; ++seg) {
        noisy_batch.insert(noisy_batch.end(),
                           dataSegmentsNoisy[seg].sound.begin(),
                           dataSegmentsNoisy[seg].sound.end());
        ++current_batch_size;
      }

      // Collect corresponding clean data.
      for (size_t seg = seg_start; seg < seg_end; ++seg) {
        size_t clean_index = seg / static_cast<size_t>(noisy_data_file_count);
        if (clean_index >= num_clean_segments) {
          std::cerr << "Error: clean_index " << clean_index << " out of range!"
                    << std::endl;
          continue; // Skip if out-of-bounds.
        }
        clean_batch.insert(clean_batch.end(),
                           dataSegmentsClean[clean_index].sound.begin(),
                           dataSegmentsClean[clean_index].sound.end());
      }

      // Ensure both batches have the same size.
      if (clean_batch.size() > noisy_batch.size()) {
        clean_batch.resize(noisy_batch.size());
      } else if (noisy_batch.size() > clean_batch.size()) {
        noisy_batch.resize(clean_batch.size());
      }

      // Build the computation graph for the batch and compute loss.
      Expression loss_expr =
          buildGraph(cg, noisy_batch, clean_batch, current_batch_size);
      loss = as_scalar(cg.forward(loss_expr));
      cg.backward(loss_expr);
      trainer.update();
    }
    std::cout << "Final Loss: " << loss << std::endl;
  }

  // Use the model to process an input file.
  soundData use_model(ParameterCollection &pc,
                      const std::vector<Parameter> &parameters,
                      const std::string &processed_file,
                      uint batch_size) const {
    soundData data_sound_noisy = readWav(processed_file);
    std::vector<soundData> segments_noisy = segment_data(data_sound_noisy);
    if (segments_noisy.empty()) {
      throw std::runtime_error("No segments found in the processed file.");
    }

    soundData outputFile;
    outputFile.headerData = segments_noisy[0].headerData;

    for (const auto &segment : segments_noisy) {
      ComputationGraph cg;
      std::vector<real> input_vector = vecToReal(segment.monoSound);
      Expression output_batch = load_model(cg, input_vector, batch_size);

      for (const auto &batch_elem : output_batch.value().batch_elems()) {
        std::vector<real> output_vector = as_vector(batch_elem);
        outputFile.monoSound.insert(outputFile.monoSound.end(),
                                    output_vector.begin(), output_vector.end());
      }
    }
    return outputFile;
  }

private:
  Parameter p_conv;
  Parameter p_reconstruct;
  // VanillaLSTMBuilder builder;
  std::vector<int> outputSound;
  bool save;
};