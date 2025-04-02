#pragma once
#include "dynet/tensor.h"
#include "wav.h"

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/io.h"
#include "dynet/model.h"
#include "dynet/training.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

using namespace dynet;

struct soundRealDataClean {
  std::vector<real> cleanSound;
};
struct soundRealDataNoisy {
  std::vector<real> noisySound;
};

class SpeechDenoisingModel {
public:
  SpeechDenoisingModel(ParameterCollection &pc, unsigned filter_width = 3,
                       unsigned out_channels = 16, uint HIDDEN_SIZE = 8) {
    // Pass in header data
    // Define p_conv with dimensions: {filter height, filter width, in_channels,
    // out_channels} Our input has height=1, width=filter_width, 1 input
    // channel.
    p_conv = pc.add_parameters({1, filter_width, 1, out_channels});

    p_W = pc.add_parameters({HIDDEN_SIZE, 2});
    p_b = pc.add_parameters({HIDDEN_SIZE});
    p_V = pc.add_parameters({1, HIDDEN_SIZE});
    p_a = pc.add_parameters({1});
    // p_reconstruct maps the out_channels back to a single output channel.
    p_reconstruct = pc.add_parameters({1, 1, out_channels, 1});
  }

  std::vector<int> realToVec(std::vector<real> &input) {
    std::vector<int> tmp;
    for (int i = 0; i < input.size(); i++) {
      tmp.push_back(input[i]);
    }
    return tmp;
  }

  // Build the computation graph for one training instance.
  // noisy: the input noisy audio signal.
  // clean: the target (clean) audio signal.
  Expression buildGraph(ComputationGraph &cg,
                        std::vector<real> &noisy_batch,
                        std::vector<real> &clean_batch,
                        unsigned batch_size) {
    assert(noisy_batch.size() == clean_batch.size());

    unsigned width = noisy_batch.size() / batch_size; // Ensure each sample

    Dim inputDim({1, width, 1}, batch_size);
    

    // Batched input
    Expression input_expr = input(cg, inputDim, noisy_batch);

    // Apply convolution
    Expression conv_filter = parameter(cg, p_conv);
    // Expression conv_filter = parameter(cg, p_conv);
    Expression conv_out =
        conv2d(input_expr, conv_filter, {1, 1}, false); 

    // Reconstruction
    Expression recon_filter = parameter(cg, p_reconstruct);


    Expression output_expr = conv2d(conv_out, recon_filter, {1, 1}, false);

    // Batched target
    Expression target_expr = input(cg, inputDim, clean_batch);

    // Compute loss for the whole batch
    Expression diff = output_expr - target_expr;
    Expression sq = square(diff);
    Expression mse = sum_batches(sum_elems(sq)) / (width * batch_size);
    return mse;
  }
  // Train the model over multiple epochs using provided training segments.
  // dataSegmentsNoisy is a vector of vectors of noisy segments that have a
  // vector of data each 6 segments has a corresponding clean segment
  void train(std::vector<soundRealDataNoisy> dataSegmentsNoisy,
             std::vector<soundRealDataClean> dataSegmentsClean,
             ParameterCollection &pc, float learning_rate = 0.01,
             size_t num_files = 6, unsigned batch_size = 8) {

    SimpleSGDTrainer trainer(pc, learning_rate);
    auto start = std::chrono::high_resolution_clock::now();

    int num_segments = dataSegmentsNoisy.size();
    int num_clean_segments = dataSegmentsClean.size();
    int num_segment_iteration = 0;

    // Process training in mini-batches over segment indices
    for (unsigned seg_start = 0; seg_start < num_segments;
         seg_start += batch_size) {

      ComputationGraph cg;
      std::vector<real> noisy_batch, clean_batch;
      unsigned current_batch_size = 0;

      unsigned seg_end =
          std::min(seg_start + batch_size, static_cast<unsigned>(num_segments));

      // First, collect noisy data for the batch
      for (unsigned seg = seg_start; seg < seg_end; seg++) {
        noisy_batch.insert(noisy_batch.end(),
                           dataSegmentsNoisy[seg].noisySound.begin(),
                           dataSegmentsNoisy[seg].noisySound.end());
        current_batch_size++;
      }

      // Now, collect corresponding clean data
      for (unsigned seg = seg_start; seg < seg_end; seg++) {
        unsigned clean_index =
            seg / 6; // Map noisy segment to correct clean segment

        if (clean_index >= num_clean_segments) {
          std::cerr << "Error: clean_index " << clean_index << " out of range!"
                    << std::endl;
          continue; // Skip if out of bounds
        }

        clean_batch.insert(clean_batch.end(),
                           dataSegmentsClean[clean_index].cleanSound.begin(),
                           dataSegmentsClean[clean_index].cleanSound.end());
      }

      // Ensure both batches have the same size
      if (clean_batch.size() > noisy_batch.size()) {
        clean_batch.resize(noisy_batch.size());
      } else if (noisy_batch.size() > clean_batch.size()) {
        noisy_batch.resize(clean_batch.size());
      }

      std::cout << "seg_start: " << seg_start << std::endl;
      std::cout << "current_batch_size: " << noisy_batch.size() << std::endl;
      std::cout << "clean_batch_size: " << clean_batch.size() << std::endl;

      // Build computation graph for the batch and compute loss
      Expression loss_expr =
          buildGraph(cg, noisy_batch, clean_batch, current_batch_size);
      float loss = as_scalar(cg.forward(loss_expr));
      cg.backward(loss_expr);
      trainer.update();

      std::cout << "Loss: " << loss << " Iteration: " << seg_start << std::endl;
    }
  }

private:
  Parameter p_conv;
  Parameter p_W; 
  Parameter p_b; 
  Parameter p_V; 
  Parameter p_a; 
  Parameter p_reconstruct;
  std::vector<int> outputSound;
  bool save;
};
