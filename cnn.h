#pragma once
#include "dynet/rnn-state-machine.h"

#include <chrono>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "dynet/tensor.h"
#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/io.h"
#include "dynet/model.h"
#include "dynet/rnn.h"
#include "dynet/training.h"
#include <dynet/dict.h>
#include <dynet/expr.h>
#include <dynet/lstm.h>


#include "wav.h"
#include "segment_data.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

using namespace dynet;


class Speech_Denoising_Model {
public:
  Speech_Denoising_Model(ParameterCollection &pc, unsigned filter_width = 3,
                       unsigned out_channels = 16, uint HIDDEN_SIZE = 8) {
    // Pass in header data
    // Define p_conv with dimensions: {filter height, filter width, in_channels,
    // out_channels} Our input has height=1, width=filter_width, 1 input
    // channel.
    p_conv = pc.add_parameters({1, filter_width, 1, out_channels});

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
  Expression createInputExpression(ComputationGraph &cg,
                                   const std::vector<real> &batch,
                                   unsigned width, unsigned batch_size) {
    Dim inputDim({1, width, 1}, batch_size);
    return input(cg, inputDim, batch);
  }

  Expression applyConvolution(ComputationGraph &cg, Expression input_expr,
                              Parameter &filter) {
    Expression conv_filter = parameter(cg, filter);
    return conv2d(input_expr, conv_filter, {1, 1}, false);
  }

  Expression computeMSE(Expression predicted, Expression target, unsigned width,
                        unsigned batch_size) {
    Expression diff = predicted - target;
    Expression sq = square(diff);
    return sum_batches(sum_elems(sq)) / (width * batch_size);
  }

  Expression buildGraph(ComputationGraph &cg, std::vector<real> &noisy_batch,
                        std::vector<real> &clean_batch, unsigned batch_size) {
    assert(noisy_batch.size() == clean_batch.size());

    unsigned width = noisy_batch.size() /
                     batch_size; // Ensure each sample has the right width

    // Batched input
    Expression input_expr =
        createInputExpression(cg, noisy_batch, width, batch_size);

    // Apply convolution layers
    Expression conv_out = applyConvolution(cg, input_expr, p_conv);
    // conv_out = applyConvolution(cg, conv_out, p_reconstruct);
    conv_out = applyConvolution(cg, conv_out, p_reconstruct);
    // conv_out = applyConvolution(cg, conv_out, p_reconstruct);

    // Batched target
    Expression target_expr =
        createInputExpression(cg, clean_batch, width, batch_size);

    // Compute loss
    return computeMSE(conv_out, target_expr, width, batch_size);
  }

  Expression load_model(ComputationGraph &cg, std::vector<real> &noisy_batch, uint batch_size) {
    unsigned width = noisy_batch.size() /
                     batch_size;


    Expression input_expr =
      createInputExpression(cg, noisy_batch, width, batch_size);

    // Apply convolution layers
    Expression conv_out = applyConvolution(cg, input_expr, p_conv);

    conv_out = applyConvolution(cg, conv_out, p_reconstruct);

    return conv_out;
  }

  // Train the model over multiple epochs using provided training segments.
  // dataSegmentsNoisy is a vector of vectors of noisy segments that have a
  // vector of data each 6 segments has a corresponding clean segment
  void train(std::vector<soundRealDataNoisy> dataSegmentsNoisy,
             std::vector<soundRealDataClean> dataSegmentsClean,
             ParameterCollection &pc, float learning_rate = 0.01,
             unsigned batch_size = 8, int noisy_data_file_count = 6) {
    
    SimpleSGDTrainer trainer(pc, learning_rate);
    auto start = std::chrono::high_resolution_clock::now();
    
    int num_segments = dataSegmentsNoisy.size();
    int num_clean_segments = dataSegmentsClean.size();
    int num_segment_iteration = 0;
    float loss;
    
    std::cout<< "Noisy batch size: " << num_segments <<std::endl;
    std::cout<< "Noisy batch size trimmed: " << num_segments / noisy_data_file_count<<std::endl;
    std::cout<< "Clean batch size: " << num_clean_segments << std::endl;

    std::cout<< "Single Noisy batch size: " << dataSegmentsNoisy[num_segments].noisy_sound.size() <<std::endl;
    std::cout<< "Single Clean batch size: " << dataSegmentsClean[num_clean_segments-1].clean_sound.size() <<std::endl;

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
                           dataSegmentsNoisy[seg].noisy_sound.begin(),
                           dataSegmentsNoisy[seg].noisy_sound.end());
        current_batch_size++;
      }

      // Now, collect corresponding clean data
      for (unsigned seg = seg_start; seg < seg_end; seg++) {
        unsigned clean_index =
            seg / noisy_data_file_count; // Map noisy segment to correct clean segment

        if (clean_index >= num_clean_segments) {
          std::cerr << "Error: clean_index " << clean_index << " out of range!"
                    << std::endl;
          continue; // Skip if out of bounds
        }

        clean_batch.insert(clean_batch.end(),
                           dataSegmentsClean[clean_index].clean_sound.begin(),
                           dataSegmentsClean[clean_index].clean_sound.end());
      }


      // std::cout << "seg_start: " << seg_start << std::endl;
      // std::cout << "current_batch_size: " << noisy_batch.size() << std::endl;
      // std::cout << "clean_batch_size: " << clean_batch.size() << std::endl;

      
      // Ensure both batches have the same size
      if (clean_batch.size() > noisy_batch.size()) {
        clean_batch.resize(noisy_batch.size());
      } else if (noisy_batch.size() > clean_batch.size()) {
        noisy_batch.resize(clean_batch.size());
      }
      
      // Build computation graph for the batch and compute loss
      Expression loss_expr =
      buildGraph(cg, noisy_batch, clean_batch, current_batch_size);
      loss = as_scalar(cg.forward(loss_expr));
      cg.backward(loss_expr);
      trainer.update();

    }
    // std::cout << "Loss: " << loss << " Iteration: " << seg_start << std::endl;
    std::cout << "Loss: " << loss << std::endl;
    // for (auto p : pc.parameters_list()) {
    //   std::cout << "Parameter: " << p << " Dimensions: " << p->dim << std::endl;
    // }
  }


  soundData use_model(ParameterCollection &pc, std::vector<Parameter> &parameters, std::string processed_file, uint batch_size) {
    std::vector<soundRealDataNoisy> data_noisy;
    soundData data_sound_noisy = readWav(processed_file);
    std::vector<soundData> segments_noisy = segment_data(data_sound_noisy);
    soundData outputFile;
    outputFile.headerData = segments_noisy[0].headerData;
    for (soundData i : segments_noisy) {
      ComputationGraph cg;
      std::vector<real> tmp = vecToReal(i.monoSound);
      Expression output_batch = load_model(cg, tmp, batch_size);
      int lol = 0;
      for (auto j : output_batch.value().batch_elems()) {
        std::vector<real> tmp = as_vector(j);
        for (auto k : tmp) {
          outputFile.monoSound.push_back(k);
        }
      }
    }
    return outputFile;
  }

private:
  Parameter p_conv;
  Parameter p_reconstruct;
  VanillaLSTMBuilder builder;
  std::vector<int> outputSound;
  bool save;
};