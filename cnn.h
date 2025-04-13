#pragma once
#include "dynet/rnn-state-machine.h"

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
private:
  // Parameter p_conv;
  // Parameter p_reconstruct;
  LookupParameter p_c;
  Parameter p_R;
  Parameter p_bias;
  unsigned LAYERS;
  unsigned INPUT_DIM;
  uint HIDDEN_SIZE;

  // VanillaLSTMBuilder builder;
  std::vector<int> output_sound;
  bool save;

public:
  Speech_Denoising_Model(ParameterCollection &pc, unsigned LAYERS = 8,
                         unsigned INPUT_DIM = 320, uint HIDDEN_SIZE = 8) {

    this->LAYERS = LAYERS;
    this->INPUT_DIM = INPUT_DIM;
    this->HIDDEN_SIZE = HIDDEN_SIZE;

    p_c = pc.add_lookup_parameters(1, {INPUT_DIM});
    p_R = pc.add_parameters({1, HIDDEN_SIZE});
    p_bias = pc.add_parameters({1});
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
    Dim input_dim({batch_size, width});
    return input(cg, input_dim, batch);
  }

  // Compute mean squared error between predicted and target expressions.
  Expression computeMSE(Expression predicted, Expression target, unsigned width,
                        unsigned batch_size) const {
    Expression diff = predicted - target;
    Expression sq = square(diff);
    return sum_batches(sum_elems(sq)) / (width * batch_size);
  }

  Expression buildGraph(ComputationGraph &cg, std::vector<real> &noisy_batch,
                        std::vector<real> &clean_batch, unsigned batch_size,
                        SimpleRNNBuilder &builder) const {
    // Ensure that both batches have the same size.
    if (noisy_batch.size() != clean_batch.size()) {
      throw std::runtime_error(
          "Noisy and clean batches must have the same size.");
    }
    while (noisy_batch.size() < INPUT_DIM * batch_size) {
      noisy_batch.push_back(0);
      clean_batch.push_back(0);
    }
    // Create a new computation graph and start a new RNN sequence.
    builder.new_graph(cg);
    builder.start_new_sequence();

    // Obtain the RNN parameters.
    Expression R = parameter(cg, p_R);
    Expression bias = parameter(cg, p_bias);

    // Determine the sequence length (number of time steps)
    // Here, width represents the number of time steps per batch.
    unsigned width = static_cast<unsigned>(noisy_batch.size() / batch_size);

    Dim input_dim({width});
    Expression input_expr = input(cg, input_dim, noisy_batch);

    // Also build the target expression from the clean batch data.
    Expression target_expr =
        createInputExpression(cg, clean_batch, width, batch_size);

    Expression rnn_output = builder.add_input(input_expr);


    Expression prediction = affine_transform({bias, R, rnn_output});

    // Compute the mean squared error between the prediction and the target.
    // Depending on your application you may choose to compare each time step's
    // output with the corresponding target instead.
    return computeMSE(prediction, target_expr, width, batch_size);
  }

  // Train the model over multiple epochs using the provided training segments.
  void train(const std::vector<SoundRealDataNoisy> &dataSegmentsNoisy,
             const std::vector<SoundRealDataClean> &dataSegmentsClean,
             ParameterCollection &pc, float learning_rate = 0.01,
             unsigned batch_size = 8, int noisy_data_file_count = 8) {
    SimpleRNNBuilder builder(LAYERS, INPUT_DIM, HIDDEN_SIZE, pc);
    builder.disable_dropout();

    if (dataSegmentsNoisy.empty() || dataSegmentsClean.empty()) {
      std::cerr << "Training data is empty." << std::endl;
      return;
    }

    SimpleSGDTrainer trainer(pc, learning_rate);

    size_t num_segments = dataSegmentsNoisy.size();
    size_t num_clean_segments = dataSegmentsClean.size();
    float loss = 0.0f;

    std::cout << "Noisy batch size: " << num_segments << std::endl;
    std::cout << "Noisy batch size trimmed: "
              << num_segments / static_cast<size_t>(noisy_data_file_count)
              << std::endl;
    std::cout << "Clean batch size: " << num_clean_segments << std::endl;
    // this means nothing as the batch changes
    if (!dataSegmentsNoisy.empty() && !dataSegmentsNoisy.back().sound.empty())
      std::cout << "Single Noisy batch size: "
                << dataSegmentsNoisy.back().sound.size() << std::endl;
    if (!dataSegmentsClean.empty() && !dataSegmentsClean.back().sound.empty())
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
          buildGraph(cg, noisy_batch, clean_batch, current_batch_size, builder);
      loss = as_scalar(cg.forward(loss_expr));
      cg.backward(loss_expr);
      trainer.update();

      // std::cout << "Segment Loss: " << loss << std::endl;
    }

    std::cout<<"HIDDEN SIZE "<<HIDDEN_SIZE<<std::endl;
    std::cout<<"LAYERS "<<LAYERS<<std::endl;
    std::cout<<"INPUT DIM "<<INPUT_DIM<<std::endl;
    std::cout << "Final Loss: " << loss << std::endl;
  }
  // Build the model for inference.
  Expression load_model(ComputationGraph &cg, SimpleRNNBuilder builder,
                        std::vector<real> &noisy_batch, uint batch_size) const {

    Expression R = parameter(cg, p_R);
    Expression bias = parameter(cg, p_bias);

    unsigned width = (uint)noisy_batch.size();
    // For the last incomplete batch, you might need to decide how to handle it (pad or process separately).
    if(noisy_batch.size() < 320) {
      // Pad with zeros (or another appropriate value) until current_length equals full_length.
      noisy_batch.resize(320, 0.0f);
      width = (uint)noisy_batch.size();
    }
    Dim input_dim({width});
    
    Expression input_expr = input(cg, input_dim, noisy_batch);
    
    Expression rnn_output = builder.add_input(input_expr);
    Expression prediction = affine_transform({bias, R, rnn_output});

    return rnn_output;
  }

  // Use the model to process an input file.
  soundData use_model(LookupParameter &p_c,
                      Parameter &p_R, Parameter &p_bias,
                      const std::string &processed_file, uint &batch_size) {
    soundData data_sound_noisy = readWav(processed_file);
    std::vector<soundData> segments_noisy = segment_data(data_sound_noisy);
    ParameterCollection builder_pc;
    this->p_c = p_c;
    this->p_R = p_R;
    this->p_bias = p_bias;

    std::cout<<"HIDDEN SIZE "<<HIDDEN_SIZE<<std::endl;
    std::cout<<"LAYERS "<<LAYERS<<std::endl;
    std::cout<<"INPUT DIM "<<INPUT_DIM<<std::endl;
    if (segments_noisy.empty()) {
      throw std::runtime_error("No segments found in the processed file.");
    }

    soundData output_file;
    output_file.headerData = segments_noisy[0].headerData;
    
    SimpleRNNBuilder builder(LAYERS, INPUT_DIM, HIDDEN_SIZE,builder_pc);

    TextFileLoader loader("/home/kek/Documents/rudens/praktika/prof_praktika/"
      "network/param/params.model");
    loader.populate(builder_pc, "/simple-rnn-builder/");

    builder.disable_dropout();
    for (const auto &segment : segments_noisy) {
      ComputationGraph cg;
      builder.new_graph(cg);
      builder.start_new_sequence();
      std::vector<real> input_vector = vecToReal(segment.monoSound);
      Expression output_batch =
          load_model(cg, builder, input_vector, batch_size);

      for (const auto &batch_elem : output_batch.value().batch_elems()) {
        std::vector<real> output_vector = as_vector(batch_elem);
        output_file.monoSound.insert(output_file.monoSound.end(),
                                     output_vector.begin(),
                                     output_vector.end());
      }
    }
    std::cout << "Output file size: " << output_file.monoSound.size()
              << std::endl;
    return output_file;
  }
};