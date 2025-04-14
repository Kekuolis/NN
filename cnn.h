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

#include "load_files.h"
#include "segment_data.h"
#include "wav.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

using namespace dynet;
class Speech_Denoising_Model {
private:
  Parameter p_W;
  Parameter p_out_bias;
  unsigned LAYERS;
  unsigned INPUT_DIM;
  uint HIDDEN_SIZE;

  // VanillaLSTMBuilder builder;
  std::vector<int> output_sound;
  bool save;

public:
  Speech_Denoising_Model(ParameterCollection &pc, unsigned LAYERS = 16,
                         unsigned INPUT_DIM = 320, uint HIDDEN_SIZE = 16) {

    this->LAYERS = LAYERS;
    this->INPUT_DIM = INPUT_DIM;
    this->HIDDEN_SIZE = HIDDEN_SIZE;

    p_W = pc.add_parameters({INPUT_DIM, HIDDEN_SIZE}); // output projection
    p_out_bias = pc.add_parameters({INPUT_DIM});
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
                                   unsigned width) const {
    return input(cg, {width}, batch);
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

    // Vectors to store the normalized inputs and the corresponding
    // normalization parameters.
    std::vector<Expression> inputs;
    std::vector<Expression> means;
    std::vector<Expression> stds;

    // Determine the sequence length (number of time steps).
    unsigned width = noisy_batch.size() / INPUT_DIM;

    for (unsigned t = 0; t < width; ++t) {
      // Extract one timestep sample.
      std::vector<real> timestep_sample(noisy_batch.begin() + t * INPUT_DIM,
                                        noisy_batch.begin() +
                                            (t + 1) * INPUT_DIM);
      // Create input expression.
      Expression x_t = input(cg, {INPUT_DIM}, timestep_sample);

      // Compute the mean (scalar) of the timestep.
      Expression mean_x = mean_elems(x_t);

      // Compute the variance and the standard deviation for x_t.
      Expression variance_x = mean_elems(square(x_t - mean_x));
      // Add a small epsilon (1e-8) for numerical stability.
      Expression std_x = sqrt(variance_x + 1e-8);

      std::vector<real> mean_vals = as_vector(cg.forward(mean_x));
      std::vector<real> std_vals = as_vector(cg.forward(std_x));
      // std::cerr << "Timestep " << t << " mean: " << mean_vals[0]
      //           << ", std: " << std_vals[0] << std::endl;

      // Normalize the current timestep.
      Expression x_norm = (x_t - mean_x) / std_x;

      // Pass the normalized input to the RNN.
            // After computing h_t from the RNN:
      Expression h_t = builder.add_input(x_norm);
      std::vector<real> h_vals = as_vector(cg.forward(h_t));
      std::cerr << "Timestep " << t << " RNN output norm: " 
                << sqrt(std::inner_product(h_vals.begin(), h_vals.end(), h_vals.begin(), 0.0))
                << std::endl;
      inputs.push_back(h_t);

      // Save the normalization parameters so they can be used for
      // denormalization.
      means.push_back(mean_x);
      stds.push_back(std_x);
      
    }
    

    // Process the clean batch: no normalization (they remain as targets).
    std::vector<Expression> targets;
    for (unsigned t = 0; t < width; ++t) {
      std::vector<real> timestep_clean(clean_batch.begin() + t * INPUT_DIM,
                                       clean_batch.begin() +
                                           (t + 1) * INPUT_DIM);
      Expression y_t = input(cg, {INPUT_DIM}, timestep_clean);
      targets.push_back(y_t);
    }

    // Obtain learnable parameters.
    Expression W = parameter(cg, p_W);
    Expression out_bias = parameter(cg, p_out_bias);

    std::vector<Expression> losses;
    for (unsigned t = 0; t < width; ++t) {
      // Instead of using a plain affine_transform on the raw input,
      // we now use it to predict a normalized output.
      Expression y_norm_pred = affine_transform({out_bias, W, inputs[t]});
      // Denormalize the prediction using the mean and std computed from the
      // noisy input.
      Expression y_pred = y_norm_pred * stds[t] + means[t];

      // Compute the squared difference between the denormalized output and the
      // clean target.
      Expression diff = y_pred - targets[t];
      Expression sq = square(diff);
      losses.push_back(sq);
    }
    // Aggregate the loss over all timesteps.
    Expression total_loss = sum_batches(sum_elems(concatenate(losses))) / (width * batch_size);
    std::cerr << "Loss for current batch: " << total_loss.value() << std::endl;
    return total_loss;
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

    // Process training mini-batches.
    // 4 loops
    // one for segments
    // two for getting noisy segments into train function
    // three for iterating to segments into file size params

    for (size_t seg_start = 0; seg_start < num_clean_segments;
         seg_start += batch_size) {
      ComputationGraph cg;
      std::vector<std::vector<real>> noisy_batch;
      std::vector<real> clean_batch;

      size_t seg_end = std::min(seg_start + batch_size, num_clean_segments);

      // Collect noisy data for the batch.// here we get 8*8 segments but in the
      // right order
      int i = 0;
      for (size_t seg = seg_start; i < batch_size;
           seg += dataSegmentsNoisy[seg_start].file_segment_count) {
        // here we get first 8 segments
        std::vector<real> tmp;
        for (int j = 0; j < batch_size; j++) {
          tmp.insert(tmp.end(), dataSegmentsNoisy[seg].sound.begin(),
                     dataSegmentsNoisy[seg].sound.end());
        }
        noisy_batch.push_back(tmp);
        i++;
      }

      // get batch size amount of segments
      for (size_t seg = seg_start; seg < seg_end; ++seg) {
        clean_batch.insert(clean_batch.end(),
                           dataSegmentsClean[seg].sound.begin(),
                           dataSegmentsClean[seg].sound.end());
      }

      // Build the computation graph for the batch and compute loss.
      for (int j = 0; j < batch_size; j++) {
        // Ensure both batches have the same size.
        if (clean_batch.size() > noisy_batch[j].size()) {
          clean_batch.resize(noisy_batch[j].size());
        } else if (noisy_batch[j].size() > clean_batch.size()) {
          noisy_batch[j].resize(clean_batch.size());
        }

        Expression loss_expr =
            buildGraph(cg, noisy_batch[j], clean_batch, batch_size, builder);
        loss = as_scalar(cg.forward(loss_expr));
        cg.backward(loss_expr);
        trainer.update();
      }
    }

    std::cout << "HIDDEN SIZE " << HIDDEN_SIZE << std::endl;
    std::cout << "LAYERS " << LAYERS << std::endl;
    std::cout << "INPUT DIM " << INPUT_DIM << std::endl;
    std::cout << "Final Loss: " << loss << std::endl;
  }
  Expression load_model(ComputationGraph &cg, SimpleRNNBuilder &builder,
                        std::vector<real> &noisy_batch, uint batch_size) const {
    Expression W = parameter(cg, p_W);
    Expression out_bias = parameter(cg, p_out_bias);

    builder.new_graph(cg);
    builder.start_new_sequence();

    if (noisy_batch.size() < INPUT_DIM) {
      noisy_batch.resize(INPUT_DIM, 0);
    }

    unsigned width = noisy_batch.size() / INPUT_DIM;
    unsigned remainder = noisy_batch.size() % INPUT_DIM;
    if (remainder != 0) {
      noisy_batch.resize(noisy_batch.size() + (INPUT_DIM - remainder), 0);
      width = noisy_batch.size() / INPUT_DIM;
    }

    std::vector<Expression> predictions;
    for (unsigned t = 0; t < width; ++t) {
      std::vector<real> timestep(noisy_batch.begin() + t * INPUT_DIM,
                                 noisy_batch.begin() + (t + 1) * INPUT_DIM);
      Expression x_t = input(cg, {INPUT_DIM}, timestep);

      // === Normalize x_t ===
      Expression mean_x = mean_elems(x_t); // Compute mean (scalar)

      // Compute variance (mean of squared differences)
      Expression variance_x = mean_elems(square(x_t - mean_x));

      // Compute standard deviation (sqrt of variance)
      Expression std_x =
          sqrt(variance_x + 1e-8); // Add small epsilon for stability

      // Normalize the input (x_t - mean) / std_dev
      Expression x_norm = (x_t - mean_x) / std_x;

      // === RNN input ===
      Expression h_t = builder.add_input(x_norm);

      // === Network prediction (still in normalized space) ===
      Expression y_norm_pred = affine_transform({out_bias, W, h_t});

      // === Denormalize prediction ===
      Expression y_t = y_norm_pred * std_x + mean_x;

      predictions.push_back(y_t);
    }

    if (predictions.empty()) {
      throw std::runtime_error(
          "No predictions were produced. Check your input data.");
    }

    return concatenate(predictions);
  }

  soundData use_model(Parameter &p_W, Parameter &p_out_bias,
                      const std::string &processed_file, uint &batch_size) {
    soundData data_sound_noisy = readWav(processed_file);
    std::vector<soundData> segments_noisy = segment_data(data_sound_noisy);

    ParameterCollection builder_pc;
    this->p_W = p_W;
    this->p_out_bias = p_out_bias;

    std::cout << "HIDDEN SIZE " << HIDDEN_SIZE << std::endl;
    std::cout << "LAYERS " << LAYERS << std::endl;
    std::cout << "INPUT DIM " << INPUT_DIM << std::endl;

    if (segments_noisy.empty()) {
      throw std::runtime_error("No segments found in the processed file.");
    }

    soundData output_file;
    output_file.headerData = segments_noisy[0].headerData;

    SimpleRNNBuilder builder(LAYERS, INPUT_DIM, HIDDEN_SIZE, builder_pc);
    builder.disable_dropout();
    TextFileLoader loader("/home/kek/Documents/rudens/praktika/prof_praktika/"
                          "network/param/params.model");
    loader.populate(builder_pc, "/simple-rnn-builder/");

    for (const auto &segment : segments_noisy) {
      ComputationGraph cg;
      std::vector<real> input_vector = vecToReal(segment.monoSound);
      Expression output_expr =
          load_model(cg, builder, input_vector, batch_size);
      std::vector<real> output_vector = as_vector(cg.forward(output_expr));
      output_file.monoSound.insert(output_file.monoSound.end(),
                                   output_vector.begin(), output_vector.end());
    }

    return output_file;
  }
};