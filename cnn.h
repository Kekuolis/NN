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
  Speech_Denoising_Model(ParameterCollection &pc, unsigned LAYERS = 8,
                         unsigned INPUT_DIM = 320, uint HIDDEN_SIZE = 8) {

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
  // Assume these are computed once and passed to the buildGraph function.
  real global_noisy_mean;
  real global_noisy_std;
  real global_clean_mean;
  real global_clean_std;
  Expression buildGraph(ComputationGraph &cg, std::vector<real> &noisy_batch,
                        std::vector<real> &clean_batch, unsigned batch_size,
                        SimpleRNNBuilder &builder) const {

    // Pad the shorter batch with zeros so they match in size.
    if (noisy_batch.size() != clean_batch.size()) {
      size_t max_size = std::max(noisy_batch.size(), clean_batch.size());
      if (noisy_batch.size() < max_size) {
        noisy_batch.resize(max_size, 0.0f); // Pad with zeros
      }
      if (clean_batch.size() < max_size) {
        clean_batch.resize(max_size, 0.0f); // Pad with zeros
      }
    }

    // Ensure total input size is at least INPUT_DIM * batch_size
    while (noisy_batch.size() < INPUT_DIM * batch_size) {
      noisy_batch.push_back(0.0f);
      clean_batch.push_back(0.0f);
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
      // std::cerr << "Timestep " << t << " RNN output norm: "
      //           << sqrt(std::inner_product(h_vals.begin(), h_vals.end(),
      //           h_vals.begin(), 0.0))
      //           << std::endl;
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
    Expression total_loss =
        sum_batches(sum_elems(concatenate(losses))) / (width * batch_size);
    // std::cerr << "Loss for current batch: " << total_loss.value() <<
    // std::endl;
    return total_loss;
  }

  // Train the model over multiple epochs using the provided training segments.
  void train(const std::vector<SoundRealDataNoisy> &dataNoisy,
             const std::vector<SoundRealDataClean> &dataClean,
             ParameterCollection &pc, float learning_rate = 0.01,
             unsigned batch_size = 8, int noisy_data_file_count = 8) {
    SimpleRNNBuilder builder(LAYERS, INPUT_DIM, HIDDEN_SIZE, pc);
    builder.disable_dropout();

    if (dataNoisy.empty() || dataClean.empty()) {
      std::cerr << "Training data is empty." << std::endl;
      return;
    }
    SimpleSGDTrainer trainer(pc, learning_rate);

    size_t num_clean = dataClean.size();
    float loss = 0.0f;

    std::cout << "Total clean files: " << num_clean << std::endl;
    std::cout << "Total noisy files: " << dataNoisy.size() << std::endl;
    // Process the clean files in batches.
    for (size_t batch_start = 0; batch_start < dataClean.size();
         batch_start += batch_size) {
      // Determine the clean file indices for this batch.
      size_t batch_end = std::min(batch_start + batch_size, dataClean.size());

      // Containers to hold the batch segments.
      // clean_segments_batch will have one segment (vector<real>) per clean
      // file in the batch.
      std::vector<std::vector<real>> clean_segments_batch;
      // noisy_segments_batch will be a vector (for the batch)
      // each entry is a vector of 8 noisy segments (each is a vector<real>)
      // corresponding to one clean file.
      std::vector<std::vector<std::vector<real>>> noisy_segments_batch;

      // Collect segmentation data for each clean file in the batch.
      for (size_t idx = batch_start; idx < batch_end; ++idx) {
        // Segment the clean data.
        auto clean_segments = segment_data(dataClean[idx].sound_data);
        if (clean_segments.empty()) {
          std::cerr << "Warning: no segmentation found for clean file at index "
                    << idx << std::endl;
          continue;
        }
        // (Select one segment; for instance, the first.)
        clean_segments_batch.push_back(vecToReal(clean_segments[0].monoSound));

        // Collect corresponding noisy segments.
        std::vector<std::vector<real>> current_noisy_set;
        int base = idx * noisy_data_file_count;
        if (base + noisy_data_file_count > dataNoisy.size()) {
          std::cerr << "Warning: not enough noisy files for clean index " << idx
                    << std::endl;
          continue;
        }
        for (int n = 0; n < noisy_data_file_count; n++) {
          auto noisy_segments = segment_data(dataNoisy[base + n].sound_data);
          if (noisy_segments.empty()) {
            std::cerr
                << "Warning: no segmentation found for noisy file at index "
                << (base + n) << std::endl;
            continue;
          }
          // (Again, we select the first segment; adjust if needed.)
          current_noisy_set.push_back(vecToReal(noisy_segments[0].monoSound));
        }
        noisy_segments_batch.push_back(current_noisy_set);
      }

      // If the batch turned out empty, skip it.
      if (clean_segments_batch.empty() || noisy_segments_batch.empty())
        continue;

      // We now iterate over the 8 corresponding noisy segment slots.
      // For each slot (k=0..noisy_data_file_count-1) we create a mini-batch.
      for (int k = 0; k < noisy_data_file_count; ++k) {
        // Flatten the k-th noisy segment from every clean file in this batch
        // into a single vector.
        std::vector<real> batch_noisy_flat;
        // And flatten the corresponding clean segments (one per clean file).
        std::vector<real> batch_clean_flat;
        for (size_t i = 0; i < noisy_segments_batch.size(); ++i) {
          // Sanity-check: we expect each noisy_segments_batch[i] to have
          // exactly noisy_data_file_count entries.
          if (k < noisy_segments_batch[i].size()) {
            // Append all values from the k-th noisy segment.
            batch_noisy_flat.insert(batch_noisy_flat.end(),
                                    noisy_segments_batch[i][k].begin(),
                                    noisy_segments_batch[i][k].end());
            // Append all values from the clean segment.
            batch_clean_flat.insert(batch_clean_flat.end(),
                                    clean_segments_batch[i].begin(),
                                    clean_segments_batch[i].end());
          }
        }

        // At this point, each flattened vector should be of length:
        // INPUT_DIM * (# samples in this mini-batch)
        // The buildGraph function pads if needed, but ideally each segment is
        // exactly INPUT_DIM.
        if (batch_noisy_flat.empty() || batch_clean_flat.empty()) {
          std::cerr << "Warning: empty flattened batch at noisy index " << k
                    << std::endl;
          continue;
        }

        // Build a new computation graph for the mini-batch.
        ComputationGraph cg;
        // Use the mini-batch size (number of clean samples in the batch) when
        // calling buildGraph.
        Expression loss_expr = buildGraph(
            cg, batch_noisy_flat, batch_clean_flat,
            static_cast<unsigned>(noisy_segments_batch.size()), builder);

        // Forward, backward and update.
        float batch_loss = as_scalar(cg.forward(loss_expr));
        cg.backward(loss_expr);
        trainer.update();
        loss += batch_loss;
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