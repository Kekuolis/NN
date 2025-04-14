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
  // These parameters represent the output projection:
  // p_W maps from the hidden state dimension to the output dimension.
  Parameter p_W;
  Parameter p_out_bias;
  unsigned LAYERS;
  unsigned INPUT_DIM;
  uint HIDDEN_SIZE;
  std::vector<int> output_sound;
  bool save;

public:
  Speech_Denoising_Model(ParameterCollection &pc, unsigned LAYERS = 8,
                         unsigned INPUT_DIM = 320, uint HIDDEN_SIZE = 8) {

    this->LAYERS = LAYERS;
    this->INPUT_DIM = INPUT_DIM;
    this->HIDDEN_SIZE = HIDDEN_SIZE;

    // Set up output projection parameters.
    // p_W should be sized so that when multiplied by a HIDDEN_SIZE vector, it
    // gives an INPUT_DIM vector.
    p_W = pc.add_parameters({INPUT_DIM, HIDDEN_SIZE});
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
    // If the data is too short, pad with zeros so that length equals INPUT_DIM
    // * batch_size.
    while (noisy_batch.size() < INPUT_DIM * batch_size) {
      noisy_batch.push_back(0);
      clean_batch.push_back(0);
    }
    // Start a new graph sequence.
    builder.new_graph(cg);
    builder.start_new_sequence();

    // Determine the number of time steps (each time step has INPUT_DIM
    // features).
    unsigned width = noisy_batch.size() / INPUT_DIM;
    std::vector<Expression> inputs;
    for (unsigned t = 0; t < width; ++t) {
      std::vector<real> timestep_sample(noisy_batch.begin() + t * INPUT_DIM,
                                        noisy_batch.begin() +
                                            (t + 1) * INPUT_DIM);
      // Create an input Expression for the current time step.
      Expression x_t =
          input(cg, {INPUT_DIM}, timestep_sample); // shape: {INPUT_DIM}
      // Get the hidden state at this time step from the RNN.
      Expression h_t = builder.add_input(x_t);
      inputs.push_back(h_t);
    }

    // Collect target expressions for each time step.
    std::vector<Expression> targets;
    for (unsigned t = 0; t < width; ++t) {
      std::vector<real> timestep_clean(clean_batch.begin() + t * INPUT_DIM,
                                       clean_batch.begin() +
                                           (t + 1) * INPUT_DIM);
      Expression y_t = input(cg, {INPUT_DIM}, timestep_clean);
      targets.push_back(y_t);
    }

    // Retrieve the learned output projection parameters.
    Expression W = parameter(cg, p_W);
    Expression out_bias = parameter(cg, p_out_bias);

    std::vector<Expression> losses;
    // For each time step, apply an affine transformation to project the hidden
    // state to the target space. This is where the model can "scale" the hidden
    // states (which may be in a small range) to values that match your
    // non-normalized targets.
    for (unsigned t = 0; t < width; ++t) {
      // Compute the affine transformation:
      // y_pred = W * h_t + out_bias
      Expression y_pred = affine_transform({out_bias, W, inputs[t]});
      // Compute the squared error loss between the prediction and the target.
      Expression diff = y_pred - targets[t];
      Expression sq = square(diff);
      losses.push_back(sq);
    }
    // Aggregate the losses over all time steps and average over batch_size.
    Expression total_loss =
        sum_batches(sum_elems(concatenate(losses))) / (width * batch_size);
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
    for (size_t seg_start = 0; seg_start < num_clean_segments;
         seg_start += batch_size) {
      ComputationGraph cg;
      std::vector<std::vector<real>> noisy_batch;
      std::vector<real> clean_batch;

      size_t seg_end = std::min(seg_start + batch_size, num_clean_segments);

      // Build noisy data batches.
      int i = 0;
      for (size_t seg = seg_start; i < batch_size;
           seg += dataSegmentsNoisy[seg_start].file_segment_count) {
        std::vector<real> tmp;
        for (int j = 0; j < batch_size; j++) {
          tmp.insert(tmp.end(), dataSegmentsNoisy[seg].sound.begin(),
                     dataSegmentsNoisy[seg].sound.end());
        }
        noisy_batch.push_back(tmp);
        i++;
      }

      // Build the clean data batch by concatenating sound vectors.
      for (size_t seg = seg_start; seg < seg_end; ++seg) {
        clean_batch.insert(clean_batch.end(),
                           dataSegmentsClean[seg].sound.begin(),
                           dataSegmentsClean[seg].sound.end());
      }

      // For each mini-batch, ensure that noisy and clean data have the same
      // size before building the graph.
      for (int j = 0; j < batch_size; j++) {
        if (clean_batch.size() > noisy_batch[j].size()) {
          clean_batch.resize(noisy_batch[j].size());
        } else if (noisy_batch[j].size() > clean_batch.size()) {
          noisy_batch[j].resize(clean_batch.size());
        }

        // Build and run the graph.
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
  // Modify load_model to apply the affine transform projection.
  Expression load_model(ComputationGraph &cg, SimpleRNNBuilder &builder,
                        std::vector<real> &noisy_batch, uint batch_size) const {
    // Get the output projection parameters.
    // p_W maps from HIDDEN_SIZE to INPUT_DIM, and p_out_bias is the added bias.
    Expression W = parameter(cg, p_W);
    Expression out_bias = parameter(cg, p_out_bias);

    // Start a new computation graph and RNN sequence.
    builder.new_graph(cg);
    builder.start_new_sequence();

    // Ensure we have at least one full frame.
    if (noisy_batch.size() < INPUT_DIM) {
      noisy_batch.resize(INPUT_DIM, 0);
    }

    // Handle partial frame by padding if necessary.
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
      // Create the input expression for this time step.
      Expression x_t = input(cg, {INPUT_DIM}, timestep);
      // Add the input to the RNN to produce the hidden state.
      Expression h_t =
          builder.add_input(x_t); // hidden state vector of size HIDDEN_SIZE

      // Apply the affine transform: project from hidden space to output space.
      Expression y_t =
          affine_transform({out_bias, W, h_t}); // now y_t has shape {INPUT_DIM}

      predictions.push_back(y_t);
    }

    if (predictions.empty()) {
      throw std::runtime_error(
          "No predictions were produced. Check your input data.");
    }
    // Concatenate predictions along the time axis.
    return concatenate(predictions); // Final shape: {INPUT_DIM * width}
  }

  // Modify use_model to match the new application graph.
  soundData use_model(Parameter &p_W, Parameter &p_out_bias,
                      const std::string &processed_file, uint &batch_size) {
    // Read the processed audio file.
    soundData data_sound_noisy = readWav(processed_file);
    std::vector<soundData> segments_noisy = segment_data(data_sound_noisy);

    // Set up the parameters from external source.
    ParameterCollection builder_pc;
    this->p_W = p_W;
    this->p_out_bias = p_out_bias;

    std::cout << "HIDDEN SIZE " << HIDDEN_SIZE << std::endl;
    std::cout << "LAYERS " << LAYERS << std::endl;
    std::cout << "INPUT DIM " << INPUT_DIM << std::endl;

    if (segments_noisy.empty()) {
      throw std::runtime_error("No segments found in the processed file.");
    }

    // Prepare the output file.
    soundData output_file;
    output_file.headerData = segments_noisy[0].headerData;

    // Create an RNN builder with the same architecture.
    SimpleRNNBuilder builder(LAYERS, INPUT_DIM, HIDDEN_SIZE, builder_pc);
    builder.disable_dropout();

    // Load saved parameters into builder_pc.
    TextFileLoader loader("/home/kek/Documents/rudens/praktika/prof_praktika/"
                          "network/param/params.model");
    loader.populate(builder_pc, "/simple-rnn-builder/");

    // Process each noisy segment.
    for (const auto &segment : segments_noisy) {
      ComputationGraph cg;
      // Convert the segment to a vector of reals.
      std::vector<real> input_vector = vecToReal(segment.monoSound);
      // Build the model graph: note that load_model applies the affine
      // transform.
      Expression output_expr =
          load_model(cg, builder, input_vector, batch_size);
      // Run the forward pass to get the output.
      std::vector<real> output_vector = as_vector(cg.forward(output_expr));
      // Append the output for this segment to the output file.
      output_file.monoSound.insert(output_file.monoSound.end(),
                                   output_vector.begin(), output_vector.end());
    }

    return output_file;
  }
};