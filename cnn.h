#pragma once
#include "dynet/tensor.h"
#include "wav.h"

#include <fstream>
#include <iostream>
#include <sstream>
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

struct soundRealData {
  std::vector<real> noisySound;
  std::vector<real> cleanSound;
};

class SpeechDenoisingModel {
public:
  // Constructor that creates the convolutional parameters.
  // p_conv: convolution filter (acts on the input signal)
  // p_reconstruct: a 1x1 filter to combine the convolution features back into
  // one channel.
  SpeechDenoisingModel(ParameterCollection &pc, soundData tmp,
                       unsigned filter_width = 3, unsigned out_channels = 16) {
    // Pass in header data
    fullNoiseData = tmp;
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
  Expression buildGraph(ComputationGraph &cg,
                        const std::vector<real> &noisy_batch,
                        const std::vector<real> &clean_batch,
                        unsigned batch_size) {
    assert(noisy_batch.size() == clean_batch.size());

    unsigned width =
        noisy_batch.size() / batch_size; // Ensure each sample has equal length
    Dim inputDim({1, width, 1}, batch_size);

    // Batched input
    Expression input_expr = input(cg, inputDim, noisy_batch);

    // Apply convolution
    Expression conv_filter = parameter(cg, p_conv);
    Expression conv_out =
        conv2d(input_expr, conv_filter, {1, 1}, false); // Enable autobatching
    Expression relu_out = rectify(conv_out);

    // Reconstruction
    Expression recon_filter = parameter(cg, p_reconstruct);
    Expression output_expr = conv2d(relu_out, recon_filter, {1, 1}, false);

    // Batched target
    Expression target_expr = input(cg, inputDim, clean_batch);

    // Compute loss for the whole batch
    Expression diff = output_expr - target_expr;
    Expression sq = square(diff);
    Expression mse = sum_batches(sum_elems(sq)) / width;
    return mse;
  }

  // Train the model over multiple epochs using provided training segments.
  void train(std::vector<soundRealData> dataSegments, ParameterCollection &pc,
             unsigned epochs = 10, float learning_rate = 0.01) {
    SimpleSGDTrainer trainer(pc, learning_rate);
    unsigned batch_size = 8; // Choose an appropriate batch size
    for (unsigned i = 0; i < dataSegments.size(); i += batch_size) {
      ComputationGraph cg;

      // Create minibatches
      std::vector<real> noisy_batch, clean_batch;
      unsigned current_batch_size = 0;

      for (unsigned j = i;
           j < std::min(i + batch_size, (unsigned)dataSegments.size()); j++) {
        noisy_batch.insert(noisy_batch.end(),
                           dataSegments[j].noisySound.begin(),
                           dataSegments[j].noisySound.end());
        clean_batch.insert(clean_batch.end(),
                           dataSegments[j].cleanSound.begin(),
                           dataSegments[j].cleanSound.end());
        current_batch_size++;
      }

      // Pass batch to buildGraph()
      Expression loss_expr =
          buildGraph(cg, noisy_batch, clean_batch, current_batch_size);
      float loss = as_scalar(cg.forward(loss_expr));
      cg.backward(loss_expr);
      cg.print_graphviz();
      trainer.update();
    }

    TextFileSaver saver("/home/kek/Documents/rudens/praktika/prof_praktika/"
                        "network/paramparams.model");
    saver.save(pc);
  }

private:
  Parameter p_conv;
  Parameter p_reconstruct;
  soundData fullNoiseData;
  std::vector<int> outputSound;
  bool save;
};
