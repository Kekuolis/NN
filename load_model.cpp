#include "load_model.h"

using namespace dynet;

void load_model(std::vector<soundRealDataNoisy> data) {
  // (1) Re-create your model: add parameters with the same dimensions
  ParameterCollection model;
  model.add_parameters({6});
  Parameter a,b,c,d,e,f;

  LookupParameter l_param;
  {
    TextFileLoader loader("./param/params.model");
    a = loader.load_param(model, "/_0");
    b = loader.load_param(model, "/_1");
    c = loader.load_param(model, "/_2");
    d = loader.load_param(model, "/_3");
    e = loader.load_param(model, "/_4");
    f = loader.load_param(model, "/_5");
    f = loader.load_param(model, "/_6");
  }
  ComputationGraph cg;

  Expression A = parameter(cg, a);
  Expression B = parameter(cg, b);
  Expression C = parameter(cg, c);
  Expression D = parameter(cg, d);
  Expression E = parameter(cg, e);
  Expression F = parameter(cg, f);
  
  // std::vector<float> denoised_output;

  // for (const auto& segment : data) {
  //   ComputationGraph cg;  // New graph for each segment

  //   std::vector<float> noisy_batch = segment.noisySound;
  //   unsigned width = noisy_batch.size(); 
  //   Dim inputDim({1, width, 1}, 1);  // Single batch

  //   // Convert noisy input to a DyNet expression
  //   Expression input_expr = input(cg, inputDim, noisy_batch);

  //   // Apply trained convolution filter
  //   Expression conv_out = conv2d(input_expr, conv_filter, {1, 1}, false);

  //   // Apply trained reconstruction filter
  //   conv_out = conv2d(conv_out, recon_filter, {1, 1}, false);

  //   // Run forward pass
  //   cg.forward(conv_out);

  //   // Extract cleaned data
  //   std::vector<float> clean_segment = as_vector(conv_out.value());
  //   denoised_output.insert(denoised_output.end(), clean_segment.begin(), clean_segment.end());
  // }

  // // Output cleaned audio (or save it as needed)
  // for (float sample : denoised_output) {
  //   std::cout << sample << " ";
  // }
  // std::cout << std::endl;
}
