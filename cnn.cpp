// cmake .. -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3/ -DENABLE_CPP_EXAMPLES=ON
#include "cnn.h"

#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/io.h"
#include "dynet/model.h"


#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <vector>

using namespace dynet;

class SpeachModel {


    std::vector<dynet::real> realInput;
    std::vector<dynet::real> features;

    void extractFeatures(const soundData& segment) {        
        dynet::real mean_mono = std::accumulate(segment.monoSound.begin(), segment.monoSound.end(), 0.0) / segment.monoSound.size();
        features.push_back(mean_mono);
        
        dynet::real mean_left = std::accumulate(segment.stereoLeft.begin(), segment.stereoLeft.end(), 0.0) / segment.stereoLeft.size();
        features.push_back(mean_left);
        
        dynet::real mean_right = std::accumulate(segment.stereoRight.begin(), segment.stereoRight.end(), 0.0) / segment.stereoRight.size();
        features.push_back(mean_right);
        
       
    }
    
    void vecToReal(std::vector<int> &input){
        for (int i = 0; i < input.size(); i++) {
            realInput.push_back(input[i]);
        }
    }
    
    void train() {
        ComputationGraph cg;
    
    }

    void learn(std::vector<soundData> dataSegments) {
        int argc = 0;
        char** argv = nullptr;
        dynet::initialize(argc, argv);

        ParameterCollection pc;
        SimpleSGDTrainer trainer(pc);
        ComputationGraph cg;
        Expression W = parameter(cg, pc.add_parameters({1, 3}));  // 3 features

        std::vector<dynet::real> x_values(3);
        Expression x = input(cg, {3}, &x_values);
        dynet::real y_value;
        Expression y = input(cg, &y_value);
        
        Expression y_pred = logistic(W * x);
        Expression l = binary_log_loss(y_pred, y);

        for (const auto& segment : dataSegments) {
            std::vector<dynet::real> features = {0.0010f};
            if (features.size() != 3) continue;  // Ensure proper feature size

            x_values = features;
            y_value = 1;

            cg.forward(l);
            cg.backward(l);
            trainer.update();
        }

        cg.print_graphviz();
    }


};

