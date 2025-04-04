#include "wav.h"
#include "cnn.h"
#include "segment_data.h"
#include <iostream>
#include <vector>
#include "load_model.h"

int main(int argc, char** argv){
    dynet::initialize(argc, argv);
    std::string path = "/home/kek/Documents/rudens/praktika/prof_praktika/network/irasai/";

    ParameterCollection pc;
    SpeechDenoisingModel model(pc);

    std::vector<soundRealDataClean> trainingDataClean;
    std::vector<soundRealDataNoisy> trainingDataNoisy; 

    std::string cleanDataPaths[8] = {"L_RA_M4_01.wav", "L_RA_M4_02.wav", "L_RA_M5_01.wav",
         "L_RA_M5_02.wav", "R_RD_F3_01.wav", 
        "R_RD_M4_01.wav",
        "R_RD_F3_02.wav", "R_RD_F3_03.wav",
    };
    std::string noisyDataPathsPrefix[8] = {"L_RA_M4_01_", "L_RA_M4_02_",
         "L_RA_M5_01_", "L_RA_M5_02_",
         "R_RD_F3_01_", "R_RD_M4_01_",
          "R_RD_F3_02_", "R_RD_F3_03_"};

    auto start = std::chrono::high_resolution_clock::now();

    // Load and prepare all clean and noisy data
    for (int i = 0; i < sizeof(cleanDataPaths) / sizeof(cleanDataPaths[0]); i++) {  // Loop over all files
        std::string clean_path = path + cleanDataPaths[i];
        soundData dataClean = readWav(clean_path);
        std::vector<soundData> segmentsClean = segment_data(dataClean);

        for (const auto& seg : segmentsClean) {
            soundRealDataClean segCl;
            segCl.cleanSound = vecToReal<int>(seg.monoSound);
            trainingDataClean.push_back(segCl);
        }
        std::vector<soundRealDataNoisy> tmp = batch_noisy_data(noisyDataPathsPrefix[i]);
        for (soundRealDataNoisy i : tmp) {
            trainingDataNoisy.push_back(i);
        }
        std::cout<< "tmp size: " << tmp.size() << std::endl;

    }

    if (false) {
        load_model(trainingDataNoisy);

    }
    std::cout<< "trainingDataNoisy size: " << trainingDataNoisy.size() << std::endl;

    
    model.train(trainingDataNoisy, trainingDataClean, pc, 0.01, 6, 1);

    // Save the model after training.
    TextFileSaver saver("/home/kek/Documents/rudens/praktika/prof_praktika/network/param/params.model");
    saver.save(pc);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Training time: " << duration.count() << " seconds" << std::endl;
   
    return 0;
}
