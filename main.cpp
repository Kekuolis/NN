#include "wav.h"
#include "cnn.h"
#include "segment_data.h"

std::vector<real> vecToReal(std::vector<int> &input){
    std::vector<real> tmp;
    for (int i = 0; i < input.size(); i++) {
        tmp.push_back(input[i]);
    }
    return tmp;
}
std::vector<int> realToVec(std::vector<real> &input){
    std::vector<int> tmp;
    for (int i = 0; i < input.size(); i++) {
        tmp.push_back(input[i]);
    }
    return tmp;
}

int main(int argc, char** argv){
    dynet::initialize(argc, argv);

    soundData dataNoisy = readWav("/home/kek/Documents/rudens/praktika/prof_praktika/network/irasai/L_RA_M4_01_10dB.wav");
    soundData dataClean = readWav("/home/kek/Documents/rudens/praktika/prof_praktika/network/irasai/L_RA_M4_01.wav");


    std::vector<soundData> segmentsClean = segment_data(dataClean);
    std::vector<soundData> segmentsNoisy = segment_data(dataNoisy);

    ParameterCollection pc;
    SpeechDenoisingModel model(pc, dataNoisy);
    std::vector<soundRealData> training_data;


    // problem nr1 this pushes back both vectors of clean and unclean data for some reason?
    // problem nr2 how are the values remembered across the epochs?
    // problem nr3 why would i rebuild the graph every time?
    // problem nr3 why would i rebuild the graph every time?

    for (int i = 0; i < segmentsClean.size(); i++) {
        soundRealData seg;
        
        seg.cleanSound = vecToReal(segmentsClean[i].monoSound);
        seg.noisySound = vecToReal(segmentsNoisy[i].monoSound);
        training_data.push_back(seg);
    }
    model.train(training_data, pc, 1000, 0.001);
}