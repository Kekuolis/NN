#include "segment_data.h"
#include "wav.h"
#include <vector>

template <typename T>
std::vector<T> realToVec(std::vector<dynet::real> &input){
    std::vector<int> tmp;
    for (int i = 0; i < input.size(); i++) {
        tmp.push_back(input[i]);
    }
    return tmp;
}

void print_segments(std::vector<soundData> &data) {

    for (int i = 0; i < data.size(); i++) {
        std::cout << "Segment " << i << std::endl;
        if (data[i].headerData.numChannels == 1) {
            for (int j = 0; j < data[i].monoSound.size(); j++) {
                std::cout << data[i].monoSound[j] << " ";
            }
        } else {
            for (int j = 0; j < data[i].stereoLeft.size(); j++) {
                std::cout << data[i].stereoLeft[j] << " ";
            }
            std::cout << std::endl;
            for (int j = 0; j < data[i].stereoRight.size(); j++) {
                std::cout << data[i].stereoRight[j] << " ";
            }
        }
        std::cout << std::endl;
    }

}

std::vector<soundData> segment_data(const soundData &data) {
    std::vector<soundData> segments;
    int samplesPerMS = (data.headerData.sampleRate / 1000) * 20; // 15ms of sound data from wav

    for (int i = 0; i < data.monoSound.size(); i += samplesPerMS) {
        soundData tmpSegment;  // Create a new instance
        tmpSegment.headerData = data.headerData;

        int endIdx = std::min(i + samplesPerMS, static_cast<int>(data.monoSound.size())); // Ensure we don't go out of bounds
        tmpSegment.monoSound.insert(tmpSegment.monoSound.end(), 
                                    data.monoSound.begin() + i, 
                                    data.monoSound.begin() + endIdx);
        segments.push_back(std::move(tmpSegment)); // Move instead of copy
    }
    return segments;
}

std::vector<SoundRealDataNoisy> batch_noisy_data(std::string prefix, std::string suffix) {
    std::string basePath = "/home/kek/Documents/rudens/praktika/prof_praktika/network/irasai/";
    std::vector<soundData> segementedNoisyData;
    std::vector<SoundRealDataNoisy> tmp;
    
    // std::vector<soundRealData> tmp;

    // Loop over a range of dB values, here from 10 to 20 with a step of 5.
    int j = 0; // index for segementedNoisyData 
    for (int db = 10; db <= 35; db += 5) {
        std::stringstream filePath;
        SoundRealDataNoisy segNs;
        filePath << basePath << prefix << db << suffix;
        std::string path = filePath.str();
        segementedNoisyData = segment_data(readWav(path));
        for (int i = 0; i < segementedNoisyData.size(); i++) {
            segNs.sound = vecToReal<int>(segementedNoisyData[i].monoSound);
            tmp.push_back(segNs);
        }
        j++;
        // segNs.noisySound = vecToReal(segementedNoisyData[i].monoSound);
        std::cout << path << std::endl;
    }

    return tmp;
}

// čia turi vektorių soundData, nešvarių garsų tada iš šio vektoriaus
// reikia patalpinti duomenis į soundRealData struktūrą kad turėčiau vektorius švarių ir nešvarių garsų
// bet soundRealData jau yra segemntų struktūra
// todėl reikia vektorį soundData išskalidyti į mažus soundRealData vektorius kurie turės savo vektorius duomenų