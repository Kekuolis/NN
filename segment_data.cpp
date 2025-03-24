#include "segment_data.h"
#include "wav.h"
#include <vector>


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
std::vector<soundData> segment_data(soundData &data) {
    soundData tmpSegment;
    std::vector<soundData> segments;
    tmpSegment.headerData = data.headerData;
    int samplesPerMS = (data.headerData.sampleRate / 1000) * 15; // 15ms of sound data from wav

    if (data.headerData.numChannels == 1) {
        for (int i = 0; i < data.monoSound.size(); i += samplesPerMS) {
            tmpSegment.monoSound.clear();  // Clear before reuse

            // Prevent out-of-bounds access
            int endIdx = std::min(i + samplesPerMS, (int)data.monoSound.size());
            tmpSegment.monoSound.assign(data.monoSound.begin() + i, data.monoSound.begin() + endIdx);

            segments.push_back(tmpSegment);
        }
    } else {
        for (int i = 0; i < data.stereoLeft.size(); i += samplesPerMS) {
            tmpSegment.stereoLeft.clear();
            tmpSegment.stereoRight.clear();

            int endIdx = std::min(i + samplesPerMS, (int)data.stereoLeft.size());
            tmpSegment.stereoLeft.assign(data.stereoLeft.begin() + i, data.stereoLeft.begin() + endIdx);
            tmpSegment.stereoRight.assign(data.stereoRight.begin() + i, data.stereoRight.begin() + endIdx);

            segments.push_back(tmpSegment);
        }
    }
    // print_segments(segments);
    return segments;
}
