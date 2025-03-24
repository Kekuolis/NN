#include "wav.h"
#include "cnn.h"
#include "segment_data.h"

int main(){
    soundData data = readWav("/home/kek/Documents/rudens/praktika/prof_praktika/lite_project/sound/BabyElephantWalk60.wav");

    std::vector<soundData> segments = segment_data(data);
    learn(segments);

}