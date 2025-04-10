#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <cstring>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <filesystem>
#include <math.h>
#include "stdlib.h"
#include <sndfile.h>
#include <SDL2/SDL.h>
#include <algorithm>
#include <complex>

struct header {
    char riffHeader[4] = {'R', 'I', 'F', 'F'}; // "RIFF"
    int wavSize;                               // Size of the WAV file
    char waveHeader[4] = {'W', 'A', 'V', 'E'}; // "WAVE"
    char fmtHeader[4] = {'f', 'm', 't', ' '};  // "fmt "
    int fmtChunkSize = 16;                     // Size of the fmt chunk
    short audioFormat = 1;                     // Audio format (1 for PCM)
    short numChannels;                          // Number of channels
    int sampleRate;                             // Sampling frequency
    int byteRate;                               // (SampleRate * NumChannels * BitsPerSample/8)
    short blockAlign;                           // (NumChannels * BitsPerSample/8)
    short bitsPerSample;                        // Bits per sample (usually 16 or 24)
    char dataHeader[4] = {'d', 'a', 't', 'a'}; // "data"
    int dataSize;                               // Size of the audio data
};

struct soundData {
    header headerData;
    std::vector<int> monoSound;
    std::vector<int> stereoLeft;
    std::vector<int> stereoRight;
};

std::string getPath();
std::string getSavePath();

soundData readWav(std::string pathToFile);

void processInput(GLFWwindow *window);
