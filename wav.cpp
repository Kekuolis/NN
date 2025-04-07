#include "wav.h"

#include <cstdint>

inline bool fileExists (const std::string& name) {
    struct stat buffer; 
    return (stat (name.c_str(), &buffer) == 0); 
}

void fromBufferToPCM(int16_t* audioPCM, const int8_t* buffer, size_t bufferSize) {
    // Ensure the buffer size is compatible with the number of int16_t values
    if (bufferSize % sizeof(int16_t) != 0) {
        throw std::invalid_argument("Buffer size must be a multiple of 2 bytes.");
    }

    size_t numSamples = bufferSize / sizeof(int16_t);
}

soundData readWav(std::string pathToFile) {
    
    header header;
    int headerSize = sizeof(header);

    if (pathToFile.empty()) 
        throw std::runtime_error("Error: No file provided.");

    if (!fileExists(pathToFile))
        throw std::runtime_error("Error: The file path is empty or not accessible. Please provide a valid file.");

    std::FILE* wavFile = std::fopen(pathToFile.c_str(), "rb");
    if (!wavFile)
        throw std::runtime_error("Unable to open wave file");

    size_t bytes = std::fread(&header, 1, headerSize, wavFile);
    if (bytes != headerSize)
        throw std::runtime_error("Failed to read the header properly.");

    // Display header information (example)
    std::cout << "Channels: " << header.numChannels << std::endl;
    std::cout << "Sample Rate: " << header.sampleRate << std::endl;
    std::cout << "Bits Per Sample: " << header.bitsPerSample << std::endl;
    std::cout << "Data size: " << header.dataSize << std::endl;

    // Buffer for reading data
    const uint16_t bufferSize = 4096;
    uint8_t buffer[bufferSize]; // Use a buffer that works for all cases

    // Determine bytes per sample (how much data each sample takes in the file)
    uint16_t bytesPerSample = header.bitsPerSample / 8;
    uint16_t numChannels = header.numChannels;

    soundData tmpStruct;

    std::vector<int> monoSound;
    std::vector<int> stereoLeft;
    std::vector<int> stereoRight;

    tmpStruct.headerData = header;
     switch (header.bitsPerSample) {
        case 8: {
            // 8-bit PCM data (unsigned)
            while ((bytes = std::fread(buffer, sizeof(uint8_t), bufferSize, wavFile)) > 0) {
                for (size_t i = 0; i < bytes; i += numChannels) {
                    if (numChannels == 1) {
                        // Mono
                        uint8_t sample = buffer[i];
                        // std::cout << "Mono 8-bit Sample: " << static_cast<int>(sample) << std::endl;
                        monoSound.push_back(sample);
                    } else if (numChannels == 2) {
                        // Stereo
                        uint8_t leftSample = buffer[i];
                        uint8_t rightSample = buffer[i + 1];
                        // std::cout << "Stereo 8-bit Left: " << static_cast<int>(leftSample)
                        //           << ", Right: " << static_cast<int>(rightSample) << std::endl;
                        stereoRight.push_back(leftSample);
                        stereoLeft.push_back(rightSample);

                    }
                }
            }
            break;
        }
        case 16: {
            // 16-bit PCM data (signed)
            int16_t audioPCM[bufferSize / 2]; // Buffer for 16-bit samples
            while ((bytes = std::fread(buffer, sizeof(uint8_t), bufferSize, wavFile)) > 0) {
                for (size_t i = 0; i < bytes / (bytesPerSample * numChannels); i++) {
                    if (numChannels == 1) {
                        // Mono
                        int16_t sample = buffer[2 * i] | (buffer[2 * i + 1] << 8);
                        // std::cout << "Mono 16-bit Sample: " << audioPCM[i] << std::endl;
                        monoSound.push_back(sample);
                    } else if (numChannels == 2) {
                        // Stereo
                        int16_t leftSample = buffer[4 * i] | (buffer[4 * i + 1] << 8);
                        int16_t rightSample = buffer[4 * i + 2] | (buffer[4 * i + 3] << 8);
                        // std::cout << "Stereo 16-bit Left: " << leftSample
                        //           << ", Right: " << rightSample << std::endl;
                        stereoRight.push_back(leftSample);
                        stereoLeft.push_back(rightSample);
                    }
                }
            }
            break;
        }
        case 24: {
            // 24-bit PCM data (signed)
            while ((bytes = std::fread(buffer, sizeof(uint8_t), bufferSize, wavFile)) > 0) {
                for (size_t i = 0; i < bytes / (3 * numChannels); i++) {
                    if (numChannels == 1) {
                        // Mono
                        int sample = (buffer[3 * i] | (buffer[3 * i + 1] << 8) | (buffer[3 * i + 2] << 16));
                        // Adjust for signed 24-bit by extending the sign
                        if (sample & 0x800000) {
                            sample |= 0xFF000000; // Sign extension for negative values
                        }
                        // std::cout << "Mono 24-bit Sample: " << sample << std::endl;
                        monoSound.push_back(sample);
                    } else if (numChannels == 2) {
                        // Stereo
                        int32_t leftSample = (buffer[6 * i] | (buffer[6 * i + 1] << 8) | (buffer[6 * i + 2] << 16));
                        if (leftSample &     0x800000) leftSample |= 0xFF000000; // Sign extension
                        int32_t rightSample = (buffer[6 * i + 3] | (buffer[6 * i + 4] << 8) | (buffer[6 * i + 5] << 16));
                        if (rightSample & 0x800000) rightSample |= 0xFF000000; // Sign extension
                        std::cout << "Stereo 24-bit Left: " << leftSample
                                  << ", Right: " << rightSample << std::endl;
                        stereoRight.push_back(leftSample);
                        stereoLeft.push_back(rightSample);
                    }
                }
            }
            break;
        }
        case 32: {
            // 32-bit PCM data (signed)
            int32_t audioPCM[bufferSize / 4]; // Buffer for 32-bit samples
            while ((bytes = std::fread(buffer, sizeof(uint8_t), bufferSize, wavFile)) > 0) {
                for (size_t i = 0; i < bytes / (4 * numChannels); i++) {
                    if (numChannels == 1) {
                        // Mono
                        int32_t sample = buffer[4 * i] | (buffer[4 * i + 1] << 8) |
                                      (buffer[4 * i + 2] << 16) | (buffer[4 * i + 3] << 24);
                        // std::cout << "Mono 32-bit Sample: " << audioPCM[i] << std::endl;
                        monoSound.push_back(sample);
                    } else if (numChannels == 2) {
                        // Stereo
                        int32_t leftSample = buffer[8 * i] | (buffer[8 * i + 1] << 8) |
                                             (buffer[8 * i + 2] << 16) | (buffer[8 * i + 3] << 24);
                        int32_t rightSample = buffer[8 * i + 4] | (buffer[8 * i + 5] << 8) |
                                              (buffer[8 * i + 6] << 16) | (buffer[8 * i + 7] << 24);
                        // std::cout << "Stereo 32-bit Left: " << leftSample
                        //           << ", Right: " << rightSample << std::endl;
                        stereoRight.push_back(leftSample);
                        stereoLeft.push_back(rightSample);
                    }
                }
            }
            break;
        }
        default:
            std::cerr << "Unsupported bit depth: " << header.bitsPerSample << std::endl;
            break;
    }
    tmpStruct.monoSound = monoSound;
    tmpStruct.stereoRight = stereoRight;
    tmpStruct.stereoLeft = stereoLeft;

    std::fclose(wavFile);

    return tmpStruct;
}
