#pragma once

#include "wav.h"
#include <fftw3.h>

struct fftwData{
    fftw_complex *out;
    fftw_complex *outLeft;
    fftw_complex *outRight;
};

fftwData computeFFT(soundData data);