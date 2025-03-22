#include "transform_data.h"

// one dimentional implementation
// http://fftw.org/fftw3_doc/Complex-One_002dDimensional-DFTs.html#Complex-One_002dDimensional-DFTs
fftwData computeFFT(soundData data){

    fftwData tmpData;
    if (data.headerData.numChannels == 1) {

        int N = data.monoSound.size();

        fftw_complex *in, *out;
        fftw_plan p;
        in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
        out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

        for (int i = 0; i < N; i++) {
            in[i][0] = static_cast<double>(data.monoSound[i]); // Real part
            in[i][1] = 0.0; // Imaginary part
        }
        
        p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(p);
        tmpData.out = out;
        fftw_destroy_plan(p);
        fftw_free(in); 
        fftw_free(out);
    }
    else {
     // Process Left Channel
     int NLeft = data.stereoLeft.size();
     fftw_complex *inLeft, *outLeft;
     fftw_plan pLeft;
     inLeft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NLeft);
     outLeft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NLeft);
 
     for (int i = 0; i < NLeft; i++) {
         inLeft[i][0] = static_cast<double>(data.stereoLeft[i]); // Real part
         inLeft[i][1] = 0.0; // Imaginary part
     }
 
     pLeft = fftw_plan_dft_1d(NLeft, inLeft, outLeft, FFTW_FORWARD, FFTW_ESTIMATE);
     fftw_execute(pLeft);
 
     fftw_destroy_plan(pLeft);
     fftw_free(inLeft); 
     fftw_free(outLeft);
 
     // Process Right Channel
     int NRight = data.stereoRight.size();
     fftw_complex *inRight, *outRight;
     fftw_plan pRight;
     inRight = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NRight);
     outRight = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NRight);
 
     for (int i = 0; i < NRight; i++) {
         inRight[i][0] = static_cast<double>(data.stereoRight[i]); // Real part
         inRight[i][1] = 0.0; // Imaginary part
     }
 
     pRight = fftw_plan_dft_1d(NRight, inRight, outRight, FFTW_FORWARD, FFTW_ESTIMATE);
     fftw_execute(pRight);
     tmpData.outLeft = outLeft;
     tmpData.outRight = outRight;

     fftw_destroy_plan(pRight);
     fftw_free(inRight); 
     fftw_free(outRight);
    }
    return tmpData;
}