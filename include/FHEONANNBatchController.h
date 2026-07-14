/*********************************************************************************************************************** 
*
* @author: Nges Brian, Njungle 
*
* MIT License
* Copyright (c) 2025 Secure, Trusted and Assured Microelectronics, Arizona State University

* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.

* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
********************************************************************************************************************/

/*******************************************************************************************************************
 * This ANN controller is used to define CNN functions used for Batched Neural Network processing
 * This is mainly used for high throughput works.
 *******************************************************************************************************************/

#ifndef FHEON_ANNBATCHCONCROLLER_H
#define FHEON_ANNBATCHCONCROLLER_H

#include <openfhe.h>
#include <thread>
// #include <cereal/types/polymorphic.hpp> // Include this header

#include "FHEONHEController.h"
#include "FHEONANNController.h"

#include "Utils.h"
#include "UtilsData.h"
#include "UtilsBatchData.h"

using namespace lbcrypto;
using namespace std;

/** secure_anncontroller defined utils */
using namespace utils;
using namespace utilsdata;
using namespace utilsbatchdata;

class FHEONANNBatchController : public FHEONANNController {

private:
    CryptoContext<DCRTPoly> context;

public:
    int num_slots = 1 << 14;
    int baseIndex = 64;

    // Constructor must forward CryptoContext to the base class
    FHEONANNBatchController( CryptoContext<DCRTPoly> ctx)
        : FHEONANNController(ctx), context(ctx) {}
    
    void setContext(CryptoContext<DCRTPoly>& in_context);
    void setNumSlots(int numSlots){
        num_slots = 1<< numSlots;
    }

    vector <int> generate_convolution_batch_rotation_positions(int batchSize, int inputWidth, int kernelWidth, int padding=0, int stride=1);
    vector <int> generate_avgpool_batch_optimized_rotation_positions(int batchSize, int inputWidth, int kernelWidth, int stride=2, 
                            bool globalPooling=false, int rotationIndex=16);
    vector <int> generate_linear_batch_rotation_positions(int batchSize, vector<int> outputSizes, vector<int> inputSizes, int rotationIndex=100);
    vector <int> generate_batch_inputs_converter_rotation_positions(int batchSize, int inputChannels, int inputWidth, int baseIndex=64);
    vector <int> generate_normalization_rotation_positions(int inputSize);

    vector<Ctext>   he_batch_convolution(FHEONHEController &fheonHEController, vector<Ctext>& encryptedInputs, vector<vector<vector<vector<vector<double>>>>>& rawKernelData, vector<vector<double>>& rawBiasData,
                            int batchSize, int inputWidth, int inputChannels, int outputChannels, int kernelWidth, int padding=0, int stride=1);
    vector<Ctext>   he_batch_convolution_optimized(vector<Ctext>& encryptedInputs, vector<vector<vector<Ptext>>>& kernelData, vector<Ptext>& biasInputs,
                            int batchSize, int inputWidth, int inputChannels, int outputChannels, int stride=1);
    vector<Ctext>   he_batch_convolution_shortcut_optimized(vector<Ctext>& encryptedInputs, vector<vector<Ptext>>& kernelData, vector<Ptext>& biasInputs,
                            int batchSize, int inputWidth, int inputChannels, int outputChannels, int stride);


    vector<Ctext>  he_batch_convolution_optimized(FHEONHEController &fheonHEController, vector<Ctext>& encryptedInputs, vector<vector<vector<vector<vector<double>>>>>& rawKernelData, vector<vector<double>>& rawBiasData,
                            int batchSize, int inputWidth, int inputChannels, int outputChannels, int stride);
    vector<Ctext>  he_batch_convolution_shortcut_optimized(FHEONHEController &fheonHEController, vector<Ctext>& encryptedInputs, vector<vector<vector<double>>>& rawKernelData, vector<vector<double>>& rawBiasData,
                            int batchSize, int inputWidth, int inputChannels, int outputChannels, int stride);

    vector<Ctext> he_optimzed_batch_avgPool(vector<Ctext>& encryptedInputs,  int batchSize, int inputWidth, int inputChannels, int kernelWidth, int stride=2);
    vector<Ctext> he_batch_globalpool(vector<Ctext>& encryptedInputs, int batchSize, int inputWidth, int inputChannels, int kernelWidth, int rotatePositions);

    Ctext he_batch_linear(Ctext& encryptedInput, vector<Ptext>& weightMatrix, Ptext& baisInput, int batchSize, int inputSize, int outputSize, int rotatePositions=100);
    Ctext he_batch_linear_memory_efficient(Ctext& encryptedInput, vector<Ptext>& weightMatrix, Ptext& baisInput, int batchSize, int inputSize, int outputSize, int rotatePositions=100);
    vector<Ctext> he_batch_linear_multiple_outputs(Ctext& encryptedInput, vector<Ptext>& weightMatrix, Ptext& baisInput, int batchSize, int inputSize, int outputSize);
    vector<Ctext> he_batch_relu(vector<Ctext>& encryptedInputs, vector<int> scaleValues, int inputChannels, int vectorSize, int polyDegree=59); 
    Ctext he_batch_inputs_converter(vector<Ctext>& encryptedInputs, int batchSize, int inputChannels, int inputWidth, int baseIndex=64);
    Ctext he_batch_inputs_converter_memory_efficient(vector<Ctext>& encryptedInputs, int batchSize, int inputChannels, int inputWidth, int baseIndex=64);
    vector<Ctext> he_batch_sum_ciphertexts(vector<Ctext>& firstEncryptedInputs, vector<Ctext>& secondEncryptedInputs, int inputChannels);


    vector<int> generate_backpass_convolution_rotation_positions(int inputSize, int kernelSize);
    
private:
    Ptext gen_zero_mask(int size, int level); 
    Ptext gen_row_mask_with_channels(int row, int width, int inputSize, int batchSize, int level);
    Ptext gen_channel_mask_with_zeros(int channel, int outputSize, int numChannels, int level );
    

};

#endif // FHEON_ANNBATCHCONCROLLER_H