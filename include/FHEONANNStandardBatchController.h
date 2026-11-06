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

#ifndef FHEON_ANNSTANDARDBATCHCONTROLLER_H
#define FHEON_ANNSTANDARDBATCHCONTROLLER_H

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

/**
 * @brief Rotation split base of a model written for one slot count, for a context holding slotScale times more slots
 *        (FHEONHEController::slot_scale). A shift s is done as a rotation by a multiple of the split followed by one by the
 *        remainder, so any split is correct; the keys of both parts are about sqrt(largest shift) when the split is close to
 *        that square root, and the largest shifts grow with the batch: the split grows with sqrt(slotScale).
 */
inline int scale_rotation_split(int split, int slotScale) {
    return (slotScale > 1) ? static_cast<int>(std::lround(split * std::sqrt(static_cast<double>(slotScale)))) : split;
}

/**
 * @brief Batch (high-throughput) ANN controller: one channel of a batch of images per ciphertext.
 *
 * Layout: [image 0 feature map][image 1 feature map]... in every channel ciphertext. Convolutions (3x3 "same", strided, 1x1
 * shortcut, with the kernels encoded inside the primitive at the level of the inputs), average and global average pooling,
 * linear layers (one ciphertext, or one per output), batch ReLU, conversion of per-image inputs to the batch layout, and the
 * rotation-key generator of every primitive (exactly the rotations it uses). Sequential: FHEONANNStandardBatchThreadedController
 * evaluates the same primitives with worker threads.
 */
class FHEONANNStandardBatchController : public FHEONANNController {

protected:
    CryptoContext<DCRTPoly> context;

public:
    int num_slots = 1 << 14;
    int baseIndex = 64;

    // Constructor must forward CryptoContext to the base class
    FHEONANNStandardBatchController(CryptoContext<DCRTPoly> ctx)
        : FHEONANNController(ctx), context(ctx) {}
    
    void setContext(CryptoContext<DCRTPoly>& in_context);
    void setNumSlots(int numSlots){
        num_slots = 1<< numSlots;
    }

    vector <int> generate_convolution_batch_rotation_positions(int batchSize, int inputWidth, int kernelWidth, int padding=0, int stride=1);
    vector <int> generate_avgpool_batch_optimized_rotation_positions(int batchSize, int inputWidth, int kernelWidth, int stride=2, 
                            bool globalPooling=false, int rotationIndex=16);
    vector <int> generate_linear_batch_rotation_positions(int batchSize, vector<int> outputSizes, vector<int> inputSizes, int rotationIndex=100, bool batchPlacement=true);
    /** rotations of EvalSum (he_batch_globalpool, he_batch_linear): replace the full-batch EvalSum keys */
    vector <int> generate_sum_rotation_positions(int sumSize);
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

    /************************************ pipeline linear layers (one channel per ciphertext) ************************************
     * Linear layer on "packed" batch ciphertexts, for pipelines whose ciphertexts hold one channel (or feature) of many images:
     * the slots are blocks of blockSize slots, one per image; input ciphertext p holds inputs p * inputsPerBlock + s in slot s
     * of every block, and output chunk c holds outputs c * outputsPerBlock + r in slot r of every block. With poolSize > 1
     * (inputsPerBlock = 1) every input is first summed over the poolSize slots of its block (global average pooling, the
     * 1/poolSize being in the masks). Every output position is one weighted sum of rotated inputs (constant products) times one
     * mask: two levels, outputsPerBlock plaintexts for the whole layer and none per weight, so it scales to wide layers and
     * any number of images. Output k of image i: slot i * blockSize + k % outputsPerBlock of chunk k / outputsPerBlock.
     ****************************************************************************************************************************/
    vector<int> generate_batch_linear_packed_rotation_positions(int inputsPerBlock, int outputsPerBlock, int poolSize = 1);
    vector<Ptext> encode_batch_linear_packed_masks(int blockSize, int outputsPerBlock, int numSlots, int encodeLevel, double inputScale = 1.0);
    vector<Ptext> encode_batch_linear_packed_bias(const vector<double>& biasData, int blockSize, int outputsPerBlock, int numSlots, int encodeLevel);
    vector<Ctext> he_batch_linear_packed(vector<Ctext>& encryptedInputs, const vector<vector<double>>& weightData, vector<Ptext>& biasInputs,
                            const vector<Ptext>& masks, int inputsPerBlock, int outputsPerBlock, int poolSize = 1);

protected:
    /** helpers of the primitives, also used by FHEONANNStandardBatchThreadedController */
    Ptext gen_zero_mask(int size, int level); 
    Ptext gen_row_mask_with_channels(int row, int width, int inputSize, int batchSize, int level);
    Ptext gen_channel_mask_with_zeros(int channel, int outputSize, int numChannels, int level );
    /** he_batch_linear_packed: rotated copies of one input (pooled first when poolSize > 1), and one output chunk */
    vector<Ctext> packed_linear_taps(const Ctext& encryptedInput, int poolSize, int inputsPerBlock, int outputsPerBlock);
    Ctext packed_linear_chunk(const vector<vector<Ctext>>& taps, const vector<vector<double>>& weightData, const vector<Ptext>& masks,
                            Ptext& biasInput, int chunk, int inputsPerBlock, int outputsPerBlock);


};

#endif // FHEON_ANNSTANDARDBATCHCONTROLLER_H