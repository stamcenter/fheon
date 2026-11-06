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
/**
 * @brief ANN Batch Controller for managing homomorphic batched ANN operations.
 *
 * This class provides high-level functions for performing batched neural network
 * operations in the encrypted domain using homomorphic encryption (FHE). It 
 * supports operations such as:
 *   - Batched convolution layers
 *   - Batched average and global pooling
 *   - Batched fully connected (linear) layers
 *   - Batched activation functions (e.g., ReLU)
 *
 * The class handles multiple input channels and multiple batches simultaneously,
 * using optimized rotation and packing strategies to minimize the number of
 * homomorphic operations while preserving the correct layout of data.
 *
 * @note
 * - All operations assume ciphertext packing follows a contiguous layout:
 *   [batch0: channel_i feature map][batch1: channel_i feature map]...
 * - Designed for high-throughput FHE-based ANN inference pipelines.
 */

#include <fstream>
#include <filesystem>
#include <iostream>
#include <cmath>
#include "FHEONANNStandardBatchController.h"

namespace fs = std::filesystem;

void FHEONANNStandardBatchController::setContext(CryptoContext<DCRTPoly>& in_context){
    context = in_context;
}

/**
 * @brief Generate rotation positions for batched convolution operations.
 *
 * This function calculates the rotation positions required for performing
 * convolution operations on batched encrypted data. It considers the input
 * dimensions, kernel size, padding, and stride to determine the optimal
 * rotation indices for aligning data during the convolution process.
 *
 * @param batchSize     Number of batches encoded in the input ciphertexts.
 * @param inputWidth    Width (and height) of the input feature maps (assumed square).
 * @param kernelWidth   Width (and height) of the convolution kernel (assumed square).
 * @param padding       Number of zero-padding slots applied around the input feature map.
 * @param stride        Stride length used for the convolution operation.
 *
 * @return vector<int>  Vector of rotation positions required for the convolution.
 */
vector<int> FHEONANNStandardBatchController::generate_convolution_batch_rotation_positions(int batchSize, int inputWidth, int kernelWidth, int padding, int stride) {

    vector<int> keys_position;
    if (kernelWidth == 3 && padding == 1) {
        /** he_batch_convolution_optimized: the 3x3 kernel slices (-1, 1, -inputWidth, inputWidth); strided, it and
         *  he_batch_convolution_shortcut_optimized add downsample_with_multiple_channels over the batch: the binary row
         *  juxtaposition (same expressions as the downsample), the row shift and the image shift inputSize - outputSize */
        keys_position = {-1, 1, -inputWidth, inputWidth};
        if (stride > 1) {
            int width_out = inputWidth/stride;
            for (int s = 1; s < log2(width_out); s++) {
                keys_position.push_back(pow(2, s-1));
            }
            keys_position.push_back(pow(2, (log2(width_out)-1)));
            if (width_out > 1) {
                keys_position.push_back(stride*inputWidth - width_out);
            }
            if (batchSize > 1) {
                keys_position.push_back(inputWidth*inputWidth - width_out*width_out);
            }
        }
        std::sort(keys_position.begin(), keys_position.end());
        keys_position.erase(std::unique(keys_position.begin(), keys_position.end()), keys_position.end());
        keys_position.erase(std::remove(keys_position.begin(), keys_position.end(), 0), keys_position.end());
        return keys_position;
    }
    int inputWidth_sq = pow(inputWidth, 2);
    int padded_width = inputWidth+(2*padding);
    int width_out = ((padded_width - (kernelWidth - 1) - 1)/stride)+1;
    int width_out_sq = pow(width_out,2);
    keys_position.push_back(inputWidth);
    keys_position.push_back(-inputWidth);
    keys_position.push_back(-1);
    keys_position.push_back(1);



    int shift = inputWidth - width_out;
    keys_position.push_back(shift);

    shift = (inputWidth_sq - width_out_sq);
    keys_position.push_back(shift);

    for(int i=1; i < kernelWidth; i++){
        keys_position.push_back(i);
        keys_position.push_back(-i);
    }

    // for(int i=1; i<width_out; i++){
    //     shift = (i*width_out);
    //     keys_position.push_back(-shift);
    // }

    // for(int b=0; b<batchSize; b++){
    //     shift = (b*width_out_sq);
    //     keys_position.push_back(-shift);
    // }


    if(stride > 1){
        for (int s=1; s<log2(width_out); s++) {
            keys_position.push_back( pow(2, s-1));
        }
        keys_position.push_back(pow(2, log2(width_out)-1));
        int rotAmount = (stride * inputWidth - width_out);
        keys_position.push_back(rotAmount);

        int shift = (inputWidth_sq - width_out_sq)* ((batchSize / stride) - 1);
        keys_position.push_back(-shift);

        shift = -(inputWidth_sq - width_out_sq);
        keys_position.push_back(shift);

        shift = (inputWidth_sq - width_out_sq);
        keys_position.push_back(shift);

        int rotateAmount = - batchSize * width_out_sq;
        keys_position.push_back(rotateAmount);
    }

    std::sort(keys_position.begin(), keys_position.end());
    keys_position.erase(std::unique(keys_position.begin(), keys_position.end()), keys_position.end());
    keys_position.erase(std::remove(keys_position.begin(), keys_position.end(), 0), keys_position.end());
    return keys_position;
}

/**
 * @brief Generate rotation positions for optimized average pooling operations.
 *
 * This function calculates the rotation positions required for performing
 * average pooling operations on batched encrypted data. It supports both
 * global pooling and standard average pooling, optimizing the rotation
 * indices based on the input dimensions, kernel size, and stride.
 *
 * @param batchSize      Number of batches encoded in the input ciphertexts.
 * @param inputWidth     Width (and height) of the input feature maps (assumed square).
 * @param kernelWidth    Width (and height) of the pooling kernel (assumed square).
 * @param stride         Stride length used for the pooling operation.
 * @param globalPooling  Flag indicating whether global pooling is used.
 * @param rotationIndex  Index for rotation optimization.
 *
 * @return vector<int>   Vector of rotation positions required for the pooling operation.
 */
vector<int> FHEONANNStandardBatchController::generate_avgpool_batch_optimized_rotation_positions(int batchSize, int inputWidth, int kernelWidth, int stride, 
                                bool globalPooling, int rotationIndex) {
    
    vector<int> keys_position;
    if(globalPooling){
        /** he_batch_globalpool: image shifts (inputSize), merges of up to rotationIndex sums (-1 ... -(rotationIndex-1))
         *  and the shifts of the merged groups (-rotationIndex, -2*rotationIndex, ...). The EvalSum rotations come from
         *  generate_sum_rotation_positions(inputSize). */
        if (batchSize > 1) {
            keys_position.push_back(inputWidth*inputWidth);
        }
        for(int i=1; i < std::min(rotationIndex, batchSize); i++){
            keys_position.push_back(-i);
        }
        for(int pos=rotationIndex; pos<batchSize; pos+=rotationIndex){
            keys_position.push_back(-pos);
        }
        return keys_position;
    }

    int width_avgpool_out = (inputWidth/stride);
    int width_avgpool_sq = pow(width_avgpool_out, 2); 
    int width_sq = pow(inputWidth, 2);
    keys_position.push_back(inputWidth);
    keys_position.push_back(width_sq);

    for(int i=1; i < kernelWidth;i++){
        keys_position.push_back(i);
    }

    if(inputWidth<=2){
        return keys_position;
    }

    for (int s=1; s<log2(width_avgpool_out); s++) {
        keys_position.push_back( pow(2, s-1));
    }

    keys_position.push_back(pow(2, log2(width_avgpool_out)-1));

    int shift = ((stride * inputWidth) - width_avgpool_out);
    keys_position.push_back(shift);

    shift = width_sq - width_avgpool_sq;
    keys_position.push_back(shift);
    
    std::sort(keys_position.begin(), keys_position.end());
    keys_position.erase(std::unique(keys_position.begin(), keys_position.end()), keys_position.end());
    keys_position.erase(std::remove(keys_position.begin(), keys_position.end(), 0), keys_position.end());

    return keys_position;
}

/**
 * @brief Rotation positions of EvalSum(ciphertext, sumSize): the powers of two below sumSize. They replace the EvalSum
 * keys of the full batch (EvalSumKeyGen), which are the same automorphisms for all powers of two below the slots.
 *
 * @param sumSize Largest number of slots summed by EvalSum.
 * @return vector<int> Rotation positions.
 */
vector<int> FHEONANNStandardBatchController::generate_sum_rotation_positions(int sumSize) {
    vector<int> keys_position;
    for (int r = 1; r < sumSize; r *= 2) {
        keys_position.push_back(r);
    }
    return keys_position;
}

/**
 * @brief Generate rotation positions for batched fully connected layers.
 *
 * This function calculates the rotation positions required for performing
 * fully connected layer operations on batched encrypted data. It considers
 * the input and output sizes, as well as the batch size, to determine the
 * optimal rotation indices for aligning data during the operation.
 *
 * @param batchSize      Number of batches encoded in the input ciphertexts.
 * @param outputSizes    Vector of output sizes for each layer.
 * @param inputSizes     Vector of input sizes for each layer.
 * @param rotationIndex  Index for rotation optimization.
 *
 * @param batchPlacement If true (he_batch_linear), add the rotations that place the outputs of image b at b * outputSize; false for he_batch_linear_multiple_outputs, which keeps the images in separate ciphertexts.
 * @return vector<int>   Vector of rotation positions required for the fully connected layer.
 */
vector<int> FHEONANNStandardBatchController::generate_linear_batch_rotation_positions(int batchSize, vector<int> outputSizes, vector<int> inputSizes, int rotationIndex, bool batchPlacement) {
    
    /** he_batch_linear: input shifts between the images of the batch (inputSize), the merge of the outputSize sums of an
     *  image (-1 ... -(outputSize-1)) and, with batchPlacement, the placement of image b at -(b * outputSize) split into
     *  -base_idx and -mod_idx. he_batch_linear_multiple_outputs keeps the images in separate ciphertexts
     *  (batchPlacement = false). The EvalSum rotations come from generate_sum_rotation_positions(inputSize). */
    vector<int> keys_position;
    for(size_t k =0; k<inputSizes.size(); k++){
        if (batchSize > 1) {
            keys_position.push_back(inputSizes[k]);
        }
    }

    for(size_t i=0; i<outputSizes.size(); i++){
        for(int j=1; j<outputSizes[i]; j++){
            keys_position.push_back(-j);
        }
        if (!batchPlacement) {
            continue;
        }
        for(int b=1; b<batchSize; b++){
            int rot_idx = (b * outputSizes[i]); 
            int base_idx = (rot_idx / rotationIndex) * rotationIndex;
            int mod_idx = rot_idx % rotationIndex; 
            keys_position.push_back(-base_idx);
            keys_position.push_back(-mod_idx);
        }
    }
   
    std::sort(keys_position.begin(), keys_position.end());
    keys_position.erase(std::unique(keys_position.begin(), keys_position.end()), keys_position.end());
    keys_position.erase(std::remove(keys_position.begin(), keys_position.end(), 0), keys_position.end());
    return keys_position;
}

/**
 * @brief Generate rotation positions for converting batched inputs.
 *
 * This function calculates the rotation positions required for converting
 * batched inputs into the appropriate format for processing. It considers
 * the input dimensions, batch size, and number of channels to determine
 * the optimal rotation indices for aligning data.
 *
 * @param batchSize      Number of batches encoded in the input ciphertexts.
 * @param inputChannels  Number of input channels.
 *
 * @param inputWidth Width (and height) of every input channel (inputWidth^2 slots per channel and image).
 * @param baseIndex Rotation split base: a shift s is done as a rotation by (s / baseIndex) * baseIndex followed by one by s % baseIndex, so only multiples of baseIndex and residues below it need keys.
 * @return vector<int>   Vector of rotation positions required for input conversion.
 */
vector<int> FHEONANNStandardBatchController::generate_batch_inputs_converter_rotation_positions(int batchSize, int inputChannels, int inputWidth, int baseIndex) {
    
    vector<int> keys_position;
    int pixelsPerChannel = inputWidth * inputWidth;
    int batchNeuronCount = inputChannels * pixelsPerChannel;

    for (int b = 0; b < batchSize; b++) {
        for (int i = 0; i < inputChannels; i++) {
            int currentPos = b * pixelsPerChannel;
            int targetPos = b * batchNeuronCount + i * pixelsPerChannel;
            int shift = currentPos - targetPos;
            
            if (shift != 0) {
                // Handle rotation splitting based on baseIndex
                int base_idx = (abs(shift) / baseIndex) * baseIndex;
                int mod_idx = abs(shift) % baseIndex;
                int sign = (shift > 0) ? 1 : -1;
                
                if (base_idx != 0) keys_position.push_back(sign * base_idx);
                if (mod_idx != 0) keys_position.push_back(sign * mod_idx);
            }
        }
    }
    
    std::sort(keys_position.begin(), keys_position.end());
    keys_position.erase(std::unique(keys_position.begin(), keys_position.end()), keys_position.end());
    keys_position.erase(std::remove(keys_position.begin(), keys_position.end(), 0), keys_position.end());
    return keys_position;
}


vector<int> FHEONANNStandardBatchController::generate_normalization_rotation_positions(int inputSize) {
    vector<int> keys_position;
    keys_position.push_back(-1);

   for(int j=1; j<log2(inputSize); j++){
        int powVal = pow(2, j);
        keys_position.push_back(-powVal);
    }
    std::sort(keys_position.begin(), keys_position.end());
    keys_position.erase(std::unique(keys_position.begin(), keys_position.end()), keys_position.end());
    keys_position.erase(std::remove(keys_position.begin(), keys_position.end(), 0), keys_position.end());
    return keys_position;
}



vector<int> FHEONANNStandardBatchController::generate_backpass_convolution_rotation_positions(int inputSize, int kernelWidth) {
    vector<int> keys_position;
    int kernelSize = kernelWidth * kernelWidth;

   for(int j=1; j<kernelSize; j++){
        keys_position.push_back(j);
    }
    std::sort(keys_position.begin(), keys_position.end());
    keys_position.erase(std::unique(keys_position.begin(), keys_position.end()), keys_position.end());
    keys_position.erase(std::remove(keys_position.begin(), keys_position.end(), 0), keys_position.end());
    return keys_position;
}



/**
 * @brief Perform a secure convolution operation on batched encrypted data.
 *
 * This function implements a convolutional layer in the encrypted domain
 * using homomorphic encryption for batched inputs. Given a set of encrypted
 * input feature maps, convolution kernels, and bias terms, it computes the
 * convolution across multiple channels and batches while respecting the
 * specified input dimensions, kernel size, padding, and stride.
 *
 * For each output channel, the function multiplies rotated input ciphertexts
 * by corresponding kernel weights, sums contributions from all input channels,
 * applies the bias, and aligns outputs across batches using rotations and
 * packing strategies. The result is one ciphertext per output channel,
 * encoding all batches contiguously.
 *
 * @param fheonHEController  HE controller that encodes the kernels (encode_kernel) at the level of the rotated inputs.
 * @param encryptedInputs   Vector of ciphertexts, one per input channel,
 *                          each encoding batched feature maps of size
 *                          (batchSize × inputWidth × inputWidth).
 * @param rawKernelData     Kernel weights [outputChannel][inputChannel][batch][rows][cols], every kernel repeated over the
 *                          batch (UtilsBatchData::load_batch_weights); encoded inside the function.
 * @param rawBiasData       Bias values [outputChannel][batch * outputWidth^2] (UtilsBatchData::load_batch_bias).
 * @param batchSize         Number of batches encoded in the input ciphertexts.
 * @param inputWidth        Width (and height) of the input feature maps (assumed square).
 * @param inputChannels     Number of input channels.
 * @param outputChannels    Number of output channels.
 * @param kernelWidth       Width (and height) of the convolution kernel (assumed square).
 * @param padding           Number of zero-padding slots applied around the input feature map.
 * @param stride            Stride length used for the convolution operation.
 *
 * @return vector<Ctext>    Vector of ciphertexts containing the encrypted results
 *                          of the convolution layer, one ciphertext per output channel,
 *                          with batch data packed contiguously:
 *                          [batch0 output channel_i][batch1 output channel_i]...
 *
 * @note
 * - The function assumes ciphertext packing follows contiguous layout:
 *   [batch0: channel_i feature map][batch1: channel_i feature map]...
 * - Optimized rotations and multi-channel processing are used to reduce
 *   the number of homomorphic operations, improving performance for FHE-based CNNs.
 */
vector<Ctext> FHEONANNStandardBatchController::he_batch_convolution(FHEONHEController &fheonHEController, vector<Ctext>& encryptedInputs, vector<vector<vector<vector<vector<double>>>>>& rawKernelData, vector<vector<double>>& rawBiasData,
                            int batchSize, int inputWidth, int inputChannels, int outputChannels, int kernelWidth, int padding, int stride){

    
    auto start_time = startTime();
int kernelSize = kernelWidth * kernelWidth;
    inputWidth += 2 * padding;
    int inputSize = inputWidth*inputWidth;
    int outputWidth = ((inputWidth - kernelWidth) / stride) + 1;
    int outputSize = outputWidth * outputWidth;
    int inVecSize = batchSize*inputSize;
    int outVecSize = batchSize * outputSize; 
    int encode_level = encryptedInputs[0]->GetLevel();

    Ptext cleaningoutputMask = context->MakeCKKSPackedPlaintext(generate_mixed_mask(outVecSize,  inVecSize), 1,  encode_level);
    Ctext encryptedzeros =  context->EvalMult(encryptedInputs[0], gen_zero_mask(inVecSize, encode_level));

    // STEP 2 - ROTATE INPUT TO FORM k^2 slices
    vector<vector<Ctext>> bactched_rotated_ciphertexts(inputChannels);
    for(int b=0; b<inputChannels; b++){
        vector<Ctext> rotated_ciphertexts;
        for (int k=0; k<kernelWidth; k++) {
            if(k >0){
                encryptedInputs[b] = context->EvalRotate(encryptedInputs[b], inputWidth);
            }
            rotated_ciphertexts.push_back(encryptedInputs[b]);
            for (int kw = 1; kw < kernelWidth; kw++) {
                rotated_ciphertexts.push_back(context->EvalRotate(encryptedInputs[b], kw));
            }
        }
        bactched_rotated_ciphertexts[b] = rotated_ciphertexts;
    }

    vector<Ctext> output_ciphertexts(outputChannels); 
    for(int outCh=0; outCh<outputChannels; outCh++){
        
        /** STEP 1: multiply rotated inputs with kernel data */
        vector<Ctext> input_channel_results(inputChannels);
        for(int inCh=0; inCh<inputChannels; inCh++){
            vector<Ctext> mult_results(kernelSize);
            auto encodeKernel = fheonHEController.encode_kernel(rawKernelData[outCh][inCh], inputSize);
            for (int k = 0; k < kernelSize; k++) {
                mult_results[k] = context->EvalMult(bactched_rotated_ciphertexts[inCh][k], encodeKernel[k]);
            }
            input_channel_results[inCh] = context->EvalAddMany(mult_results);
            mult_results.clear();
        }

        /** STEP 2: sum all results to create convolution results for every input */
        Ctext result = context->EvalAddMany(input_channel_results);
        input_channel_results.clear();

        // Step 3: Cleaning the resulting ciphertext by removing unwanted row elements
        Ctext cleanedrows =  encryptedzeros->Clone();
        for (int row = 0; row < outputWidth; row++) {
            Ctext masked = context->EvalMult(result, gen_row_mask_with_channels(row, outputWidth, inputSize, batchSize, result->GetLevel()));
            cleanedrows = context->EvalAdd(cleanedrows, masked);
            if(row < outputWidth-1){
                result = context->EvalRotate(result, (inputWidth - outputWidth));
            }
        }

        /** step 4: process per channel ******/ 
        Ctext cleanedchannels =  encryptedzeros->Clone();
        for (int ch=0; ch < batchSize; ch++){
            Ctext masked = context->EvalMult(cleanedrows, gen_channel_mask_with_zeros(ch, outputSize, batchSize, result->GetLevel()));
            cleanedchannels = context->EvalAdd(cleanedchannels, masked);
            if(ch < batchSize-1){
                cleanedrows = context->EvalRotate(cleanedrows, (inputSize - outputSize));
            }
        }

        /** Step 5: Clean out all irrelevant data from ciphertext with new dimensions */
        auto inbiasEncoded = context->MakeCKKSPackedPlaintext(rawBiasData[outCh], 1, result->GetLevel(), nullptr, nextPowerOf2(outVecSize));
        output_ciphertexts[outCh] = context->EvalAdd(context->EvalMult(cleanedchannels, cleaningoutputMask), inbiasEncoded); 

    }
    printTiming("Batch convolution evaluated...", "Conv", start_time);
    return output_ciphertexts;
}

/**
 * @brief Perform an optimized secure convolution layer evaluation on batched encrypted data.
 *
 * This function computes a convolution layer over multiple input channels for batched
 * data under homomorphic encryption. It is optimized for the special case where
 * stride = 1, kernel size = 3, and padding = 1, but also supports larger strides
 * by applying striding across multiple channels simultaneously (multi-channel approach),
 * improving efficiency for deep FHE-based networks.
 *
 * For each output channel, the function multiplies input ciphertexts by the corresponding
 * plaintext convolution kernels, sums contributions from all input channels, applies
 * bias, and rotates/aggregates results across batches to produce the final output.
 *
 * @param encryptedInputs     Vector of ciphertexts, one per input channel, each encoding
 *                            batched feature maps of size (batchSize × inputWidth × inputWidth).
 * @param batchSize            Number of batches encoded in the input ciphertexts.
 * @param inputWidth           Width (and height) of the input feature maps (assumed square).
 * @param inputChannels        Number of input channels.
 * @param outputChannels       Number of output channels.
 * @param stride          stride length used for the convolution operation.
 *
 * @param fheonHEController HE controller that encodes the kernels (encode_kernel_optimized) at the level of the rotated inputs.
 * @param rawKernelData Kernel weights [outputChannel][inputChannel][batch][3][3], repeated over the batch (UtilsBatchData::load_batch_weights).
 * @param rawBiasData Bias values [outputChannel][batch * outputWidth^2] (UtilsBatchData::load_batch_bias).
 * @return vector<Ctext>       Vector of ciphertexts containing the encrypted results
 *                            of the convolution layer, one ciphertext per output channel,
 *                            with batch data packed contiguously:
 *                            [batch0 output channel_i][batch1 output channel_i]...
 *
 * @note
 * - This function assumes ciphertext packing follows contiguous layout:
 *   [batch0: channel_i feature map][batch1: channel_i feature map]...
 * - Multi-channel striding is used to reduce the number of rotations and multiplications,
 *   making it efficient for FHE-based CNNs with multiple input channels.
 */
vector<Ctext> FHEONANNStandardBatchController::he_batch_convolution_optimized(
                FHEONHEController &fheonHEController,vector<Ctext> &encryptedInputs,
                vector<vector<vector<vector<vector<double>>>>> &rawKernelData,
                vector<vector<double>> &rawBiasData,
                int batchSize, int inputWidth, int inputChannels,
                int outputChannels, int stride) {

    
    auto start_time = startTime();
constexpr int kernelSq = 9;
    int outputWidth = inputWidth / stride;
    int inputSize = inputWidth * inputWidth;
    int outputSize = outputWidth * outputWidth;
    int inVecSize = batchSize * inputSize;
    int outVecSize = batchSize * outputSize;
    vector<double> cleaningoutputVec;
    if (stride > 1) {
        cleaningoutputVec = generate_mixed_mask(outVecSize, inVecSize);
    }


    // Precompute rotations (same as before)
    vector<vector<Ctext>> batched_rotated_ciphertexts(inputChannels);
    for (int i = 0; i < inputChannels; i++) {
        vector<Ctext> rotated_ciphertexts;
        Ctext encryptedInput = encryptedInputs[i]->Clone();
        auto digits = context->EvalFastRotationPrecompute(encryptedInput);

        Ctext first_shot = context->EvalFastRotation(encryptedInput, -1, context->GetCyclotomicOrder(), digits);
        Ctext second_shot = context->EvalFastRotation(encryptedInput, 1, context->GetCyclotomicOrder(), digits);
        rotated_ciphertexts.push_back(context->EvalRotate(first_shot, -inputWidth));
        rotated_ciphertexts.push_back(context->EvalFastRotation(encryptedInput, -inputWidth, context->GetCyclotomicOrder(), digits));
        rotated_ciphertexts.push_back(context->EvalRotate(second_shot, -inputWidth));
        rotated_ciphertexts.push_back(first_shot);
        rotated_ciphertexts.push_back(encryptedInput);
        rotated_ciphertexts.push_back(second_shot);
        rotated_ciphertexts.push_back(context->EvalRotate(first_shot, inputWidth));
        rotated_ciphertexts.push_back(context->EvalFastRotation(encryptedInput, inputWidth, context->GetCyclotomicOrder(), digits));
        rotated_ciphertexts.push_back(context->EvalRotate(second_shot, inputWidth));
        batched_rotated_ciphertexts[i] = rotated_ciphertexts;
        rotated_ciphertexts.clear();
    }

    // Allocate outputs
    vector<Ctext> output_ciphertexts(outputChannels);
    for (int outCh = 0; outCh < outputChannels; outCh++) {
        vector<Ctext> batched_results(inputChannels);
        vector<Ctext> mult_results(kernelSq);

        for (int inCh = 0; inCh < inputChannels; inCh++) {
            /** encoded at the level of the rotated input (like the shortcut weights): fewer towers to encode and store */
            auto encodeKernel = fheonHEController.encode_kernel_optimized(rawKernelData[outCh][inCh], inputSize,
                                                                          batched_rotated_ciphertexts[inCh][0]->GetLevel());
            for (int k = 0; k < kernelSq; k++) {
                mult_results[k] = context->EvalMult(batched_rotated_ciphertexts[inCh][k], encodeKernel[k]);
            }
            batched_results[inCh] = context->EvalAddMany(mult_results);
        }

        // Sum input-channel results
        Ctext result = context->EvalAddMany(batched_results);

        if (stride > 1) {
            result = FHEONANNController::downsample_with_multiple_channels(result, inputWidth, stride, batchSize);
            result = context->EvalMult(result, context->MakeCKKSPackedPlaintext(cleaningoutputVec, 1, result->GetLevel()));
        }

        auto inbiasEncoded = context->MakeCKKSPackedPlaintext(rawBiasData[outCh], 1, result->GetLevel(), nullptr, outVecSize);
        output_ciphertexts[outCh] = context->EvalAdd(result, inbiasEncoded);
        batched_results.clear();
    }

    batched_rotated_ciphertexts.clear();
    printTiming("Optimized batch convolution evaluated...", "Conv", start_time);
    return output_ciphertexts;
}

/**
 * @brief Perform a secure convolution operation on batched encrypted data.
 *
 * This function implements a convolutional layer in the encrypted domain
 * using homomorphic encryption for batched inputs. Given a set of encrypted
 * input feature maps, convolution kernels, and bias terms, it computes the
 * convolution across multiple channels and batches while respecting the
 * specified input dimensions, kernel size, padding, and stride.
 *
 * For each output channel, the function multiplies rotated input ciphertexts
 * by corresponding kernel weights, sums contributions from all input channels,
 * applies the bias, and aligns outputs across batches using rotations and
 * packing strategies. The result is one ciphertext per output channel,
 * encoding all batches contiguously.
 *
 * @param encryptedInputs   Vector of ciphertexts, one per input channel,
 *                          each encoding batched feature maps of size
 *                          (batchSize × inputWidth × inputWidth).
 * @param batchSize         Number of batches encoded in the input ciphertexts.
 * @param inputWidth        Width (and height) of the input feature maps (assumed square).
 * @param inputChannels     Number of input channels.
 * @param outputChannels    Number of output channels.
 * @param stride          stride length used for the convolution operation.
 *
 * @param fheonHEController HE controller that encodes the 1x1 weights (encode_input) at the level of each input.
 * @param rawKernelData 1x1 weights [outputChannel][inputChannel][batch * inputWidth^2], every weight broadcast over its channel.
 * @param rawBiasData Bias values [outputChannel][batch * outputWidth^2].
 * @return vector<Ctext>       Vector of ciphertexts containing the encrypted results
 *                            of the convolution layer, one ciphertext per output channel,
 *                            with batch data packed contiguously:
 *                            [batch0 output channel_i][batch1 output channel_i]...
 *
 * @note
 * - This function assumes ciphertext packing follows contiguous layout:
 *   [batch0: channel_i feature map][batch1: channel_i feature map]...
 * - Multi-channel striding is used to reduce the number of rotations and multiplications,
 *   making it efficient for FHE-based CNNs with multiple input channels.
 */
vector<Ctext> FHEONANNStandardBatchController::he_batch_convolution_shortcut_optimized(FHEONHEController &fheonHEController, vector<Ctext>& encryptedInputs, 
                            vector<vector<vector<double>>>& rawKernelData, vector<vector<double>>& rawBiasData,
                            int batchSize, int inputWidth, int inputChannels, int outputChannels, int stride){
    
    
    auto start_time = startTime();
int outputWidth = inputWidth / stride;
    int inputSize = inputWidth * inputWidth;
    int outputSize = outputWidth * outputWidth;
    // int encodeLevel = encryptedInputs[0]->GetLevel();
    int inVecSize = batchSize * inputSize;
    int outVecSize = batchSize * outputSize;
    vector<double> cleaningoutputVec = generate_mixed_mask(outVecSize, inVecSize);
    vector<Ctext> output_ciphertexts(outputChannels); 
 
           
    for (int outCh = 0; outCh < outputChannels; outCh++) {
        vector<Ctext> batched_results(inputChannels);
        for(int inCh=0; inCh<inputChannels; inCh++){
            auto encodeWeights = fheonHEController.encode_input(rawKernelData[outCh][inCh], inVecSize, encryptedInputs[inCh]->GetLevel());
            batched_results[inCh] = context->EvalMult(encryptedInputs[inCh], encodeWeights);
        }
        
        Ctext result = context->EvalAddMany(batched_results);
        result = FHEONANNController::downsample_with_multiple_channels(result, inputWidth, stride, batchSize);
        result = context->EvalMult(result, context->MakeCKKSPackedPlaintext(cleaningoutputVec, 1, result->GetLevel()));

        auto biasVectorEncoded = fheonHEController.encode_input(rawBiasData[outCh], outVecSize, result->GetLevel());
        output_ciphertexts[outCh] = context->EvalAdd(result, biasVectorEncoded);
        batched_results.clear();
    }

    printTiming("Optimized batch convolution and shortcut evaluated...", "Conv", start_time);
    return output_ciphertexts;
}


/**
 * @brief Securely sum two sets of encrypted inputs channel-wise.
 *
 * This function takes two vectors of ciphertexts, each representing encrypted 
 * input data across multiple channels, and performs homomorphic addition on 
 * corresponding elements. The result is a new vector of ciphertexts where each 
 * ciphertext is the sum of the two inputs for that channel.
 *
 * @param firstEncryptedInputs   Vector of ciphertexts for the first input, 
 *                                sized [inputChannels].
 * @param secondEncryptedInputs  Vector of ciphertexts for the second input, 
 *                                sized [inputChannels].
 * @param inputChannels           Number of channels (i.e., number of ciphertexts 
 *                                in each input vector).
 *
 * @return A vector of ciphertexts where each element is the homomorphic sum of 
 *         the corresponding inputs from the two provided vectors.
 */

vector<Ctext> FHEONANNStandardBatchController::he_batch_sum_ciphertexts(
    vector<Ctext>& firstEncryptedInputs,
    vector<Ctext>& secondEncryptedInputs,
    int inputChannels
) {

    
    auto start_time = startTime();
vector<Ctext> summed_ciphertexts(inputChannels);

    for (int i = 0; i < inputChannels; i++) {
        summed_ciphertexts[i] = context->EvalAdd(firstEncryptedInputs[i], secondEncryptedInputs[i]);
    }

    printTiming("Batch addition evaluated...", "Add", start_time);
    return summed_ciphertexts;
}




/**
 * @brief Perform optimized secure average pooling on batched encrypted feature maps.
 *
 * This function computes average pooling across multiple input channels 
 * for batched data under homomorphic encryption. For each channel, it 
 * partitions the input feature map into local regions of size 
 * (kernelWidth × kernelWidth), sums the values in each region, 
 * and applies scaling to obtain the average. The stride length determines 
 * how pooling regions are traversed across the feature map. 
 *
 * The operation is applied independently to each input channel. Within each 
 * channel, ciphertext rotations and additions are used efficiently to align 
 * and sum pooling regions across batches, minimizing the number of 
 * homomorphic operations. The output is one ciphertext per channel, where 
 * each ciphertext encodes the pooled results for all batches.
 *
 * @param encryptedInputs   Vector of ciphertexts, one per input channel, 
 *                          each encoding batched feature maps of size 
 *                          (batchSize × inputWidth × inputWidth).
 * @param batchSize         Number of batches encoded in the ciphertexts.
 * @param inputWidth        Width (and height) of the input feature maps (assumed square).
 * @param inputChannels     Number of input channels to pool.
 * @param kernelWidth       Width (and height) of the pooling kernel (assumed square).
 * @param stridingLen       Stride length used to slide the pooling window 
 *                          across the input feature maps.
 *
 * @return vector<Ctext>    Vector of ciphertexts where each element corresponds 
 *                          to one pooled channel across all batches. Each output 
 *                          ciphertext encodes multiple pooled values per batch, 
 *                          arranged as:
 *                          [batch0 pooled values][batch1 pooled values]...
 *
 * @note 
 * - This implementation is optimized for FHE-based CNN/ANN architectures where 
 *   reducing the number of ciphertext rotations and multiplications is critical 
 *   for efficiency. 
 * - The function assumes ciphertext packing follows contiguous layout: 
 *   [batch0: channel_i feature map][batch1: channel_i feature map]...
 */
vector<Ctext> FHEONANNStandardBatchController::he_optimzed_batch_avgPool(vector<Ctext>& encryptedInputs,  int batchSize, int inputWidth, int inputChannels, int kernelWidth, int stridingLen){
    
    
    auto start_time = startTime();
int kernelSize = pow(kernelWidth, 2);
    int inputSize = pow(inputWidth, 2);
    int outputWidth = inputWidth / stridingLen;
    int outputSize = outputWidth * outputWidth;
    int inVecSize = batchSize * inputSize;
    int outVecSize = batchSize * outputSize; 

    int encode_level = encryptedInputs[0]->GetLevel();
    Ptext reducingMask =  context->MakeCKKSPackedPlaintext(generate_scale_mask(kernelSize, inVecSize), 1, encode_level);
    Ptext cleaningInMask = context->MakeCKKSPackedPlaintext(generate_mixed_mask(inputSize, inVecSize), 1, encode_level);
    vector<double> cleaningoutputVec = generate_mixed_mask(outVecSize, inVecSize);

    /*** STEP 1 - ROTATE THE CIPHERTEXT into by k^2-1 and create a k^2 rotated right positions ***/
    vector<Ctext> batched_ciphertexts(inputChannels);
    for(int i=0; i<inputChannels; i++){
        vector<Ctext> rotated_ciphertexts;
        Ctext encryptedInput = encryptedInputs[i];
        auto digits = context->EvalFastRotationPrecompute(encryptedInput);
        rotated_ciphertexts.push_back(encryptedInput);
        rotated_ciphertexts.push_back(context->EvalFastRotation(encryptedInput, 1, context->GetCyclotomicOrder(), digits));
        rotated_ciphertexts.push_back(context->EvalFastRotation(encryptedInput, inputWidth, context->GetCyclotomicOrder(), digits));
        rotated_ciphertexts.push_back(context->EvalRotate(context->EvalFastRotation(encryptedInput, inputWidth, context->GetCyclotomicOrder(), digits), 1));
        Ctext sum_cipher = context->EvalAddMany(rotated_ciphertexts);

        /*** STEP 3: Multiply the scale value with the sum cipher */
        batched_ciphertexts[i] = context->EvalMult(sum_cipher, reducingMask);
    }
    
    /**** Caryout the average pooling if we have less than 4 elements per channel elements*/
    vector<Ctext> pooled_ciphertexts(inputChannels);
    if(inputWidth <= 2){
        for(int i=0; i<inputChannels; i++){
            Ctext inn_ciphertext = batched_ciphertexts[i];
            vector<Ctext> inn_ciphers;
            for(int i = 0; i<batchSize; i++){

                inn_ciphers.push_back(context->EvalSum(context->EvalMult(inn_ciphertext, cleaningInMask), inputSize));
                inn_ciphertext = context->EvalRotate(inn_ciphertext, inputSize);
            }
            
            pooled_ciphertexts[i] = context->EvalMerge(inn_ciphers);
        }
        cout << "input width is less than 2" << endl;
    }
    else{
        for(int i=0; i<inputChannels; i++){
            Ctext result = FHEONANNController::downsample_with_multiple_channels(batched_ciphertexts[i], inputWidth, stridingLen, batchSize);
            pooled_ciphertexts[i] = context->EvalMult(result, context->MakeCKKSPackedPlaintext(cleaningoutputVec, 1, result->GetLevel()));
        }
    }
    printTiming("Optimized batch average pooling evaluated...", "AvgPool", start_time);
    return pooled_ciphertexts;
}


/**
 * @brief Perform homomorphic global pooling over batched encrypted inputs.
 *
 * This function applies a global pooling operation on encrypted feature maps
 * across the spatial dimensions for each input channel and batch element.
 * For each channel, the encrypted input is rotated and summed to aggregate
 * all spatial positions, merged across batches, and finally scaled using
 * a plaintext reduction mask to compute the pooled result.
 *
 * The channels are processed one after the other (FHEONANNStandardBatchThreadedController runs them in parallel).
 *
 * @param encryptedInputs  Vector of ciphertexts representing encrypted feature
 *                         maps, sized [inputChannels], where each ciphertext
 *                         packs batched spatial data.
 * @param batchSize        Number of images packed in each ciphertext.
 * @param inputWidth       Width (and height) of the input feature map.
 * @param inputChannels    Number of input channels.
 * @param kernelWidth      Width of the pooling kernel (used for layout logic).
 * @param rotatePositions  Number of rotations required to align batch elements
 *                         during ciphertext merging.
 *
 * @return A vector of ciphertexts, one per channel, where each ciphertext
 *         contains the globally pooled and scaled encrypted output.
 */

vector<Ctext> FHEONANNStandardBatchController::he_batch_globalpool(vector<Ctext>& encryptedInputs, int batchSize, int inputWidth, 
                        int inputChannels, int kernelWidth, int rotatePositions){
    
    
    auto start_time = startTime();
    int inputSize = inputWidth*inputWidth;
    // auto reducingMask =  context->MakeCKKSPackedPlaintext(generate_scale_mask(inputSize, batchSize), 1, encryptedInputs[0]->GetLevel());
    vector<double> reducingMask = generate_scale_mask(inputSize, batchSize);
    vector<Ctext> pooled_ciphertexts(inputChannels);

    for(int i = 0; i< inputChannels; i++){
        vector<Ctext> result_ciphers;
        vector<Ctext> batched_ciphertexts;
        int rotation_index = 0;
        int j = 0;
        Ctext encryptedInput = encryptedInputs[i]->Clone();
        for(int b=0; b<batchSize; b++){
            if(b != 0){
                encryptedInput = context->EvalRotate(encryptedInput, inputSize);
            }
            result_ciphers.push_back(context->EvalSum(encryptedInput, inputSize));
            // cout << "inputchannels: " << i << " batchSize: "<< b << endl; 
            /** check whether is equal to imgcols, merge them and rotate by imgCols. 
            * If i is equal to the outputSize, merge and rotate by imgCols */
            if(j == (rotatePositions-1) || b == (batchSize-1)){
                // cout << "rotatePositions: "<< rotation_index << endl; 
                Ctext merged = context->EvalMerge(result_ciphers);
                if(rotation_index > 0){
                    merged = context->EvalRotate(merged, -rotation_index);
                    batched_ciphertexts.push_back(merged);
                }
                else{
                    batched_ciphertexts.push_back(merged); 
                }
                rotation_index += rotatePositions;
                result_ciphers.clear();
                j=0;
            }
            else{
                j+=1;
            }
        }
        pooled_ciphertexts[i] = context->EvalMult(context->EvalAddMany(batched_ciphertexts), context->MakeCKKSPackedPlaintext(reducingMask, 1, encryptedInput->GetLevel()));
        batched_ciphertexts.clear();
    }
    printTiming("Batch global average pooling evaluated...", "AvgPool", start_time);
    return pooled_ciphertexts; 
}


/**
 * @brief Perform a homomorphic batched linear (fully connected) layer.
 *
 * This function evaluates a fully connected layer over encrypted batched inputs
 * using CKKS packing. The encrypted input is multiplied with a plaintext weight
 * matrix, followed by rotations and slot-wise summations to compute inner
 * products for each output neuron and each batch element.
 *
 * Intermediate results are merged per batch, rotated into their correct global
 * slot positions, and finally accumulated across batches. A plaintext bias is
 * then added to produce the final encrypted output.
 *
 * The output neurons and batch elements are processed one after the other
 * (FHEONANNStandardBatchThreadedController runs them in parallel).
 *
 * @param encryptedInput  Ciphertext containing packed batched input features.
 * @param weightMatrix    Vector of plaintexts representing the weight matrix,
 *                        sized [outputSize], where each plaintext corresponds
 *                        to one output neuron.
 * @param baisInput       Plaintext bias packed and aligned with the output slots.
 * @param batchSize       Number of inputs packed in the ciphertext.
 * @param inputSize       Number of input features per batch element.
 * @param outputSize      Number of output neurons.
 * @param rotationIndex   Slot grouping size used to correctly align merged
 *                        batch outputs via rotations.
 *
 * @return A ciphertext containing the encrypted outputs of the linear layer
 *         for all batches and output neurons.
 */

Ctext FHEONANNStandardBatchController::he_batch_linear(Ctext& encryptedInput, vector<Ptext>& weightMatrix,
    Ptext& baisInput, int batchSize, int inputSize,
    int outputSize, int rotationIndex){
    
    vector<Ctext> result_matrix(batchSize);
    vector<vector<Ctext>> batched_ciphertexts(batchSize, vector<Ctext>(outputSize));

    for(int i = 0; i< outputSize; i++){
    // For each batch element, rotate base and sum -> push into that batch's vector
        Ctext rotated_temp = context->EvalMult(encryptedInput, weightMatrix[i]);
        for (int b = 0; b < batchSize; b++) {
            if (b != 0) {
                rotated_temp = context->EvalRotate(rotated_temp, inputSize);
            }
            batched_ciphertexts[b][i] = context->EvalSum(rotated_temp, inputSize);
        }
    }

    // cout << "Inputs rows all calculated" << endl; 

    for(int b = 0; b < batchSize; b++){
        // cout << "batched_ciphertexts[" << b << "] size: " << batched_ciphertexts[b].size() << " --batch Rotate: " << -(b * outputSize) << " --rotation_index "<< rotation_index << endl;
        Ctext inn_ciphertext = context->EvalMerge(batched_ciphertexts[b]);
        // Apply rotations to place merged block into the correct global slot
        if (b != 0){ 
            int rot_idx = (b * outputSize); 
            int base_idx = (rot_idx / rotationIndex) * rotationIndex;
            int mod_idx = rot_idx % rotationIndex; 
            if(base_idx > 0){
                inn_ciphertext = context->EvalRotate(inn_ciphertext, -base_idx);
            }
            if(mod_idx > 0){
                inn_ciphertext = context->EvalRotate(inn_ciphertext, -mod_idx);
            }
        }
        result_matrix[b] = inn_ciphertext;
        // batched_ciphertexts[b].clear();
    }

    // Combine results and add bias
    Ctext fc_results = context->EvalAdd(context->EvalAddMany(result_matrix), baisInput);
    result_matrix.clear();
    batched_ciphertexts.clear();
    return fc_results;
}


/**
 * @brief Perform a homomorphic batched linear layer with per-batch outputs.
 *
 * This function evaluates a fully connected (linear) layer over encrypted,
 * batched inputs and returns a separate ciphertext for each batch element.
 * The encrypted input is multiplied with plaintext weight vectors, followed
 * by rotations and slot-wise summations to compute inner products for each
 * output neuron.
 *
 * For each batch element, the per-output ciphertexts are merged into a single
 * ciphertext and a plaintext bias is added. Unlike the single-output variant,
 * this function preserves batch separation in the output by returning a
 * vector of ciphertexts, one per batch.
 *
 * The output neurons and batch elements are processed one after the other
 * (FHEONANNStandardBatchThreadedController runs them in parallel).
 *
 * @param encryptedInput  Ciphertext containing packed batched input features.
 * @param weightMatrix    Vector of plaintexts representing the weight matrix,
 *                        sized [outputSize], where each plaintext corresponds
 *                        to one output neuron.
 * @param baisInput       Plaintext bias packed and aligned with the output slots.
 * @param batchSize       Number of inputs packed in the ciphertext.
 * @param inputSize       Number of input features per batch element.
 * @param outputSize      Number of output neurons.
 *
 * @return A vector of ciphertexts of size [batchSize], where each ciphertext
 *         contains the encrypted outputs of the linear layer for one batch
 *         element.
 */

vector<Ctext> FHEONANNStandardBatchController::he_batch_linear_multiple_outputs(Ctext& encryptedInput, vector<Ptext>& weightMatrix,
    Ptext& baisInput, int batchSize, int inputSize, int outputSize){
    
    auto start_time = startTime();
    vector<vector<Ctext>> batched_ciphertexts(batchSize, vector<Ctext>(outputSize));

    for(int i = 0; i< outputSize; i++){
        Ctext temp = context->EvalMult(encryptedInput, weightMatrix[i]);
        for (int b = 0; b < batchSize; b++) {
            if (b != 0) {
                temp = context->EvalRotate(temp, inputSize);
            }
            batched_ciphertexts[b][i] = context->EvalSum(temp, inputSize);
        }
    }


    weightMatrix.clear();
    weightMatrix.shrink_to_fit();

    
    vector<Ctext> result_matrix(batchSize);
    for(int b = 0; b < batchSize; b++){
        result_matrix[b] = context->EvalAdd(context->EvalMerge(batched_ciphertexts[b]), baisInput);
    }

    // Combine results and add bias
    batched_ciphertexts.clear();
    printTiming("Batch linear with multiple outputs evaluated...", "FC", start_time);
    return result_matrix;
}



/**
 * @brief Apply a homomorphic ReLU activation using Chebyshev polynomial approximation.
 *
 * This function evaluates a ReLU activation on encrypted inputs using a
 * Chebyshev polynomial approximation over a bounded interval. Each input
 * channel is optionally rescaled using a plaintext mask to control the output
 * magnitude before applying the polynomial.
 *
 * The ReLU function is approximated as:
 *   f(x) = 0              if x < 0
 *   f(x) = scaleVal * x   otherwise
 *
 * where the approximation is evaluated over the interval [lowerBound, upperBound].
 * The computation is parallelized across input channels.
 *
 * @param encryptedInputs  Vector of ciphertexts representing encrypted inputs,
 *                         sized [inputChannels].
 * @param scaleValues      Per-channel scaling factors applied before ReLU.
 * @param inputChannels    Number of input channels.
 * @param vectorSize       Number of valid slots per ciphertext.
 * @param polyDegree       Degree of the Chebyshev polynomial approximation.
 *
 * @return A vector of ciphertexts where each element contains the encrypted
 *         ReLU-activated output for the corresponding input channel.
 */

vector<Ctext> FHEONANNStandardBatchController::he_batch_relu(vector<Ctext>& encryptedInputs, vector<int> scaleValues,  int inputChannels, int vectorSize, int polyDegree) {
    
    auto start_time = startTime();
    double lowerBound = -1;
    double upperBound = 1;
    
    // int scaleVal = 110;
    vector<Ctext> relu_results(inputChannels);
    int totalElements =  nextPowerOf2(vectorSize);

    for(int i= 0; i< inputChannels; i++){
        int scaleVal = scaleValues[i];
        if(scaleVal >= 1){
            scaleVal = 2*scaleVal;
            // auto mask_data = context->MakeCKKSPackedPlaintext(generate_scale_mask(scaleVal, vectorSize), 1, 1, nullptr, totalElements);
            auto mask_data = context->MakeCKKSPackedPlaintext(generate_scale_mask(scaleVal, vectorSize), 1, encryptedInputs[i]->GetLevel(), nullptr, totalElements);
            encryptedInputs[i] = context->EvalMult(encryptedInputs[i], mask_data);
        }
        relu_results[i] = context->EvalChebyshevFunction(
                        [scaleVal](double x) -> double { if (x < 0) return 0; else return (scaleVal) * x; }, 
                                            encryptedInputs[i],
                                            lowerBound,
                                            upperBound, 
                                            polyDegree);
    }
    
    printTiming("Batch ReLU evaluated...", "ReLU", start_time);
    return relu_results;
}

/**
 * @brief Convert channel-separated encrypted inputs into a single batched ciphertext.
 *
 * This function takes encrypted inputs where each ciphertext corresponds to one input channel. 
 * It aligns all channel data together within a batch, producing a single ciphertext that 
 * represents all inputs across channels and batches. This transformation is necessary 
 * when passing data into fully connected layers, which expect a flat neuron-style input 
 * rather than channel-separated structures (as in convolutional layers).
 *
 * @param encryptedInputs A vector of ciphertexts, where each ciphertext corresponds to one input channel.
 * @param batchSize Number of batches to process.
 * @param inputChannels Number of input channels per batch.
 *
 * @param inputWidth Width (and height) of every input channel (inputWidth^2 slots per channel and image).
 * @param baseIndex Rotation split base: a shift s is done as a rotation by (s / baseIndex) * baseIndex followed by one by s % baseIndex, so only multiples of baseIndex and residues below it need keys.
 * @return A single ciphertext that encodes the entire batch across all channels, 
 *         aligned in a flat neuron-compatible format.
 *
 * @note 
 * - Internally, the function:
 *   - Uses a cleaning mask to isolate values in each channel.
 *   - Applies rotations to align channel data.
 *   - Aggregates all channel ciphertexts for each batch.
 *   - Applies additional rotations to stack batches correctly.
 * - This ensures compatibility between convolutional outputs (channel-based) 
 *   and fully connected layers (neuron-based).
 */
Ctext FHEONANNStandardBatchController::he_batch_inputs_converter(vector<Ctext>& encryptedInputs, int batchSize, int inputChannels, int inputWidth, int baseIndex) {
    
    // auto start_time = startTime();
    int pixelsPerChannel = inputWidth * inputWidth; 
    int batchNeuronCount = inputChannels * pixelsPerChannel;
    int level = encryptedInputs[0]->GetLevel();
    int rotIndex = 1;
    vector<Ctext> allProjections;

    for (int b = 0; b < batchSize; b++) {
        // Create mask for batch b
        vector<double> mask(batchSize * pixelsPerChannel, 0.0);
        for (int p = 0; p < pixelsPerChannel; p++) {
            mask[b * pixelsPerChannel + p] = 1.0;
        }
        Ptext pMask = context->MakeCKKSPackedPlaintext(mask, 1, level);

        for (int i = 0; i < inputChannels; i++) {
            Ctext isolated = context->EvalMult(encryptedInputs[i], pMask);
            
            int currentPos = b * pixelsPerChannel;
            int targetPos = b * batchNeuronCount + i * pixelsPerChannel;
            int shift = currentPos - targetPos;

            if (shift != 0) {
                int base_idx = (abs(shift) / baseIndex) * baseIndex;
                int mod_idx = abs(shift) % baseIndex;
                int sign = (shift > 0) ? 1 : -1;
                
                if (base_idx != 0){
                    rotIndex = sign * base_idx; 
                    isolated = context->EvalRotate(isolated, rotIndex);
                }
                if (mod_idx != 0){
                    rotIndex = sign * mod_idx;
                    isolated = context->EvalRotate(isolated, rotIndex);
                }
            }
            allProjections.push_back(isolated);
        }
    }

    Ctext finalCipher = context->EvalAddMany(allProjections);
    allProjections.clear();
    // printTiming("Batch inputs converter evaluated...", "Conv", start_time);
    return finalCipher;
}


/**
 * @brief Generate a zero mask of given size.
 *
 * @param size Number of elements in the mask.
 * @param level Encryption level for CKKS plaintext.
 * @return Packed plaintext mask with all zeros.
 */
Ptext FHEONANNStandardBatchController::gen_zero_mask(int size, int level) {
    vector<double> mask(size, 0.0);
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}


/**
 * @brief Generate a mask selecting a specific row in every channel.
 *
 * @param row Row index to select.
 * @param width Width of each channel.
 * @param inputSize Number of elements per channel.
 * @param level Encryption level for CKKS plaintext.
 * @param batchSize Number of images: the row mask of one image (inputSize slots) is repeated batchSize times.
 * @return Packed plaintext mask with the row selected in all channels.
 */
Ptext FHEONANNStandardBatchController::gen_row_mask_with_channels(int row, int width, int inputSize, int batchSize, int level) {
    
    vector<double> baseMask;
    for (int j = 0; j < (row * width); j++) {
        baseMask.push_back(0);
    }
    for (int j = 0; j < width; j++) {
        baseMask.push_back(1);
    }
    for (int j = 0; j < (inputSize - width - (row * width)); j++) {
        baseMask.push_back(0);
    }

    // repeat baseMask n times
    vector<double> mask;
    mask.reserve(baseMask.size() * batchSize);
    for (int i = 0; i < batchSize; i++) {
        mask.insert(mask.end(), baseMask.begin(), baseMask.end());
    }

    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}

/**
 * @brief Generate a mask selecting a specific channel while zeroing all others.
 *
 * @param channel Channel index to select.
 * @param outputSize Number of elements per channel.
 * @param numChannels Total number of channels.
 * @param level Encryption level for CKKS plaintext.
 * @return Packed plaintext mask with the selected channel set to 1.
 */
Ptext FHEONANNStandardBatchController::gen_channel_mask_with_zeros(int channel, int outputSize, int numChannels, int level ){
    
    int totalSlots = outputSize * numChannels;
    vector<double> mask(totalSlots, 0.0);

    int pos = channel * outputSize;
    for (int i = 0; i < outputSize; i++) {
        mask[pos + i] = 1.0;
    }
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}

Ctext FHEONANNStandardBatchController::he_batch_inputs_converter_memory_efficient(vector<Ctext>& encryptedInputs, int batchSize, int inputChannels, int inputWidth, int baseIndex) {
    auto start_time = startTime();
    int pixelsPerChannel = inputWidth * inputWidth; 
    int batchNeuronCount = inputChannels * pixelsPerChannel;
    int level = encryptedInputs[0]->GetLevel();

    // Precompute plaintext masks for each batch
    vector<Ptext> pMasks(batchSize);
    for (int b = 0; b < batchSize; b++) {
        vector<double> mask(batchSize * pixelsPerChannel, 0.0);
        for (int p = 0; p < pixelsPerChannel; p++) {
            mask[b * pixelsPerChannel + p] = 1.0;
        }
        pMasks[b] = context->MakeCKKSPackedPlaintext(mask, 1, level);
    }

    /** the projections of all the input channels are summed in one binary tree */
    vector<Ctext> local_tree;
    for (int b = 0; b < batchSize; b++) {
        for (int i = 0; i < inputChannels; i++) {
            Ctext isolated = context->EvalMult(encryptedInputs[i], pMasks[b]);
            
            int currentPos = b * pixelsPerChannel;
            int targetPos = b * batchNeuronCount + i * pixelsPerChannel;
            int shift = currentPos - targetPos;

            if (shift != 0) {
                int base_idx = (abs(shift) / baseIndex) * baseIndex;
                int mod_idx = abs(shift) % baseIndex;
                int sign = (shift > 0) ? 1 : -1;
                
                if (base_idx != 0){
                    int rotVal = sign * base_idx; 
                    isolated = context->EvalRotate(isolated, rotVal);
                }
                if (mod_idx != 0){
                    int rotVal = sign * mod_idx;
                    isolated = context->EvalRotate(isolated, rotVal);
                }
            }
            
            // Merge isolated into local_tree
            Ctext current = isolated;
            size_t tree_level = 0;
            while (tree_level < local_tree.size() && local_tree[tree_level] != nullptr) {
                current = context->EvalAdd(current, local_tree[tree_level]);
                local_tree[tree_level] = nullptr;
                tree_level++;
            }
            if (tree_level == local_tree.size()) {
                local_tree.push_back(current);
            } else {
                local_tree[tree_level] = current;
            }
        }
    }

    // Reduce local_tree to a single ciphertext
    Ctext finalCipher = nullptr;
    for (auto& cl : local_tree) {
        if (cl != nullptr) {
            if (finalCipher == nullptr) {
                finalCipher = cl;
            } else {
                finalCipher = context->EvalAdd(finalCipher, cl);
            }
        }
    }

    printTiming("Batch inputs converter memory efficient evaluated...", "Conv", start_time);
    return finalCipher;
}

Ctext FHEONANNStandardBatchController::he_batch_linear_memory_efficient(
    Ctext& encryptedInput, vector<Ptext>& weightMatrix,
    Ptext& baisInput, int batchSize, int inputSize,
    int outputSize, int rotationIndex){
    
    auto start_time = startTime();
    vector<Ctext> result_matrix(batchSize);
    
    vector<Ctext> current_rotated_temp(outputSize);
    for(int i = 0; i < outputSize; i++){
        current_rotated_temp[i] = context->EvalMult(encryptedInput, weightMatrix[i]);
    }

    for (int b = 0; b < batchSize; b++) {
        vector<Ctext> batch_i_ciphers(outputSize);
        for(int i = 0; i < outputSize; i++){
            if (b != 0) {
                current_rotated_temp[i] = context->EvalRotate(current_rotated_temp[i], inputSize);
            }
            batch_i_ciphers[i] = context->EvalSum(current_rotated_temp[i], inputSize);
        }
        
        Ctext inn_ciphertext = context->EvalMerge(batch_i_ciphers);
        if (b != 0){ 
            int rot_idx = (b * outputSize); 
            int base_idx = (rot_idx / rotationIndex) * rotationIndex;
            int mod_idx = rot_idx % rotationIndex; 
            if(base_idx > 0){
                inn_ciphertext = context->EvalRotate(inn_ciphertext, -base_idx);
            }
            if(mod_idx > 0){
                inn_ciphertext = context->EvalRotate(inn_ciphertext, -mod_idx);
            }
        }
        result_matrix[b] = inn_ciphertext;
    }

    // Combine results and add bias
    Ctext fc_results = context->EvalAdd(context->EvalAddMany(result_matrix), baisInput);
    result_matrix.clear();
    printTiming("Batch linear memory efficient evaluated...", "FC", start_time);
    return fc_results;
}


/************************************ pipeline linear layers (one channel per ciphertext) ************************************/

/**
 * @brief Rotations of he_batch_linear_packed: the pooling sums (powers of two below poolSize) and the shifts d = s - r
 *        between input position s and output position r of a block.
 */
vector<int> FHEONANNStandardBatchController::generate_batch_linear_packed_rotation_positions(int inputsPerBlock, int outputsPerBlock, int poolSize){
    vector<int> keys_position;
    for (int step = 1; step < poolSize; step <<= 1) keys_position.push_back(step);
    for (int d = -(outputsPerBlock - 1); d < inputsPerBlock; d++) {
        if (d != 0) keys_position.push_back(d);
    }
    return keys_position;
}

/**
 * @brief Masks of he_batch_linear_packed: mask r keeps slot r of every block of blockSize slots, times inputScale.
 */
vector<Ptext> FHEONANNStandardBatchController::encode_batch_linear_packed_masks(int blockSize, int outputsPerBlock, int numSlots, int encodeLevel, double inputScale){
    vector<Ptext> masks(outputsPerBlock);
    vector<double> values(numSlots);
    for (int r = 0; r < outputsPerBlock; r++) {
        std::fill(values.begin(), values.end(), 0.0);
        for (int block = 0; block < numSlots / blockSize; block++) values[block * blockSize + r] = inputScale;
        masks[r] = context->MakeCKKSPackedPlaintext(values, 1, encodeLevel, nullptr, numSlots);
    }
    return masks;
}

/**
 * @brief Bias of he_batch_linear_packed: chunk c holds bias[c * outputsPerBlock + r] in slot r of every block.
 */
vector<Ptext> FHEONANNStandardBatchController::encode_batch_linear_packed_bias(const vector<double>& biasData, int blockSize, int outputsPerBlock,
                                                                              int numSlots, int encodeLevel){
    int outputs = biasData.size();
    int numChunks = (outputs + outputsPerBlock - 1) / outputsPerBlock;
    vector<Ptext> encoded(numChunks);
    vector<double> values(numSlots);
    for (int c = 0; c < numChunks; c++) {
        std::fill(values.begin(), values.end(), 0.0);
        for (int r = 0; r < outputsPerBlock && c * outputsPerBlock + r < outputs; r++) {
            for (int block = 0; block < numSlots / blockSize; block++) values[block * blockSize + r] = biasData[c * outputsPerBlock + r];
        }
        encoded[c] = context->MakeCKKSPackedPlaintext(values, 1, encodeLevel, nullptr, numSlots);
    }
    return encoded;
}

/**
 * @brief Taps of one input of he_batch_linear_packed: the input (summed over the poolSize slots of every block when
 *        poolSize > 1) rotated by every shift d in [-(outputsPerBlock - 1), inputsPerBlock - 1], hoisted.
 */
vector<Ctext> FHEONANNStandardBatchController::packed_linear_taps(const Ctext& encryptedInput, int poolSize, int inputsPerBlock, int outputsPerBlock){
    Ctext input = encryptedInput->Clone();
    for (int step = 1; step < poolSize; step <<= 1) {
        input = context->EvalAdd(input, context->EvalRotate(input, step));
    }
    auto digits = context->EvalFastRotationPrecompute(input);
    vector<Ctext> taps;
    for (int d = -(outputsPerBlock - 1); d < inputsPerBlock; d++) {
        taps.push_back(d == 0 ? input : context->EvalFastRotation(input, d, context->GetCyclotomicOrder(), digits));
    }
    return taps;
}

/**
 * @brief Output chunk c of he_batch_linear_packed: for every output position r, the weighted sum of the taps that bring
 *        input j = p * inputsPerBlock + s to position r (shift s - r), times mask r; then the bias.
 */
Ctext FHEONANNStandardBatchController::packed_linear_chunk(const vector<vector<Ctext>>& taps, const vector<vector<double>>& weightData,
                                                           const vector<Ptext>& masks, Ptext& biasInput, int chunk, int inputsPerBlock, int outputsPerBlock){
    int outputs = weightData.size();
    int inputs = weightData[0].size();
    Ctext result;
    for (int r = 0; r < outputsPerBlock && chunk * outputsPerBlock + r < outputs; r++) {
        int k = chunk * outputsPerBlock + r;
        /** weighted sum of the taps that reach position r (EvalMult by a constant: one level) */
        Ctext weightedSum;
        for (int j = 0; j < inputs; j++) {
            if (std::abs(weightData[k][j]) < 1e-10) continue;
            Ctext weighted = context->EvalMult(taps[j / inputsPerBlock][j % inputsPerBlock - r + outputsPerBlock - 1], weightData[k][j]);
            if (weightedSum) context->EvalAddInPlace(weightedSum, weighted);
            else weightedSum = weighted;
        }
        if (!weightedSum) continue;
        Ctext term = context->EvalMult(weightedSum, masks[r]);
        if (result) context->EvalAddInPlace(result, term);
        else result = term;
    }
    if (!result) {
        /** no weight at all: a zero ciphertext at the level of the other chunks */
        result = context->EvalMult(context->EvalMult(taps[0][outputsPerBlock - 1], 0.0), masks[0]);
    }
    return context->EvalAdd(result, biasInput);
}

/**
 * @brief Linear layer on packed batch ciphertexts (see the declaration for the layout), in two levels.
 *
 * Every input is rotated once per shift (hoisted); every output position of a chunk is one weighted sum of the taps
 * that reach it, masked to that position. weightData is [outputs][inputs].
 */
vector<Ctext> FHEONANNStandardBatchController::he_batch_linear_packed(vector<Ctext>& encryptedInputs, const vector<vector<double>>& weightData,
                            vector<Ptext>& biasInputs, const vector<Ptext>& masks, int inputsPerBlock, int outputsPerBlock, int poolSize){
    auto start_time = startTime();
    vector<vector<Ctext>> taps(encryptedInputs.size());
    for (size_t a = 0; a < encryptedInputs.size(); a++) {
        taps[a] = packed_linear_taps(encryptedInputs[a], poolSize, inputsPerBlock, outputsPerBlock);
    }
    int numChunks = biasInputs.size();
    vector<Ctext> outputs(numChunks);
    for (int c = 0; c < numChunks; c++) {
        outputs[c] = packed_linear_chunk(taps, weightData, masks, biasInputs[c], c, inputsPerBlock, outputsPerBlock);
    }
    printTiming("Packed batch linear layer evaluated (" + to_string(numChunks) + " output chunks)...", "FC", start_time);
    return outputs;
}
