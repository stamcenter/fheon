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
// #include <thread>
#include "FHEONANNBatchController.h"

namespace fs = std::filesystem;

void FHEONANNBatchController::setContext(CryptoContext<DCRTPoly>& in_context){
    context = in_context;
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
 * @param kernelData        Convolution kernels represented as a 3D vector of
 *                          plaintexts: [outputChannel][inputChannel][kernelWeights].
 * @param biasInputs        Bias terms for each output channel (plaintexts).
 * @param batchSize         Number of batches encoded in the input ciphertexts.
 * @param inputWidth        Width (and height) of the input feature maps (assumed square).
 * @param inputChannels     Number of input channels.
 * @param outputChannels    Number of output channels.
 * @param kernelWidth       Width (and height) of the convolution kernel (assumed square).
 * @param paddingLen        Number of zero-padding slots applied around the input feature map.
 * @param stride       stride length used for the convolution operation.
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
vector<int> FHEONANNBatchController::generate_convolution_batch_rotation_positions(int batchSize, int inputWidth, int kernelWidth, int padding, int stride) {
        
    vector<int> keys_position;
    int inputWidth_sq = pow(inputWidth, 2);
    int padded_width = inputWidth+(2*padding);
    int width_out = ((padded_width - (kernelWidth - 1) - 1)/stride)+1;
    int width_out_sq = pow(width_out,2);
    keys_position.push_back(inputWidth);
    keys_position.push_back(-inputWidth);
    keys_position.push_back(-1);
    keys_position.push_back(1);

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
    auto new_end = std::remove(keys_position.begin(), keys_position.end(), 0);
    new_end = std::unique(keys_position.begin(), keys_position.end());
    unique(keys_position.begin(), keys_position.end());
    keys_position.erase(new_end, keys_position.end());
    std::sort(keys_position.begin(), keys_position.end());
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
vector<int> FHEONANNBatchController::generate_avgpool_batch_optimized_rotation_positions(int batchSize, int inputWidth, int kernelWidth, int stride, 
                                bool globalPooling, int rotationIndex) {
    
    vector<int> keys_position;
    if(globalPooling){
        keys_position.push_back(kernelWidth*kernelWidth);
        keys_position.push_back((inputWidth*inputWidth));
        for(int pos=0; pos<batchSize; pos+=rotationIndex){
            keys_position.push_back(-pos);
        }
        for(int i=1; i<=rotationIndex; i++){
            keys_position.push_back(i);
            keys_position.push_back(-i);
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
    auto new_end = std::remove(keys_position.begin(), keys_position.end(), 0);
    new_end = std::unique(keys_position.begin(), keys_position.end());
    unique(keys_position.begin(), keys_position.end());
    keys_position.erase(new_end, keys_position.end());
    std::sort(keys_position.begin(), keys_position.end());

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
 * @return vector<int>   Vector of rotation positions required for the fully connected layer.
 */
vector<int> FHEONANNBatchController::generate_linear_batch_rotation_positions(int batchSize, vector<int> outputSizes, vector<int> inputSizes, int rotationIndex) {
    
    vector<int> keys_position;
    // int chunkSize = 128;
    for(size_t k =0; k<inputSizes.size(); k++){
        keys_position.push_back(inputSizes[k]);

        // if(chunkSize < batchSize){
        //     for(int i=0; i<batchSize/chunkSize; i++){
        //         int rotIndex = chunkSize * inputSizes[k];
        //         keys_position.push_back(rotIndex);
        //     }
        // }
    }

    for(size_t i=0; i<outputSizes.size(); i++){
        for(int j=1; j<=outputSizes[i]; j++){
            keys_position.push_back(j);
            keys_position.push_back(-j);

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
    auto new_end = std::remove(keys_position.begin(), keys_position.end(), 0);
    new_end = std::unique(keys_position.begin(), keys_position.end());
    unique(keys_position.begin(), keys_position.end());
    keys_position.erase(new_end, keys_position.end());
    std::sort(keys_position.begin(), keys_position.end());
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
 * @param inputSize      Size of the input feature maps (width × height).
 *
 * @return vector<int>   Vector of rotation positions required for input conversion.
 */
vector<int> FHEONANNBatchController::generate_batch_inputs_converter_rotation_positions(int batchSize, int inputChannels, int inputSize) {
    
    vector<int> keys_position;
    int batchKey = inputChannels * pow(inputSize, 2);
    keys_position.push_back(inputSize);
    keys_position.push_back(inputSize*inputSize);

    for(int b=1; b<batchSize; b++){
        int rot_idx = (b * batchKey); 
        int base_idx = (rot_idx / baseIndex) * baseIndex;
        int mod_idx = rot_idx % baseIndex; 
        keys_position.push_back(-base_idx);
        keys_position.push_back(-mod_idx);
    }

    for(int i=1; i<=inputChannels; i++){
        keys_position.push_back(-i);
        keys_position.push_back(i);
    }
    
    std::sort(keys_position.begin(), keys_position.end());
    auto new_end = std::remove(keys_position.begin(), keys_position.end(), 0);
    new_end = std::unique(keys_position.begin(), keys_position.end());
    unique(keys_position.begin(), keys_position.end());
    keys_position.erase(new_end, keys_position.end());
    std::sort(keys_position.begin(), keys_position.end());
    return keys_position;
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
 * @param kernelData           Convolution kernels, represented as a 3D vector of plaintexts:
 *                             [outputChannel][inputChannel][kernelWeights].
 * @param biasInputs           Bias terms for each output channel (plaintexts).
 * @param batchSize            Number of batches encoded in the input ciphertexts.
 * @param inputWidth           Width (and height) of the input feature maps (assumed square).
 * @param inputChannels        Number of input channels.
 * @param outputChannels       Number of output channels.
 * @param stride          stride length used for the convolution operation.
 *
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
vector<Ctext> FHEONANNBatchController::he_batch_convolution_optimized(
                FHEONHEController &fheonHEController,vector<Ctext> &encryptedInputs,
                vector<vector<vector<vector<vector<double>>>>> &rawKernelData,
                vector<vector<double>> &rawBiasData,
                int batchSize, int inputWidth, int inputChannels,
                int outputChannels, int stride) {

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

    int numThreads = std::min(inputChannels, (int)thread::hardware_concurrency());
    vector<thread> threads(numThreads);

    // Precompute rotations (same as before)
    vector<vector<Ctext>> batched_rotated_ciphertexts(inputChannels);
    auto in_worker = [&](int start, int end) {
        for (int i = start; i < end; i++) {
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
    };

    int block = (inputChannels + numThreads - 1) / numThreads;
    for (int t = 0; t < numThreads; t++) {
        int start = t * block;
        int end = min(start + block, inputChannels);
        threads[t] = thread(in_worker, start, end);
    }
    for (auto &th : threads) th.join();

    // Allocate outputs
    int outnumThreads = std::min(outputChannels, (int)thread::hardware_concurrency());
    vector<thread> outThreads(outnumThreads);
    vector<Ctext> output_ciphertexts(outputChannels);
    auto worker = [&](int start, int end) {
        for (int outCh = start; outCh < end; outCh++) {
            vector<Ctext> batched_results(inputChannels);
            vector<Ctext> mult_results(kernelSq);

            for (int inCh = 0; inCh < inputChannels; inCh++) {
                auto encodeKernel = fheonHEController.encode_kernel_optimized(rawKernelData[outCh][inCh], inputSize);
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
    };

    // Divide output channels among threads
    block = (outputChannels + outnumThreads - 1) / outnumThreads;
    for (int t = 0; t < outnumThreads; t++) {
        int start = t * block;
        int end = min(start + block, outputChannels);
        outThreads[t] = thread(worker, start, end);
    }
    for (auto &thout : outThreads) thout.join();

    batched_rotated_ciphertexts.clear();
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
 * @param kernelData        Convolution kernels represented as a 3D vector of
 *                          plaintexts: [outputChannel][inputChannel][kernelWeights].
 * @param biasInputs        Bias terms for each output channel (plaintexts).
 * @param batchSize         Number of batches encoded in the input ciphertexts.
 * @param inputWidth        Width (and height) of the input feature maps (assumed square).
 * @param inputChannels     Number of input channels.
 * @param outputChannels    Number of output channels.
 * @param stride          stride length used for the convolution operation.
 *
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
vector<Ctext> FHEONANNBatchController::he_batch_convolution_shortcut_optimized(
                            FHEONHEController &fheonHEController, vector<Ctext>& encryptedInputs, 
                            vector<vector<vector<double>>>& rawKernelData, vector<vector<double>>& rawBiasData,
                            int batchSize, int inputWidth, int inputChannels, int outputChannels, int stride){
    
    int outputWidth = inputWidth / stride;
    int inputSize = inputWidth * inputWidth;
    int outputSize = outputWidth * outputWidth;
    // int encodeLevel = encryptedInputs[0]->GetLevel();
    int inVecSize = batchSize * inputSize;
    int outVecSize = batchSize * outputSize;
    vector<double> cleaningoutputVec = generate_mixed_mask(outVecSize, inVecSize);
    vector<Ctext> output_ciphertexts(outputChannels); 
 
    int numThreads = min(outputChannels, (int)thread::hardware_concurrency());
    vector<thread> threads(numThreads);
           
    // Lambda for per-thread work
    auto worker = [&](int start, int end) {
        for (int outCh = start; outCh < end; outCh++) {
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
    };

     // Divide output channels among threads
    int block = (outputChannels + numThreads - 1) / numThreads;
    for (int t = 0; t < numThreads; t++) {
        int start = t * block;
        int end = min(start + block, outputChannels);
        threads[t] = thread(worker, start, end);
    }

    for (auto &th : threads) th.join();

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

vector<Ctext> FHEONANNBatchController::he_batch_sum_ciphertexts(
    vector<Ctext>& firstEncryptedInputs,
    vector<Ctext>& secondEncryptedInputs,
    int inputChannels
) {

    vector<Ctext> summed_ciphertexts(inputChannels);

    int hwThreads = thread::hardware_concurrency();
    int numThreads = min(inputChannels, hwThreads > 0 ? hwThreads : 4);
    int block = (inputChannels + numThreads - 1) / numThreads;

    vector<thread> threads;
    threads.reserve(numThreads);
    auto worker = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            summed_ciphertexts[i] = context->EvalAdd(firstEncryptedInputs[i], secondEncryptedInputs[i]);
        }
    };

    for (int t = 0; t < numThreads; ++t) {
        int start = t * block;
        int end = min(start + block, inputChannels);
        if (start < end){
            threads.emplace_back(worker, start, end);
        }
    }
    for (auto &th : threads) th.join();

    return summed_ciphertexts;
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
 * The computation is parallelized across channels using multiple threads.
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

vector<Ctext> FHEONANNBatchController::he_batch_globalpool(vector<Ctext>& encryptedInputs, int batchSize, int inputWidth, 
                        int inputChannels, int kernelWidth, int rotatePositions){
    
    int inputSize = inputWidth*inputWidth;
    // auto reducingMask =  context->MakeCKKSPackedPlaintext(generate_scale_mask(inputSize, batchSize), 1, encryptedInputs[0]->GetLevel());
    vector<double> reducingMask = generate_scale_mask(inputSize, batchSize);
    vector<Ctext> pooled_ciphertexts(inputChannels);

    int hwThreads = thread::hardware_concurrency();
    int numThreads = min(inputChannels, hwThreads > 0 ? hwThreads : 4);
    // Only create real threads (never invalid placeholders)

    // threads.reserve(numThreads);
    auto worker = [&](int start, int end) {
        for(int i = start; i< end; i++){
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
    };

    vector<thread> threads(numThreads);
    int block = (inputChannels + numThreads - 1) / numThreads;
    for (int t = 0; t < numThreads; t++) {
        int start = t * block;
        int end = std::min(start + block, inputChannels);
        threads[t] = std::thread(worker, start, end);
    }

    for (auto &th : threads) th.join();
    return pooled_ciphertexts; 
}


//  Ctext FHEONANNBatchController::he_batch_linear(
//     Ctext& encryptedInput, vector<Ptext>& weightMatrix,
//     Ptext& baisInput, int batchSize, int inputSize,
//     int outputSize, int rotatePositions){
    
//     int totalSize = batchSize*inputSize;
//     vector<Ctext> result_matrix;
//     vector<vector<Ctext>> batched_ciphertexts(batchSize);
//     vector<double> cleaningInVec = generate_mixed_mask(inputSize, totalSize);
    
//     int j = 0;
//     int rotation_index = 0;
//     for(int i=0; i<outputSize; i++){
//         // For each batch element, rotate base and sum -> push into that batch's vector
//         Ctext temp = context->EvalMult(encryptedInput, weightMatrix[i]);
//         for (int b = 0; b < batchSize; b++) {
//             if (b != 0) {
//                 temp = context->EvalRotate(temp, inputSize);
//             }
//             Ctext summed = context->EvalSum(temp, inputSize);
//             // Ctext summed = context->EvalSum(context->EvalMult(temp, context->MakeCKKSPackedPlaintext(cleaningInVec, 1, temp->GetLevel())), inputSize);
//             batched_ciphertexts[b].push_back(summed);
//         }
//         // cout << "outputSize: " << i << " -- Rotate: " << inputSize << endl;

//         if ((j == rotatePositions - 1) || (i == outputSize - 1)) {
//             for (int b = 0; b < batchSize; b++) {
//                 // cout << "batched_ciphertexts[" << b << "] size: " << batched_ciphertexts[b].size() << " --batch Rotate: " << -(b * outputSize) << " --rotation_index "<< rotation_index << endl;
//                 Ctext inn_ciphertext = context->EvalMerge(batched_ciphertexts[b]);
//                 // cout << "After Merge" << endl; 
//                 // Apply rotations to place merged block into the correct global slot
//                 if (b != 0) {
//                     // cout << "b " << b << ": " << (b * outputSize) << endl;
//                     inn_ciphertext = context->EvalRotate(inn_ciphertext, -(b * outputSize));
//                 }
//                 if (rotation_index != 0) {
//                     inn_ciphertext = context->EvalRotate(inn_ciphertext, -rotation_index);
//                 }
//                 result_matrix.push_back(inn_ciphertext);
//                 batched_ciphertexts[b].clear();
//             }
//             if(j == rotatePositions - 1){
//                 rotation_index += rotatePositions;
//             }
//             j = 0;
//         } else {
//             j++;
//         }
//     } 
//     // Combine results and add bias
//     Ctext with_bias = context->EvalAdd(context->EvalAddMany(result_matrix), baisInput);
//     result_matrix.clear();
//     batched_ciphertexts.clear();

//     return with_bias;
// }

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
 * The computation is parallelized across output neurons and batch elements
 * using multiple threads.
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

Ctext FHEONANNBatchController::he_batch_linear(
    Ctext& encryptedInput, vector<Ptext>& weightMatrix,
    Ptext& baisInput, int batchSize, int inputSize,
    int outputSize, int rotationIndex){
    
    vector<Ctext> result_matrix(batchSize);
    vector<vector<Ctext>> batched_ciphertexts(batchSize, vector<Ctext>(outputSize));

    int hwThreads = thread::hardware_concurrency();
    int numThreads = min(outputSize, hwThreads > 0 ? hwThreads : 4);
    auto worker = [&](int start, int end) {
        for(int i = start; i< end; i++){
        // For each batch element, rotate base and sum -> push into that batch's vector
            Ctext rotated_temp = context->EvalMult(encryptedInput, weightMatrix[i]);
            for (int b = 0; b < batchSize; b++) {
                if (b != 0) {
                    rotated_temp = context->EvalRotate(rotated_temp, inputSize);
                }
                batched_ciphertexts[b][i] = context->EvalSum(rotated_temp, inputSize);
            }
        }
    };
    
    vector<thread> threads(numThreads);
    int block = (outputSize + numThreads - 1) / numThreads;
    for (int t = 0; t < numThreads; t++) {
        int start = t * block;
        int end = std::min(start + block, outputSize);
        threads[t] = std::thread(worker, start, end);
    }
    for (auto &th : threads) th.join();

    // cout << "Inputs rows all calculated" << endl; 
    int numBatchThreads = min(batchSize, hwThreads > 0 ? hwThreads : 4);

    auto batchworker = [&](int bstart, int bend) {
        for(int b = bstart; b < bend; b++){
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
    };

    vector<thread> bthreads(numBatchThreads);
    int bblock = (batchSize + numBatchThreads - 1) / numBatchThreads;
    for (int t = 0; t < numBatchThreads; t++) {
        int bstart = t * bblock;
        int bend = std::min(bstart + bblock, batchSize);
        bthreads[t] = std::thread(batchworker, bstart, bend);
    }
    for (auto &th : bthreads) th.join();

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
 * The computation is parallelized across output neurons and batch elements
 * using multiple threads.
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

vector<Ctext> FHEONANNBatchController::he_batch_linear_multiple_outputs(
    Ctext& encryptedInput, vector<Ptext>& weightMatrix,
    Ptext& baisInput, int batchSize, int inputSize, int outputSize){
    
    vector<vector<Ctext>> batched_ciphertexts(batchSize, vector<Ctext>(outputSize));

    int hwThreads = thread::hardware_concurrency();
    int numThreads = min(outputSize, hwThreads > 0 ? hwThreads : 4);
    auto worker = [&](int start, int end) {
        for(int i = start; i< end; i++){
            Ctext temp = context->EvalMult(encryptedInput, weightMatrix[i]);
            for (int b = 0; b < batchSize; b++) {
                if (b != 0) {
                    temp = context->EvalRotate(temp, inputSize);
                }
                batched_ciphertexts[b][i] = context->EvalSum(temp, inputSize);
            }
        }
    };
    
    vector<thread> threads(numThreads);
    int block = (outputSize + numThreads - 1) / numThreads;
    for (int t = 0; t < numThreads; t++) {
        int start = t * block;
        int end = std::min(start + block, outputSize);
        threads[t] = std::thread(worker, start, end);
    }
    for (auto &th : threads) th.join();


    weightMatrix.clear();
    weightMatrix.shrink_to_fit();

    
    vector<Ctext> result_matrix(batchSize);
    int numBatchThreads = min(batchSize, hwThreads > 0 ? hwThreads : 4);
    auto batchworker = [&](int bstart, int bend) {
        for(int b = bstart; b < bend; b++){
            result_matrix[b] = context->EvalAdd(context->EvalMerge(batched_ciphertexts[b]), baisInput);
        }
    };

    vector<thread> bthreads(numBatchThreads);
    int bblock = (batchSize + numBatchThreads - 1) / numBatchThreads;
    for (int t = 0; t < numBatchThreads; t++) {
        int bstart = t * bblock;
        int bend = std::min(bstart + bblock, batchSize);
        bthreads[t] = std::thread(batchworker, bstart, bend);
    }
    for (auto &th : bthreads) th.join();

    // Combine results and add bias
    batched_ciphertexts.clear();
    return result_matrix;
}



// vector<Ctext> FHEONANNBatchController::he_batch_linear_multiple_outputs(
    
//     Ctext& encryptedInput, vector<Ptext>& weightMatrix,
//     Ptext& baisInput, int batchSize, 
//     int inputSize, int outputSize){
    
//     vector<Ctext> result_matrix(batchSize);

//     int hwThreads = thread::hardware_concurrency();
//     int numThreads = min(outputSize, hwThreads > 0 ? hwThreads : 4);

//     // Process batches in smaller chunks to reduce memory usage
//     int chunkSize = 128; // Process 128 batches at a time
//     for (int chunkStart = 0; chunkStart < batchSize; chunkStart += chunkSize) {
//         int chunkEnd = std::min(chunkStart + chunkSize, batchSize);
//         int currentChunkSize = chunkEnd - chunkStart;

//         vector<vector<Ctext>> batched_ciphertexts(currentChunkSize, vector<Ctext>(outputSize));
//         auto worker = [&](int start, int end) {
//             for (int i = start; i < end; i++) {
//                 int idxW = chunkStart +i;
//                 Ctext temp = context->EvalMult(encryptedInput, weightMatrix[idxW]);
//                 if( chunkStart != 0 ){
//                     // Rotate to the start of the current chunk
//                     temp = context->EvalRotate(temp, chunkStart * inputSize);
//                 }
//                 for (int b = 0; b < currentChunkSize; b++) {
//                     if (b != 0) {
//                         temp = context->EvalRotate(temp, inputSize);
//                     }
//                     batched_ciphertexts[b][i] = context->EvalSum(temp, inputSize);
//                 }
//             }
//         };

//         vector<thread> threads(numThreads);
//         int block = (outputSize + numThreads - 1) / numThreads;
//         for (int t = 0; t < numThreads; t++) {
//             int start = t * block;
//             int end = std::min(start + block, outputSize);
//             threads[t] = std::thread(worker, start, end);
//         }
//         for (auto &th : threads) th.join();

//         cout << "Inputs rows for chunk [" << chunkStart << ", " << chunkEnd << ") all calculated" << endl;

//         auto batchworker = [&](int bstart, int bend) {
//             for (int b = bstart; b < bend; b++) {
//                 int indexInResult = chunkStart + b;
//                 result_matrix[indexInResult] =  context->EvalAdd(context->EvalMerge(batched_ciphertexts[b]), baisInput);
//             }
//         };

//         int numBatchThreads = min(currentChunkSize, hwThreads > 0 ? hwThreads : 4);
//         vector<thread> bthreads(numBatchThreads);
//         int bblock = (currentChunkSize + numBatchThreads - 1) / numBatchThreads;
//         for (int t = 0; t < numBatchThreads; t++) {
//             int bstart = t * bblock + chunkStart;
//             int bend = std::min(bstart + bblock, chunkEnd);
//             bthreads[t] = std::thread(batchworker, bstart, bend);
//         }
//         for (auto &th : bthreads) th.join();

//         batched_ciphertexts.clear(); // Clear memory for this chunk
//     }

//     return result_matrix;
// }



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

vector<Ctext> FHEONANNBatchController::he_batch_relu(vector<Ctext>& encryptedInputs, vector<int> scaleValues,  int inputChannels, int vectorSize, int polyDegree) {
    double lowerBound = -1;
    double upperBound = 1;
    
    auto encryptInn = encryptedInputs;
    // int scaleVal = 110;
    vector<Ctext> relu_results(inputChannels);
    int totalElements =  nextPowerOf2(vectorSize);
    int numThreads = std::min(inputChannels, (int)std::thread::hardware_concurrency());
    std::vector<std::thread> threads(numThreads);

    // Lambda for per-thread work
    auto worker = [&](int start, int end) {
        for(int i=start; i<end; i++){
            int scaleVal = scaleValues[i];
            if(scaleVal <= 1){
                encryptInn[i] = encryptedInputs[i];
                scaleVal = 1;
            }
            else{
                scaleVal = 2*scaleVal;
                // auto mask_data = context->MakeCKKSPackedPlaintext(generate_scale_mask(scaleVal, vectorSize), 1, 1, nullptr, totalElements);
                auto mask_data = context->MakeCKKSPackedPlaintext(generate_scale_mask(scaleVal, vectorSize), 1, encryptedInputs[i]->GetLevel(), nullptr, totalElements);
                encryptInn[i] = context->EvalMult(encryptedInputs[i], mask_data);
            }
            relu_results[i] = context->EvalChebyshevFunction(
                            [scaleVal](double x) -> double { if (x < 0) return 0; else return scaleVal*x; }, 
                                                encryptInn[i],
                                                lowerBound,
                                                upperBound, 
                                                polyDegree);
        }
    };

     // Divide output channels among threads
    int block = (inputChannels + numThreads - 1) / numThreads;
    for (int t = 0; t < numThreads; t++) {
        int start = t * block;
        int end = std::min(start + block, inputChannels);
        threads[t] = std::thread(worker, start, end);
    }

    for (auto &th : threads) th.join();

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
 * @param inputSize Size of each input channel (number of slots used per channel).
 *
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
Ctext FHEONANNBatchController::he_batch_inputs_converter(vector<Ctext>& encryptedInputs, int batchSize, int inputChannels, int inputWidth) {
    
    int inputSize = pow(inputWidth, 2); 
    int batchKey = (inputChannels * inputSize);
    // int inVecSize = batchSize * inputSize; // total packed vector size
    // auto cleaningMask = context->MakeCKKSPackedPlaintext(generate_mixed_mask(inputSize, inVecSize), 1, encryptedInputs[0]->GetLevel());
    vector<Ctext> batch_ciphertexts(batchSize);
    for (int b = 0; b < batchSize; b++) {
        for (int i = 0; i < inputChannels; i++) {
            if(b != 0){
                encryptedInputs[i] = context->EvalRotate(encryptedInputs[i], inputSize);
            }
        }
        
        // Combine all channels for this batch into one ciphertext
        Ctext combinedChannels = context->EvalMerge(encryptedInputs);
        if (b != 0) {
            int rot_idx = (b * batchKey);
            int base_idx = (rot_idx / baseIndex) * baseIndex;
            int mod_idx = rot_idx % baseIndex; 
            if(base_idx != 0){
                combinedChannels = context->EvalRotate(combinedChannels, -base_idx);
            }
            if(mod_idx !=0){
                combinedChannels = context->EvalRotate(combinedChannels, -mod_idx);
            }
        }
        batch_ciphertexts[b] = combinedChannels;
    }

    // Combine all batch ciphertexts into the final output
    Ctext finalCipher = context->EvalAddMany(batch_ciphertexts);
    batch_ciphertexts.clear();
    return finalCipher;
}


/**
 * @brief Generate a zero mask of given size.
 *
 * @param size Number of elements in the mask.
 * @param level Encryption level for CKKS plaintext.
 * @return Packed plaintext mask with all zeros.
 */
Ptext FHEONANNBatchController::gen_zero_mask(int size, int level) {
    vector<double> mask(size, 0.0);
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}


/**
 * @brief Generate a mask selecting a specific row in every channel.
 *
 * @param row Row index to select.
 * @param width Width of each channel.
 * @param inputSize Number of elements per channel.
 * @param stride Unused here but kept for consistency.
 * @param numChannels Total number of channels.
 * @param level Encryption level for CKKS plaintext.
 * @return Packed plaintext mask with the row selected in all channels.
 */
Ptext FHEONANNBatchController::gen_row_mask_with_channels(int row, int width, int inputSize, int batchSize, int level) {
    
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
Ptext FHEONANNBatchController::gen_channel_mask_with_zeros(int channel, int outputSize, int numChannels, int level ){
    
    int totalSlots = outputSize * numChannels;
    vector<double> mask(totalSlots, 0.0);

    int pos = channel * outputSize;
    for (int i = 0; i < outputSize; i++) {
        mask[pos + i] = 1.0;
    }
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}

