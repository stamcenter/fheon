
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
 * @brief FHE Controller for defining and managing homomorphic ANN functions.
 *
 * This class provides different methods for convolution, pooling, fully connected
 *  relu, etc for neural network development on encrypted data using FHE.
 */

#include <fstream>
#include <filesystem>
#include <iostream>
#include <cmath>
#include <thread>
#include "FHEONANNController.h"

namespace fs = std::filesystem;

/**
 * @brief Set the global crypto context.
 *
 * @param in_context The crypto context to set.
 */
void FHEONANNController::setContext(CryptoContext<DCRTPoly>& in_context){
    context = in_context;
}

/**
 * @brief Generate rotation index positions needed for standard convolution.
 *
 * @param inputWidth Dimensions of input channel.
 * @param inputChannels Number of input channels.
 * @param outputChannels Number of output channels.
 * @param kernelWidth Dimension of square kernel.
 * @param padding Size of zero padding.
 * @param stride Convolution stride.
 * @return List of unique rotation steps.
 */
vector <int> FHEONANNController::generate_convolution_rotation_positions(int inputWidth, int inputChannels, int outputChannels,
                                                    int kernelWidth, int padding, int stride){
        
    vector<int> keys_position;
    int inputWidth_sq = pow(inputWidth, 2);
    int padded_width = inputWidth+(2*padding);
    int padding_width_sq = pow(padded_width, 2);
    int width_out = ((padded_width - (kernelWidth - 1) - 1)/stride)+1;
    int width_out_sq = pow(width_out,2);
    keys_position.push_back(inputWidth);
    keys_position.push_back(padded_width);
    keys_position.push_back(padding_width_sq);
    keys_position.push_back(inputWidth_sq);
    keys_position.push_back(width_out);
    keys_position.push_back(width_out_sq);
     keys_position.push_back(-1);
    keys_position.push_back(1);
    int rot_val;

    /** Convolution rotations */
    for(int i=1; i < kernelWidth;i++){
        keys_position.push_back(i);
    }

    for(int i=1; i<width_out; i++){
        rot_val = (i*width_out);
        keys_position.push_back(-rot_val);
    }
    for(int i=1; i<outputChannels; i++){
        rot_val = (i*width_out_sq);
        keys_position.push_back(-rot_val);
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
 * @brief Generate rotation index positions needed for average pooling.
 *
 * @param inputWidth Dimension of input.
 * @param kernelWidth Dimension of pooling window.
 * @param stride Pooling stride.
 * @param inputChannels Number of channels.
 * @return List of unique rotation steps.
 */
vector <int> FHEONANNController::generate_avgpool_rotation_positions(int inputWidth, int kernelWidth,
                int stride, int inputChannels){
    
    vector<int> keys_position;
    int width_avgpool_out = (inputWidth/stride);
    int width_avgpool_sq = pow(width_avgpool_out, 2); 
    int width_sq = pow(inputWidth, 2);
    keys_position.push_back(width_sq);
    keys_position.push_back(inputWidth);
    keys_position.push_back(kernelWidth);
    keys_position.push_back(stride);
    keys_position.push_back(width_avgpool_out);
    keys_position.push_back((stride*inputWidth));

    for(int i=1; i<inputChannels; i++){
        int rot_val = i*width_avgpool_out;
        keys_position.push_back(-rot_val);

        rot_val = i*width_avgpool_sq;
        keys_position.push_back(-rot_val);
    }

    for(int i =1; i<width_avgpool_out; i++){
        int rot_val = i*width_avgpool_out;
        keys_position.push_back(i);
        keys_position.push_back(-rot_val);
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
 * @brief Generate rotation positions optimized for convolution.
 *
 * @param inputWidth Dimension of input.
 * @param inputChannels Number of input channels.
 * @param outputChannels Number of output channels.
 * @param stride Convolution stride.
 * @param stridingType Type of striding optimization.
 * @return List of unique rotation steps.
 */
vector <int> FHEONANNController::generate_optimized_convolution_rotation_positions(
    int inputWidth,  int inputChannels, int outputChannels, int stride, string stridingType){
        
    vector<int> keys_position;
    int inputWidth_sq = pow(inputWidth, 2);
    int width_out = (inputWidth/stride);
    int width_out_sq = pow(width_out,2);
    keys_position.push_back(-1);
    keys_position.push_back(1);
    keys_position.push_back(inputWidth_sq);
    keys_position.push_back(inputWidth);
    keys_position.push_back(-inputWidth);

    if(stride > 1){
        if(stridingType == "basic"){
            /**** USE THIS BLOCK FOR BASIC STRIDING */
            for(int i=1; i < inputChannels; i++){
                int rot_val = i*width_out;
                keys_position.push_back(-rot_val);

                rot_val = i*width_out_sq;
                keys_position.push_back(-rot_val);
            }
            for(int i =1; i< width_out; i++){
                int rot_val = i*width_out;
                keys_position.push_back(i);
                keys_position.push_back(-rot_val);
            }
        }
        else if(stridingType == "single_channel"){
            for (int s=1; s<log2(width_out); s++) {
                keys_position.push_back( pow(2, s-1));
            }
            keys_position.push_back(pow(2, log2(width_out)-1));
            int rotAmount = (stride * inputWidth - width_out);
            keys_position.push_back(rotAmount);

            int shift = (inputWidth_sq - width_out_sq)* ((outputChannels / stride) - 1);
            keys_position.push_back(-shift);

            shift = -(inputWidth_sq - width_out_sq);
            keys_position.push_back(shift);

            for(int i=1; i< outputChannels; i++){
                int rotateAmount = - i * width_out_sq;
                keys_position.push_back(rotateAmount);
            }
        }
        else if(stridingType == "multi_channels"){
            for(int i=1; i<inputChannels; i++){
                int rot_val = (i*inputWidth_sq);
                keys_position.push_back(-rot_val);
            }

            for (int s=1; s<log2(width_out); s++) {
                keys_position.push_back( pow(2, s-1));
            }
            keys_position.push_back(pow(2, log2(width_out)-1));
            int rotAmount = (stride * inputWidth - width_out);
            keys_position.push_back(rotAmount);

            int shift = (inputWidth_sq - width_out_sq)* ((outputChannels / stride) - 1);
            keys_position.push_back(-shift);

            shift = -(inputWidth_sq - width_out_sq);
            keys_position.push_back(shift);

            shift = (inputWidth_sq - width_out_sq);
            keys_position.push_back(shift);

            int rotateAmount = - inputChannels * width_out_sq;
            keys_position.push_back(rotateAmount);

            // realign
            int outputSlots = outputChannels * width_out_sq;
            if (num_slots < outputSlots) {
                int j =0;
                for(int i=0; i< outputChannels; i+=inputChannels){
                    j++;
                    int rotateAmount = - j * (inputChannels * width_out_sq);
                    keys_position.push_back(rotateAmount);
                }
            }
        }
    }
    else{
        for(int i=1; i<outputChannels; i++){
            int rot_val = (i*width_out_sq);
            keys_position.push_back(-rot_val);
        }
    }

     std::sort(keys_position.begin(), keys_position.end());
    keys_position.erase(std::unique(keys_position.begin(), keys_position.end()), keys_position.end());
    keys_position.erase(std::remove(keys_position.begin(), keys_position.end(), 0), keys_position.end());
    return keys_position;
}


/**
 * @brief Generate rotation positions optimized for average pooling.
 *
 * @param inputWidth Dimension of input.
 * @param inputChannels Number of channels.
 * @param kernelWidth Dimension of pooling window.
 * @param stride Pooling stride.
 * @param globalPooling Whether it's global average pooling.
 * @param stridingType Type of striding optimization.
 * @param rotationIndex Base rotation index.
 * @return List of unique rotation steps.
 */
vector <int> FHEONANNController::generate_avgpool_optimized_rotation_positions(int inputWidth,  int inputChannels, 
                                        int kernelWidth, int stride, bool globalPooling, string stridingType, int rotationIndex){
    
    vector<int> keys_position;
    if(globalPooling){
        keys_position.push_back((inputWidth*inputWidth));
        keys_position.push_back(-inputChannels);
        for(int pos=0; pos<inputChannels; pos+=rotationIndex){
            keys_position.push_back(-pos);
        }
        for(int i=1; i<=rotationIndex; i++){
            keys_position.push_back(i);
        }
        return keys_position;
    }

    int width_avgpool_out = (inputWidth/stride);
    int width_avgpool_sq = pow(width_avgpool_out, 2); 
    int width_sq = pow(inputWidth, 2);
    keys_position.push_back(width_sq);
    keys_position.push_back(inputWidth);
    keys_position.push_back(stride);
    keys_position.push_back(width_avgpool_out);
    keys_position.push_back(width_avgpool_sq);
    keys_position.push_back((stride*inputWidth));

    if(inputWidth<=2){
        for(int pos=0; pos<inputChannels; pos++){
            keys_position.push_back(pos);
        }
        return keys_position;
    }

    if(stride>1){
        if(stridingType == "basic"){
            /**** USE THIS BLOCK FOR BASIC STRIDING */
            for(int i=1; i < inputChannels; i++){
                int rot_val = i*width_avgpool_out;
                keys_position.push_back(-rot_val);

                rot_val = i*width_avgpool_sq;
                keys_position.push_back(-rot_val);
            }
            for(int i =1; i< width_avgpool_out; i++){
                int rot_val = i*width_avgpool_out;
                keys_position.push_back(i);
                keys_position.push_back(-rot_val);
            }
        }
        else if(stridingType == "single_channel"){
            for (int s=1; s<log2(width_avgpool_out); s++) {
                keys_position.push_back( pow(2, s-1));
            }
            keys_position.push_back(pow(2, log2(width_avgpool_out)-1));
            int rotAmount = (stride * inputWidth - width_avgpool_out);
            keys_position.push_back(rotAmount);
    
            rotAmount = (width_sq - width_avgpool_sq)* ((inputChannels / stride) - 1);
            keys_position.push_back(-rotAmount);

            rotAmount = (width_sq - width_avgpool_sq);
            keys_position.push_back(rotAmount);

            for(int i=1; i< inputChannels; i++){
                int rotateAmount = - i * width_avgpool_sq;
                keys_position.push_back(rotateAmount);
            }
        }
        else if(stridingType == "multi_channels"){
            for(int i=1; i<inputChannels; i++){
                int rot_val = (i*width_sq);
                keys_position.push_back(-rot_val);
            }

            for (int s=1; s<log2(width_avgpool_out); s++) {
                keys_position.push_back( pow(2, s-1));
            }
            keys_position.push_back(pow(2, log2(width_avgpool_out)-1));
            int rotAmount = (stride * inputWidth - width_avgpool_out);
            keys_position.push_back(rotAmount);

            int shift = (width_sq - width_avgpool_sq)* ((inputChannels / stride) - 1);
            keys_position.push_back(-shift);

            shift = -(width_sq - width_avgpool_sq);
            keys_position.push_back(shift);

            shift = (width_sq - width_avgpool_sq);
            keys_position.push_back(shift);

            int rotateAmount = - inputChannels * width_avgpool_sq;
            keys_position.push_back(rotateAmount);
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
 * @brief Generate rotation positions for fully connected linear layers.
 *
 * @param maxFCLayeroutputs Maximum output size.
 * @param rotationPositions Base rotation steps.
 * @return List of unique rotation steps.
 */
vector <int> FHEONANNController::generate_linear_rotation_positions(int maxFCLayeroutputs, int rotationPositions){
    
    vector<int> keys_position;

    for(int counter=0; counter<maxFCLayeroutputs; counter+=rotationPositions){
        keys_position.push_back(-counter);
    }
    
    for(int i=1; i<=rotationPositions; i++){
        //  keys_position.push_back(i);
         keys_position.push_back(-i);
    }
    std::sort(keys_position.begin(), keys_position.end());
    auto new_end = std::remove(keys_position.begin(), keys_position.end(), 0);
    new_end = std::unique(keys_position.begin(), keys_position.end());
    unique(keys_position.begin(), keys_position.end());
    keys_position.erase(new_end, keys_position.end());
    keys_position.erase(std::remove(keys_position.begin(), keys_position.end(), 0), keys_position.end());
    std::sort(keys_position.begin(), keys_position.end());
    return keys_position;
}

/**
 * @brief Homomorphic standard convolution layer.
 *
 * @param encryptedInput Encrypted input ciphertext.
 * @param kernelData Encoded weights filter matrix.
 * @param biasInput Encoded bias vector.
 * @param inputWidth Dimension of input.
 * @param inputChannels Number of input channels.
 * @param outputChannels Number of output channels.
 * @param kernelWidth Dimension of square kernel.
 * @param paddingLen Padding size.
 * @param stride Stride size.
 * @return Encrypted convolution output.
 */
Ctext FHEONANNController::he_convolution(Ctext& encryptedInput, vector<vector<Ptext>>& kernelData, Ptext& biasInput,
        int inputWidth,  int inputChannels, int outputChannels,  int kernelWidth, int paddingLen, int stride) {
    auto start_time = startTime();

    int kernelSq = kernelWidth * kernelWidth;
    int inputSize = inputWidth * inputWidth;
    int outputWidth = ((inputWidth - kernelWidth) / stride) + 1;
    int outputSize = outputWidth * outputWidth;
    int encode_level = encryptedInput->GetLevel();

    // STEP 1 - Generate mixed mask for cleaning multi-channel inputs
    int zero_elements = inputSize * (inputChannels - 1);
    if(inputChannels  < 2){
        zero_elements = inputSize;
    }
    vector<double> mixed_mask = generate_mixed_mask(inputSize, zero_elements);
    Ptext cleaning_mask = context->MakeCKKSPackedPlaintext(mixed_mask, 1, encode_level);

    vector<double> mixed_mask_out = generate_mixed_mask(outputWidth, zero_elements);
    Ptext cleaning_mask_out = context->MakeCKKSPackedPlaintext(mixed_mask_out, 1, encode_level);

    // STEP 2 - ROTATE INPUT TO FORM k^2 slices
    vector<Ctext> rotated_ciphertexts;
    for (int i = 0; i < kernelWidth; i++) {
        if(i >0){
            encryptedInput = context->EvalRotate(encryptedInput, inputWidth);
        }
        rotated_ciphertexts.push_back(encryptedInput);
        for (int j = 1; j < kernelWidth; j++) {
            rotated_ciphertexts.push_back(context->EvalRotate(encryptedInput, j));
        }
    }

    // STEP 3-6 - Convolution over all output channels
    Ctext strided_cipher;
    vector<Ctext> final_vec;
    for (int out_ch = 0; out_ch < outputChannels; out_ch++) {
        vector<Ctext> mult_results;

        // Per-kernel value multiplies
        for (int k = 0; k < kernelSq; k++) {
            mult_results.push_back(context->EvalMult(rotated_ciphertexts[k], kernelData[out_ch][k]));
        }

        Ctext conv_sum = context->EvalAddMany(mult_results);

        // STEP 4 - Sum all input channels (rotating and adding)
        if (inputChannels > 1) {
            vector<Ctext> channel_sums = { conv_sum };
            for (int ch = 1; ch < inputChannels; ch++) {
                conv_sum = context->EvalRotate(conv_sum, inputSize);
                channel_sums.push_back(conv_sum);
            }
            conv_sum = context->EvalAddMany(channel_sums);
        }
        conv_sum = context->EvalMult(conv_sum, cleaning_mask);

        // STEP 5 - Striding
        if (stride > 1) {
            strided_cipher = downsample(conv_sum, inputWidth, stride);
        } 
        else {
            vector<Ctext> strided_vec;
            for (int l = 0; l < outputWidth; l++) {
                Ctext cleaned_cipher;
                if (l == 0) {
                    cleaned_cipher = context->EvalMult(conv_sum, cleaning_mask_out);
                } else {
                    conv_sum = context->EvalRotate(conv_sum, inputWidth); 
                    cleaned_cipher = context->EvalRotate(context->EvalMult(conv_sum, cleaning_mask_out), -(outputWidth * l));
                }
                strided_vec.push_back(cleaned_cipher);
            }
            strided_cipher = context->EvalAddMany(strided_vec);
        }

        // STEP 7 - Rotate for output layout reconstruction
        if (out_ch == 0) {
            final_vec.push_back(strided_cipher);
        } else {
            final_vec.push_back(context->EvalRotate(strided_cipher, -(out_ch * outputSize)));
        }
    }
    Ctext res = context->EvalAdd(context->EvalAddMany(final_vec), biasInput);
    final_vec.clear();
    rotated_ciphertexts.clear();
    printTiming("Convolution evaluated...", "Conv", start_time);
    return res;
}

/**
 * @brief Homomorphic convolution with advanced padding and stride configurations.
 *
 * @param encryptedInput Encrypted input ciphertext.
 * @param kernelData Pre-encoded filter weight vectors.
 * @param biasInput Pre-encoded bias input vector.
 * @param inputWidth Dimensions of the input channels.
 * @param inputChannels Number of input channels.
 * @param outputChannels Number of output filter channels.
 * @param kernelWidth Dimension of the square convolution kernel.
 * @param padding Size of zero padding to apply.
 * @param stride Convolution stride.
 * @return Encrypted output of the convolution layer.
 */
Ctext FHEONANNController::he_convolution_advanced(Ctext& encryptedInput, vector<vector<Ptext>>& kernelData, Ptext& biasInput, 
            int inputWidth,  int inputChannels, int outputChannels, int kernelWidth, int padding, int stride){
    auto start_time = startTime();

    /** If padding is 0 */    
    if(padding ==0){
        auto conv_basic_cipher = he_convolution(encryptedInput, kernelData, biasInput, inputWidth, 
                                            kernelWidth, inputChannels, outputChannels, padding, stride);
        return conv_basic_cipher;
    }
    int padded_width = inputWidth + (2*padding);
    int padded_width_sq = pow(padded_width, 2);
    int width_sq = pow(inputWidth, 2);
    int zeros_elements = ((inputChannels*width_sq) - inputWidth);
    int encode_level = encryptedInput->GetLevel();
    auto padding_mix_mask = generate_mixed_mask(inputWidth, zeros_elements);
    Ptext in_clean_mask =  context->MakeCKKSPackedPlaintext(padding_mix_mask, 1, encode_level);

    /** generate vector of padding width */
    Ctext channel_cipher = encryptedInput;
    vector<Ctext> channel_vector_ciphers;
    for(int i = 0; i< inputChannels; i++){
        if(i!=0){
            channel_cipher = context->EvalRotate(channel_cipher, width_sq);
        }
        vector<Ctext> in_chan_vec;
        Ctext in_chan_cipher = channel_cipher;
        for(int k=0; k<inputWidth; k++){
            Ctext in_clean_cipher = context->EvalMult(in_chan_cipher, in_clean_mask);
            in_chan_cipher =  context->EvalRotate(in_chan_cipher, inputWidth);
            if(k==0){
                in_chan_vec.push_back(in_clean_cipher);
            }
            else{
                int in_rot_position = k*padded_width;
                Ctext padded_cipher = context->EvalRotate(in_clean_cipher, -in_rot_position);
                in_chan_vec.push_back(padded_cipher);
            }
        }
        Ctext in_sum_cipher = context->EvalAddMany(in_chan_vec);
        if(i==0){
            channel_vector_ciphers.push_back(in_sum_cipher);
        }
        else{
            int in_rotate = i*padded_width_sq;
            Ctext in_rotate_cipher = context->EvalRotate(in_sum_cipher, -in_rotate);
            channel_vector_ciphers.push_back(in_rotate_cipher);
        }
    }
    int padd_extra = (padding*padded_width)+padding;
    Ctext padded_cipher = context->EvalAddMany(channel_vector_ciphers);
    if(padd_extra != 0){
        padded_cipher = context->EvalRotate(padded_cipher, -padd_extra);
    }
    Ctext conv_basic_cipher = he_convolution(padded_cipher, kernelData, biasInput, inputWidth, kernelWidth, 
                                             inputChannels, outputChannels, padding, stride); 
    printTiming("Advanced convolution evaluated...", "Conv", start_time);
    return conv_basic_cipher;
}

/**
 * @brief Optimized homomorphic convolution using parallel channels.
 *
 * @param encryptedInput Encrypted input ciphertext.
 * @param kernelData Pre-encoded filter weight vectors.
 * @param biasInput Pre-encoded bias input vector.
 * @param inputWidth Dimensions of the input channels.
 * @param inputChannels Number of input channels.
 * @param outputChannels Number of output filter channels.
 * @param stride Convolution stride.
 * @param index Optimization index indicator.
 * @return Encrypted convolution output.
 */
Ctext FHEONANNController::he_convolution_optimized(Ctext& encryptedInput,  vector<vector<Ptext>>& kernelData, Ptext& biasInput, 
            int inputWidth, int inputChannels, int outputChannels, int stride, int index){
    auto start_time = startTime();
    
    int kernelSq = 9;
    int inputSize = inputWidth*inputWidth;
    int widthOut = inputWidth/stride;
    int outputSize = widthOut*widthOut;
    int encode_level = encryptedInput->GetLevel();
    vector<Ctext> rotated_ciphertexts;
    
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
    Ptext cleaning_mask =  context->MakeCKKSPackedPlaintext(generate_mixed_mask(inputSize, (inputChannels*inputSize)), 1, encode_level);
            
    vector<Ctext> kernelSum(kernelSq);
    vector<Ctext> sumVec(inputChannels);
    vector<Ctext> finalVec(outputChannels);
    for(int outCh=0; outCh<outputChannels; outCh++){
        for(int j=0; j<kernelSq; j++){
            kernelSum[j] = context->EvalMult(rotated_ciphertexts[j], kernelData[outCh][j]);
        }
        sumVec[0] = context->EvalAddMany(kernelSum);
        
        /*** STEP 4: SUM RESULTS OF ALL INPUT CHANNELS ***/
        for(int k=1; k<inputChannels; k++){
            sumVec[k] = context->EvalRotate(sumVec[k-1], inputSize);
        }
        Ctext interCipher = context->EvalMult(context->EvalAddMany(sumVec), cleaning_mask);
        /**** STEP 5: THIS IS EXTRACTING DATA FROM CONVOLUTION WITH STRIDING > 1 */
        if(stride != 1){
            interCipher = downsample(interCipher, inputWidth, stride);
        }
        /** STEP 7: ROTATE THE CIPHERTEXT FOR EACH CHANNEL TO RECONSTRUCT THE OUTPUT */
        if(outCh == 0){
            finalVec[outCh] = interCipher;
        }
        else{
            finalVec[outCh] = context->EvalRotate(interCipher, -outCh*outputSize);
        }
    }

    Ctext finalResults  = context->EvalAdd(context->EvalAddMany(finalVec), biasInput);
    finalVec.clear();
    sumVec.clear();
    kernelSum.clear();
    rotated_ciphertexts.clear();
    printTiming("Optimized convolution evaluated...", "Conv", start_time);
    return finalResults;
}


/**
 * @brief Optimized convolution supporting multiple packed channels.
 *
 * @param encryptedInput Encrypted input ciphertext.
 * @param kernelData Pre-encoded filter weight vectors.
 * @param biasInput Pre-encoded bias input vector.
 * @param inputWidth Dimensions of the input channels.
 * @param inputChannels Number of input channels.
 * @param outputChannels Number of output filter channels.
 * @return Encrypted convolution output.
 */
Ctext  FHEONANNController::he_convolution_optimized_with_multiple_channels(Ctext& encryptedInput, vector<vector<Ptext>>& kernelData, 
    Ptext& biasInput,  int inputWidth,  int inputChannels, int outputChannels){
    auto start_time = startTime();
    
    constexpr int stride = 2;
    constexpr int kernelSq = 9;  // Assuming 3x3 kernel (kernelWidthSq = 9)
    int outputWidth = inputWidth / stride;
    int inputSize = inputWidth * inputWidth;
    int outputSize = outputWidth * outputWidth;
    int encodeLevel = encryptedInput->GetLevel();

    // Precompute rotations only once with minimal rotation set
    vector<Ctext> rotatedInputs;
    int vectorSize = inputSize * inputChannels;
    Ptext cleaningMask = context->MakeCKKSPackedPlaintext(generate_mixed_mask(inputSize, vectorSize), 1, encodeLevel);

    Ptext cleaningoutputMask = context->MakeCKKSPackedPlaintext(generate_mixed_mask((inputChannels*outputSize), vectorSize), 1, encodeLevel);
    
    // Horizontal rotations
    auto digits = context->EvalFastRotationPrecompute(encryptedInput);
    Ctext first_shot = context->EvalFastRotation(encryptedInput, -1, context->GetCyclotomicOrder(), digits);
    Ctext second_shot = context->EvalFastRotation(encryptedInput, 1, context->GetCyclotomicOrder(), digits);
    rotatedInputs.push_back(context->EvalRotate(first_shot, -inputWidth));
    rotatedInputs.push_back(context->EvalFastRotation(encryptedInput, -inputWidth, context->GetCyclotomicOrder(), digits));
    rotatedInputs.push_back(context->EvalRotate(second_shot, -inputWidth));
    rotatedInputs.push_back(first_shot);
    rotatedInputs.push_back(encryptedInput);
    rotatedInputs.push_back(second_shot);
    rotatedInputs.push_back(context->EvalRotate(first_shot, inputWidth));
    rotatedInputs.push_back(context->EvalFastRotation(encryptedInput, inputWidth, context->GetCyclotomicOrder(), digits));
    rotatedInputs.push_back(context->EvalRotate(second_shot, inputWidth));

    // Create vectors to store results
    int innerCount = 0;
    int outCount = 0;
    int outchanSize = outputChannels/inputChannels;
    Ctext mainResult, shortcutResult;
    vector<Ctext> mainResults(outchanSize);
    vector<Ctext> inChannelsResults(inputChannels);
    vector<Ctext> convChannelSum(inputChannels);
    vector<Ctext> kernelSum(kernelSq);
    // Process output channels with batch approach
    for (int outCh = 0; outCh < outputChannels; outCh++) {

        // Apply convolution with batch channel processing
        for (int j = 0; j<kernelSq; ++j) {
            kernelSum[j] = context->EvalMult(rotatedInputs[j], kernelData[outCh][j]);
        }
        convChannelSum[0] = context->EvalAddMany(kernelSum);
        for (int inCh = 1; inCh<inputChannels; inCh++) {
            convChannelSum[inCh] = context->EvalRotate(convChannelSum[inCh-1] , inputSize);
        }

        if(innerCount == 0){
            inChannelsResults[innerCount] = context->EvalMult(context->EvalAddMany(convChannelSum), cleaningMask);
        }
        else{
            inChannelsResults[innerCount] = context->EvalRotate(context->EvalMult(context->EvalAddMany(convChannelSum), cleaningMask), (-innerCount*inputSize));
        }

        if(innerCount == inputChannels-1){
            mainResult = context->EvalAddMany(inChannelsResults);
            mainResult = downsample_with_multiple_channels(mainResult, inputWidth, stride, inputChannels);
            mainResult = context->EvalMult(mainResult, cleaningoutputMask);
            shortcutResult = context->EvalMult(shortcutResult, cleaningoutputMask);
            
            if(outCount == 0){
                mainResults[outCount] = mainResult;
            }
            else{
                int rotateAmount = - outCount * (inputChannels * outputSize);
                mainResults[outCount] = context->EvalRotate(mainResult, rotateAmount);
            }
            outCount++;
            innerCount = 0;
        }
        else{
            innerCount++;
        }
    }

    // Combine results and add biases
    Ctext finalMainResult = context->EvalAdd(context->EvalAddMany(mainResults), biasInput);
    rotatedInputs.clear();
    mainResults.clear();
    convChannelSum.clear();
    printTiming("Optimized convolution with multiple channels evaluated...", "Conv", start_time);
    return finalMainResult;
}

/**
 * @brief Shortcut connection convolution layer for ResNet skip connections.
 *
 * @param encryptedInput Encrypted input ciphertext.
 * @param kernelData Pre-encoded filter weight vectors.
 * @param biasInput Pre-encoded bias input vector.
 * @param inputWidth Dimensions of the input channels.
 * @param inputChannels Number of input channels.
 * @param outputChannels Number of output filter channels.
 * @return Encrypted convolution output.
 */
Ctext FHEONANNController::he_shortcut_convolution(Ctext& encryptedInput,  vector<Ptext>& kernelData, Ptext& biasInput,  
            int inputWidth, int inputChannels, int outputChannels){
    auto start_time = startTime();

    int width_sq = pow(inputWidth, 2);
    int stride = 2;
    int width_out = (inputWidth/stride);
    int width_out_sq =  pow(width_out, 2);
    int encode_level = encryptedInput->GetLevel();
    int noSlots = inputChannels*width_sq;

    int zeros_elements = width_sq*(inputChannels-1);
    auto mixed_mask = generate_mixed_mask(width_sq, zeros_elements);
    Ptext cleaning_mask =  context->MakeCKKSPackedPlaintext(mixed_mask, 1, encode_level, nullptr, noSlots);

    /*** we do this in a loop contains the output layers since this is repeated for every output layer */
    vector<Ctext> final_vec(outputChannels);
    vector<Ctext> sum_vec(inputChannels);
    Ctext interCipher;
    for(int i=0; i<outputChannels; i++){
        sum_vec[0] = context->EvalMult(encryptedInput, kernelData[i]);

        /*** STEP 4: SUM RESULTS OF ALL INPUT CHANNELS **/
        for(int k=1; k<inputChannels; k++){
            sum_vec[k] = context->EvalRotate(sum_vec[k-1], width_sq);
        }
        interCipher = context->EvalMult(context->EvalAddMany(sum_vec), cleaning_mask);

        /**** STEP 5: THIS IS EXTRACTING DATA FROM CONVOLUTION WITH STRIDING > 1 */
        interCipher = downsample(interCipher, inputWidth, stride);

        /** STEP 7: ROTATE THE CIPHERTEXT FOR EACH CHANNEL TO RECONSTRUCT THE OUTPUT */
        if(i == 0){
            final_vec[0] = interCipher;
        }
        else{
            final_vec[i] = context->EvalRotate(interCipher, -(i*width_out_sq));
        }
    }

    Ctext finalResults = context->EvalAdd(context->EvalAddMany(final_vec), biasInput);
    final_vec.clear();
    sum_vec.clear();
    printTiming("Shortcut convolution evaluated...", "Conv", start_time);
    return finalResults;
}

/**
 * @brief Combined optimized convolution and shortcut branch computation.
 *
 * @param encryptedInput Input ciphertext.
 * @param kernelData Convolution weights.
 * @param shortcutKernelData Shortcut weights.
 * @param biasVector Convolution bias.
 * @param shortcutBiasVector Shortcut bias.
 * @param inputWidth Input dimension.
 * @param inputChannels Input channels count.
 * @param outputChannels Output channels count.
 * @return Vector containing [convolution_output, shortcut_output].
 */
vector<Ctext>  FHEONANNController::he_convolution_and_shortcut_optimized( const Ctext& encryptedInput, const vector<vector<Ptext>>& kernelData, 
    const vector<Ptext>& shortcutKernelData, Ptext& biasInput, Ptext& shortcutBiasVector,  
    int inputWidth,  int inputChannels, int outputChannels){
    auto start_time = startTime();
    constexpr int stride = 2;
    constexpr int kernelSq = 9;  // Assuming 3x3 kernel (kernelWidthSq = 9)
    int outputWidth = inputWidth / stride;
    int inputSize = inputWidth * inputWidth;
    int outputSize = outputWidth * outputWidth;
    int encodeLevel = encryptedInput->GetLevel();

    // Precompute rotations only once with minimal rotation set
    vector<Ctext> rotatedInputs;
    int vectorSize = inputSize * inputChannels;
    Ptext cleaningMask = context->MakeCKKSPackedPlaintext(generate_mixed_mask(inputSize, vectorSize), 1, encodeLevel);
    
    // Horizontal rotations
    auto digits = context->EvalFastRotationPrecompute(encryptedInput);
    Ctext first_shot = context->EvalFastRotation(encryptedInput, -1, context->GetCyclotomicOrder(), digits);
    Ctext second_shot = context->EvalFastRotation(encryptedInput, 1, context->GetCyclotomicOrder(), digits);
    rotatedInputs.push_back(context->EvalRotate(first_shot, -inputWidth));
    rotatedInputs.push_back(context->EvalFastRotation(encryptedInput, -inputWidth, context->GetCyclotomicOrder(), digits));
    rotatedInputs.push_back(context->EvalRotate(second_shot, -inputWidth));
    rotatedInputs.push_back(first_shot);
    rotatedInputs.push_back(encryptedInput);
    rotatedInputs.push_back(second_shot);
    rotatedInputs.push_back(context->EvalRotate(first_shot, inputWidth));
    rotatedInputs.push_back(context->EvalFastRotation(encryptedInput, inputWidth, context->GetCyclotomicOrder(), digits));
    rotatedInputs.push_back(context->EvalRotate(second_shot, inputWidth));

    // Create vectors to store results
    vector<Ctext> convChannelSum(inputChannels), shortcutChannelSum(inputChannels);
    vector<Ctext> mainResults(outputChannels), shortcutResults(outputChannels);
    vector<Ctext> kernelSum(kernelSq);
    Ctext mainResult, shortcutResult;
    
    // Process output channels with batch approach
    for (int outCh = 0; outCh < outputChannels; outCh++) {

        // Apply convolution with batch channel processing
        for (int j = 0; j < kernelSq; ++j) {
            kernelSum[j] = context->EvalMult(rotatedInputs[j], kernelData[outCh][j]);
        }
        convChannelSum[0] = context->EvalAddMany(kernelSum);
        shortcutChannelSum[0] = context->EvalMult(encryptedInput, shortcutKernelData[outCh]);

        for (int inCh = 1; inCh < inputChannels; ++inCh) {
            convChannelSum[inCh] = context->EvalRotate(convChannelSum[inCh-1] , inputSize);
            shortcutChannelSum[inCh] = context->EvalRotate(shortcutChannelSum[inCh-1], inputSize);
        }

        mainResult = context->EvalMult(context->EvalAddMany(convChannelSum), cleaningMask);
        shortcutResult = context->EvalMult(context->EvalAddMany(shortcutChannelSum), cleaningMask);

        /** Compute Striding */
        mainResult = downsample(mainResult, inputWidth, stride);
        shortcutResult = downsample(shortcutResult, inputWidth, stride);
        
        if( outCh==0 ){
            mainResults[outCh] = mainResult;
            shortcutResults[outCh] = shortcutResult;
        }
        else{
            int rotateAmount = - outCh * outputSize;
            // cout <<"Rotattion Positon: " << rotateAmount << endl;
            mainResults[outCh] = context->EvalRotate(mainResult, rotateAmount);                
            shortcutResults[outCh] = context->EvalRotate(shortcutResult, rotateAmount);
        }
    }
    
    // Combine results and add biases
    Ctext finalMainResult = context->EvalAdd(context->EvalAddMany(mainResults), biasInput);
    Ctext finalShortcutResult = context->EvalAdd(context->EvalAddMany(shortcutResults), shortcutBiasVector);

    rotatedInputs.clear();
    mainResults.clear();
    shortcutResults.clear();
    convChannelSum.clear();
    shortcutChannelSum.clear();
    printTiming("Optimized convolution and shortcut evaluated...", "Conv", start_time);
    return {finalMainResult, finalShortcutResult};
}

/**
 * @brief Combined optimized convolution and shortcut branch for multiple channels.
 *
 * @param encryptedInput Input ciphertext.
 * @param kernelData Convolution weights.
 * @param shortcutKernelData Shortcut weights.
 * @param biasInput Convolution bias.
 * @param shortcutBiasInput Shortcut bias.
 * @param inputWidth Input dimension.
 * @param inputChannels Input channels count.
 * @param outputChannels Output channels count.
 * @return Vector containing [convolution_output, shortcut_output].
 */
vector<Ctext>  FHEONANNController::he_convolution_and_shortcut_optimized_with_multiple_channels( const Ctext& encryptedInput, const vector<vector<Ptext>>& kernelData, 
    const vector<Ptext>& shortcutKernelData, Ptext& biasInput, Ptext& shortcutBiasInput,  
    int inputWidth,  int inputChannels, int outputChannels){
    auto start_time = startTime();
    constexpr int stride = 2;
    constexpr int kernelSq = 9;  // Assuming 3x3 kernel (kernelWidthSq = 9)
    int outputWidth = inputWidth / stride;
    int inputSize = inputWidth * inputWidth;
    int outputSize = outputWidth * outputWidth;
    int encodeLevel = encryptedInput->GetLevel();

    // Precompute rotations only once with minimal rotation set
    vector<Ctext> rotatedInputs;
    int vectorSize = inputSize * inputChannels;
    Ptext cleaningMask = context->MakeCKKSPackedPlaintext(generate_mixed_mask(inputSize, vectorSize), 1, encodeLevel);

    Ptext cleaningoutputMask = context->MakeCKKSPackedPlaintext(generate_mixed_mask((inputChannels*outputSize), vectorSize), 1, encodeLevel);
    
    // Horizontal rotations
    auto digits = context->EvalFastRotationPrecompute(encryptedInput);
    Ctext first_shot = context->EvalFastRotation(encryptedInput, -1, context->GetCyclotomicOrder(), digits);
    Ctext second_shot = context->EvalFastRotation(encryptedInput, 1, context->GetCyclotomicOrder(), digits);
    rotatedInputs.push_back(context->EvalRotate(first_shot, -inputWidth));
    rotatedInputs.push_back(context->EvalFastRotation(encryptedInput, -inputWidth, context->GetCyclotomicOrder(), digits));
    rotatedInputs.push_back(context->EvalRotate(second_shot, -inputWidth));
    rotatedInputs.push_back(first_shot);
    rotatedInputs.push_back(encryptedInput);
    rotatedInputs.push_back(second_shot);
    rotatedInputs.push_back(context->EvalRotate(first_shot, inputWidth));
    rotatedInputs.push_back(context->EvalFastRotation(encryptedInput, inputWidth, context->GetCyclotomicOrder(), digits));
    rotatedInputs.push_back(context->EvalRotate(second_shot, inputWidth));

    // Create vectors to store results
    int innerCount = 0;
    int outCount = 0;
    int outchanSize = outputChannels/inputChannels;
    Ctext mainResult, shortcutResult;
    vector<Ctext> mainResults(outchanSize), shortcutResults(outchanSize);
    vector<Ctext> inChannelsResults(inputChannels), inshortcutResults(inputChannels);
    vector<Ctext> convChannelSum(inputChannels), shortcutChannelSum(inputChannels);
    vector<Ctext> kernelSum(kernelSq);

    // Process output channels with batch approach
    for (int outCh = 0; outCh < outputChannels; outCh++) {

        // Apply convolution with batch channel processing
        for (int j = 0; j<kernelSq; ++j) {
            kernelSum[j] = context->EvalMult(rotatedInputs[j], kernelData[outCh][j]);
        }
        convChannelSum[0] = context->EvalAddMany(kernelSum);
        shortcutChannelSum[0] = context->EvalMult(encryptedInput, shortcutKernelData[outCh]);

        for (int inCh = 1; inCh<inputChannels; inCh++) {
            convChannelSum[inCh] = context->EvalRotate(convChannelSum[inCh-1] , inputSize);
            shortcutChannelSum[inCh] = context->EvalRotate(shortcutChannelSum[inCh-1], inputSize);
        }

        if(innerCount == 0){
            inChannelsResults[innerCount] = context->EvalMult(context->EvalAddMany(convChannelSum), cleaningMask);
            inshortcutResults[innerCount] = context->EvalMult(context->EvalAddMany(shortcutChannelSum), cleaningMask);
        }
        else{
            inChannelsResults[innerCount] = context->EvalRotate(context->EvalMult(context->EvalAddMany(convChannelSum), cleaningMask), (-innerCount*inputSize));
            inshortcutResults[innerCount] = context->EvalRotate(context->EvalMult(context->EvalAddMany(shortcutChannelSum), cleaningMask), (-innerCount*inputSize));
        }

        if(innerCount == inputChannels-1){
            mainResult = context->EvalAddMany(inChannelsResults);
            shortcutResult = context->EvalAddMany(inshortcutResults);
            mainResult = downsample_with_multiple_channels(mainResult, inputWidth, stride, inputChannels);
            shortcutResult = downsample_with_multiple_channels(shortcutResult, inputWidth,stride,  inputChannels);
            mainResult = context->EvalMult(mainResult, cleaningoutputMask);
            shortcutResult = context->EvalMult(shortcutResult, cleaningoutputMask);
            
            if(outCount == 0){
                mainResults[outCount] = mainResult;
                shortcutResults[outCount] = shortcutResult;
            }
            else{
                
                int rotateAmount = - outCount * (inputChannels * outputSize);
                mainResults[outCount] = context->EvalRotate(mainResult, rotateAmount);
                shortcutResults[outCount] = context->EvalRotate(shortcutResult, rotateAmount);   
            }
            outCount++;
            innerCount = 0;
        }
        else{
            innerCount++;
        }
    }
    
    // Combine results and add biases
    Ctext finalMainResult = context->EvalAdd(context->EvalAddMany(mainResults), biasInput);
    Ctext finalShortcutResult = context->EvalAdd(context->EvalAddMany(shortcutResults), shortcutBiasInput);

    rotatedInputs.clear();
    mainResults.clear();
    shortcutResults.clear();
    convChannelSum.clear();
    shortcutChannelSum.clear();
    printTiming("Optimized convolution and shortcut with multiple channels evaluated...", "Conv", start_time);
    return {finalMainResult, finalShortcutResult};
}

/**
 * @brief Homomorphic standard average pooling layer.
 *
 * @param encryptedInput Encrypted input ciphertext.
 * @param inputWidth Dimension of input.
 * @param inputChannels Number of channels.
 * @param kernelWidth Dimension of pooling window.
 * @param stride Pooling stride.
 * @return Average-pooled encrypted output.
 */
Ctext FHEONANNController::he_avgpool(Ctext encryptedInput,  int inputWidth, int inputChannels, int kernelWidth, int stride){
    auto start_time = startTime();

    int outputWidth = inputWidth/stride;
    int kernelSq = pow(kernelWidth, 2);
    int inputSize = pow(inputWidth, 2);
    int outputSize = pow(outputWidth, 2);
    int encode_level = encryptedInput->GetLevel();

    /*** STEP 1 - ROTATE THE CIPHERTEXT into by k^2-1 and create a k^2 rotated right positions ***/
    vector<Ctext> rotated_ciphertexts;
    for (int i = 0; i < kernelWidth; i++) {
        if(i >0){
            encryptedInput = context->EvalRotate(encryptedInput, inputWidth);
        }
        rotated_ciphertexts.push_back(encryptedInput);
        for (int j = 1; j < kernelWidth; j++) {
            rotated_ciphertexts.push_back(context->EvalRotate(encryptedInput, j));
        }
    }
    
    /**** STEP 2: Sum the rotated ciphertext */
    Ctext sum_cipher = context->EvalAddMany(rotated_ciphertexts);
    vector<Ctext> channel_ciphers;
    if(inputWidth <= 2){
        for(int i = 1; i<inputChannels; i++){
            sum_cipher = context->EvalRotate(sum_cipher, inputSize);
            channel_ciphers.push_back(sum_cipher);
        }
        return context->EvalMerge(channel_ciphers);
    }
    
    /*** STEP 3: Multiply the scale value with the sum cipher */
    int num_of_elements = inputChannels*inputSize;
    auto masked_data = generate_scale_mask(kernelSq, num_of_elements);
    auto masked_cipher =  context->MakeCKKSPackedPlaintext(masked_data, 1, encode_level);
    sum_cipher = context->EvalMult(sum_cipher, masked_cipher);

    /*** STEP 4: Extract the values needed in the ciphertext */
    Ctext strided_cipher = downsample(sum_cipher, inputWidth,  stride);
    channel_ciphers.push_back(strided_cipher);
    for(int i = 1; i<inputChannels; i++){
        sum_cipher = context->EvalRotate(sum_cipher, inputSize);
        channel_ciphers.push_back(context->EvalRotate(downsample(sum_cipher, inputWidth, stride), -(i*outputSize)));
    }
    Ctext finalResults = context->EvalAddMany(channel_ciphers);
    channel_ciphers.clear();
    rotated_ciphertexts.clear();
    printTiming("Average pooling evaluated...", "AvgPool", start_time);
    return finalResults;
}

/**
 * @brief Advanced homomorphic average pooling supporting padding.
 *
 * @param encryptedInput Encrypted input.
 * @param inputWidth Input dimension.
 * @param outputChannels Output channels count.
 * @param kernelWidth Pooling kernel size.
 * @param stride Pooling stride.
 * @param padding Pooling padding.
 * @return Average-pooled encrypted output.
 */
Ctext FHEONANNController::he_avgpool_advanced(Ctext encryptedInput, int inputWidth, int outputChannels, 
    int kernelWidth, int stride, int padding){
    auto start_time = startTime();
 
    int encode_level = encryptedInput->GetLevel();
    if(padding ==0){
        auto avgpool_cipher = he_avgpool(encryptedInput, inputWidth, outputChannels, kernelWidth, stride);
        printTiming("Advanced average pooling evaluated...", "AvgPool", start_time);
        return avgpool_cipher;
    }
    int padded_width = inputWidth + (2*padding);
    int padded_width_sq = pow(padded_width, 2);
    int width_sq = pow(inputWidth, 2);
    int zeros_elements = ((outputChannels*width_sq) - inputWidth);
    auto padding_mix_mask = generate_mixed_mask(inputWidth, zeros_elements);
    Ptext in_clean_mask =  context->MakeCKKSPackedPlaintext(padding_mix_mask, 1, encode_level);

    /** generate vector of padding width */
    Ctext channel_cipher = encryptedInput;
    vector<Ctext> channel_vector_ciphers;
    for(int i = 0; i< outputChannels; i++){
        if(i!=0){
            channel_cipher = context->EvalRotate(channel_cipher, width_sq);
        }
        vector<Ctext> in_chan_vec;
        Ctext in_chan_cipher = channel_cipher;
        for(int k=0; k<inputWidth; k++){
            Ctext in_clean_cipher = context->EvalMult(in_chan_cipher, in_clean_mask);
            in_chan_cipher =  context->EvalRotate(in_chan_cipher, inputWidth);
            if(k==0){
                in_chan_vec.push_back(in_clean_cipher);
            }
            else{
                int in_rot_position = k*padded_width;
                Ctext padded_cipher = context->EvalRotate(in_clean_cipher, -in_rot_position);
                in_chan_vec.push_back(padded_cipher);
            }
        }
        Ctext in_sum_cipher = context->EvalAddMany(in_chan_vec);
        if(i==0){
            channel_vector_ciphers.push_back(in_sum_cipher);
        }
        else{
            int in_rotate = i*padded_width_sq;
            Ctext in_rotate_cipher = context->EvalRotate(in_sum_cipher, -in_rotate);
            channel_vector_ciphers.push_back(in_rotate_cipher);
        }
    }
    int padd_extra = (padding*padded_width)+padding;
    Ctext padded_cipher = context->EvalAddMany(channel_vector_ciphers);
    if(padd_extra != 0){
        padded_cipher = context->EvalRotate(padded_cipher, -padd_extra);
    }
    Ctext avgpool_cipher = he_avgpool(padded_cipher, inputWidth, outputChannels, kernelWidth, stride); 
    printTiming("Advanced average pooling evaluated...", "AvgPool", start_time);
    return avgpool_cipher;
}

/**
 * @brief Optimized average pooling using slot shift aggregation.
 *
 * @param encryptedInput Encrypted input.
 * @param inputWidth Input dimension.
 * @param inputChannels Input channels.
 * @param kernelWidth pooling kernel size.
 * @param stride pooling stride.
 * @return Average-pooled encrypted output.
 */
Ctext FHEONANNController::he_avgpool_optimzed(Ctext& encryptedInput,  int inputWidth, int inputChannels, int kernelWidth, int stride){
    auto start_time = startTime();

    int kernelSq = pow(kernelWidth, 2);
    int inputSize = pow(inputWidth, 2);
    int outputWidth = inputWidth/stride;
    int outputSize = pow(outputWidth, 2);
    int encode_level = encryptedInput->GetLevel();
    
    /*** STEP 1 - ROTATE THE CIPHERTEXT into by k^2-1 and create a k^2 rotated right positions ***/
    vector<Ctext> rotated_ciphertexts;
    auto digits = context->EvalFastRotationPrecompute(encryptedInput);
    rotated_ciphertexts.push_back(encryptedInput);
    rotated_ciphertexts.push_back(context->EvalFastRotation(encryptedInput, 1, context->GetCyclotomicOrder(), digits));
    rotated_ciphertexts.push_back(context->EvalFastRotation(encryptedInput, inputWidth, context->GetCyclotomicOrder(), digits));
    rotated_ciphertexts.push_back(context->EvalRotate(context->EvalFastRotation(encryptedInput, inputWidth, context->GetCyclotomicOrder(), digits), 1));
    Ctext sum_cipher = context->EvalAddMany(rotated_ciphertexts);

    /*** STEP 3: Multiply the scale value with the sum cipher */
    int num_of_elements = inputChannels*inputSize;
    auto masked_data = generate_scale_mask(kernelSq, num_of_elements);
    auto masked_cipher =  context->MakeCKKSPackedPlaintext(masked_data, 1, encode_level);
    sum_cipher = context->EvalMult(sum_cipher, masked_cipher);
    
    vector<Ctext> channel_ciphers;
    
    /**** Caryout the average pooling ofif we have just 3 elements in a channel */
    if(inputWidth <= 2){
        for(int i = 1; i<inputChannels; i++){
            sum_cipher = context->EvalRotate(sum_cipher, inputSize);
            channel_ciphers.push_back(sum_cipher);
        }
        return context->EvalMerge(channel_ciphers);
    }

    /*** STEP 4: Extract the values needed in the ciphertext */
    Ctext strided_cipher = downsample(sum_cipher, inputWidth,  stride);
    channel_ciphers.push_back(strided_cipher);
    for(int i = 1; i<inputChannels; i++){
        sum_cipher = context->EvalRotate(sum_cipher, inputSize);
        channel_ciphers.push_back(context->EvalRotate(downsample(sum_cipher, inputWidth, stride), -i*outputSize));
    }
    Ctext finalResult = context->EvalAddMany(channel_ciphers);
    channel_ciphers.clear();
    printTiming("Optimized average pooling evaluated...", "AvgPool", start_time);
    return finalResult;
}


/**
 * @brief Optimized average pooling supporting multiple channels.
 *
 * @param encryptedInput Encrypted input.
 * @param inputWidth Input dimension.
 * @param inputChannels Input channels.
 * @param kernelWidth pooling kernel size.
 * @param stride pooling stride.
 * @return Average-pooled encrypted output.
 */
Ctext FHEONANNController::he_avgpool_optimzed_with_multiple_channels(Ctext& encryptedInput,  int inputWidth, int inputChannels, int kernelWidth, int stride){
    auto start_time = startTime();

    int kernelSq = pow(kernelWidth, 2);
    int inputSize = pow(inputWidth, 2);
    int encode_level = encryptedInput->GetLevel();
    
    /*** STEP 1 - ROTATE THE CIPHERTEXT into by k^2-1 and create a k^2 rotated right positions ***/
    vector<Ctext> rotated_ciphertexts;
    auto digits = context->EvalFastRotationPrecompute(encryptedInput);
    rotated_ciphertexts.push_back(encryptedInput);
    rotated_ciphertexts.push_back(context->EvalFastRotation(encryptedInput, 1, context->GetCyclotomicOrder(), digits));
    rotated_ciphertexts.push_back(context->EvalFastRotation(encryptedInput, inputWidth, context->GetCyclotomicOrder(), digits));
    rotated_ciphertexts.push_back(context->EvalRotate(context->EvalFastRotation(encryptedInput, inputWidth, context->GetCyclotomicOrder(), digits), 1));
    Ctext sum_cipher = context->EvalAddMany(rotated_ciphertexts);

    /*** STEP 3: Multiply the scale value with the sum cipher */
    int num_of_elements = inputChannels*inputSize;
    auto masked_data = generate_scale_mask(kernelSq, num_of_elements);
    auto masked_cipher =  context->MakeCKKSPackedPlaintext(masked_data, 1, encode_level);
    sum_cipher = context->EvalMult(sum_cipher, masked_cipher);
    
    vector<Ctext> channel_ciphers;
    
    /**** Caryout the average pooling ofif we have just 3 elements in a channel */
    if(inputWidth <= 2){
        for(int i = 1; i<inputChannels; i++){
            sum_cipher = context->EvalRotate(sum_cipher, inputSize);
            channel_ciphers.push_back(sum_cipher);
        }
        Ctext finalResult = context->EvalMerge(channel_ciphers);
        printTiming("Optimized average pooling with multiple channels evaluated...", "AvgPool", start_time);
        return finalResult;
    }

    Ctext finalResult = downsample_with_multiple_channels(sum_cipher, inputWidth, stride, inputChannels);
    printTiming("Optimized average pooling with multiple channels evaluated...", "AvgPool", start_time);
    return finalResult; 
}

/**
 * @brief Homomorphically add two ciphertexts together.
 *
 * @param firstInput The first encrypted input.
 * @param secondInput The second encrypted input.
 * @return Encrypted sum of the inputs.
 */
Ctext FHEONANNController::he_sum_two_ciphertexts(Ctext& firstInput, Ctext& secondInput){
    auto start_time = startTime();
    Ctext sumCipher = context->EvalAdd(firstInput, secondInput);
    printTiming("Addition evaluated...", "Add", start_time);
    return sumCipher;
}

/**
 * @brief Homomorphic global average pooling (reduces space dimension to 1).
 *
 * @param encryptedInput Input ciphertext.
 * @param inputWidth Input dimension.
 * @param outputChannels Output channels.
 * @param kernelWidth Pooling window dimension.
 * @param rotatePositions Rotation positions.
 * @return Encrypted global pooled output.
 */
Ctext FHEONANNController::he_globalavgpool(Ctext& encryptedInput,  int inputWidth, 
                        int outputChannels, int kernelWidth, int rotatePositions){
    auto start_time = startTime();
    
    // int encode_level = encryptedInput->GetLevel();
    /**** STEP 2: Sum the rotated ciphertext */
    // Ctext sum_cipher = context->EvalAddMany(rotated_ciphertexts);
    int width_sq = inputWidth*inputWidth;
    // int zero_elements = outputChannels*pow(kernelWidth, 2);
    auto masked_cipher =  context->MakeCKKSPackedPlaintext(generate_scale_mask(width_sq, outputChannels), 1);
    // auto mixed_masked_cipher =  context->MakeCKKSPackedPlaintext(generate_mixed_mask(width_sq, zero_elements), 1);

    /*** STEP 4: Extract the values needed in the ciphertext
    ****        Generate  ciphers = number of output channels rotating at width^2 
    **** rotate at each cipher at w and extract values in w *********/
    vector<Ctext> channel_ciphers;
    vector<Ctext> result_ciphers;
    Ctext in_rot_ciphertext;
    int rotation_index = 0;
    int j = 0;
    
    for(int i = 0; i< outputChannels; i++){
        if(i != 0){
            encryptedInput = context->EvalRotate(encryptedInput, width_sq);
        }
        // Ctext sumRe = context->EvalMult(encryptedInput, mixed_masked_cipher);
        result_ciphers.push_back(context->EvalSum(encryptedInput, width_sq));
        j+=1;

        /** check whether is equal to imgcols, merge them and rotate by imgCols. 
        * If i is equal to the outputSize, merge and rotate by imgCols */
        if(j == rotatePositions || i == (outputChannels-1)){
            if(rotation_index > 0){
                channel_ciphers.push_back(context->EvalRotate(context->EvalMerge(result_ciphers), -rotation_index));
            }
            else{
                Ctext merged = context->EvalMerge(result_ciphers);
                channel_ciphers.push_back(merged);
                // cout << "merged" << endl; 
            }
            rotation_index += rotatePositions;
            result_ciphers.clear();
            j=0;
        }
    }

    Ctext res = context->EvalMult(context->EvalAddMany(channel_ciphers), masked_cipher);
    channel_ciphers.clear();
    result_ciphers.clear();
    printTiming("Global average pooling evaluated...", "AvgPool", start_time);
    return res;
}

/**
 * @brief Homomorphic fully connected linear layer.
 *
 * @param encryptedInput Input ciphertext vector.
 * @param weightMatrix Weight plaintexts vector.
 * @param biasInput Bias plaintext.
 * @param inputSize Size of input dimension.
 * @param outputSize Size of output dimension.
 * @param rotatePositions Rotation positions.
 * @return Encrypted output of linear layer.
 */
Ctext FHEONANNController::he_linear(Ctext& encryptedInput, vector<Ptext>& weightMatrix, Ptext& biasInput, 
                    int inputSize, int outputSize, int rotatePositions){
    auto start_time = startTime();

    int output_size = weightMatrix.size();
    if(outputSize > output_size){
        /** need to handle error here because outputSize should never be grater than output_size */
        cout << "There is an error: ouputsize cannot be larger than weightedMatrix" << endl;
        return encryptedInput;
    }
    /* calculate the results of weights * encrypted vector + bais. 
    Shit the results by number of elements in inputsize*iteration value */
    vector<Ctext> result_matrix;
    vector<Ctext> inner_matrix;
    int j = 0;
    int rotation_index = 0;
    for(int i = 0; i < outputSize; i++){
        inner_matrix.push_back(context->EvalSum(context->EvalMult(encryptedInput, weightMatrix[i]), inputSize));
        j+=1;
        /** check whether is equal to imgcols, merge them and rotate by imgCols. 
         * If i is equal to the outputSize, merge and rotate by imgCols */
        if(j == rotatePositions || i == (outputSize-1)){
            if(rotation_index > 0){
                result_matrix.push_back(context->EvalRotate(context->EvalMerge(inner_matrix), -rotation_index));
            }
            else{
                result_matrix.push_back(context->EvalMerge(inner_matrix));
            }
            inner_matrix.clear();
            rotation_index +=rotatePositions;
            j=0;
            
        }
    }

    /**** convert everything to one vector. and add the biasInput  ***/
    Ctext fResults = context->EvalAdd(context->EvalAddMany(result_matrix), biasInput);
    inner_matrix.clear();
    result_matrix.clear();
    printTiming("Linear evaluated...", "FC", start_time);
    return fResults;
}

/**
 * @brief Optimized homomorphic fully connected linear layer.
 *
 * @param encryptedInput Input ciphertext vector.
 * @param weightMatrix Weight plaintexts vector.
 * @param biasInput Bias plaintext.
 * @param inputSize Size of input.
 * @param outputSize Size of output.
 * @return Encrypted output of linear layer.
 */
Ctext FHEONANNController::he_linear_optimized(Ctext& encryptedInput, vector<Ptext>& weightMatrix, 
                Ptext& biasInput, int inputSize, int outputSize){
    auto start_time = startTime();

    int output_size = weightMatrix.size();
    if(outputSize > output_size){
        /** need to handle error here because outputSize should never be grater than output_size */
        cout << "There is an error: ouputsize cannot be larger than weightedMatrix" << endl;
        return encryptedInput;
    }
    /* calculate the results of weights * encrypted vector + bais. 
    Shit the results by number of elements in inputsize*iteration value */
    vector<Ctext> inner_matrix;
    for(int i = 0; i < outputSize; i++){
        inner_matrix.push_back(context->EvalSum(context->EvalMult(encryptedInput, weightMatrix[i]), inputSize));
    }
    Ctext res = context->EvalAdd(context->EvalMerge(inner_matrix), biasInput);
    printTiming("Optimized linear evaluated...", "FC", start_time);
    return res;
}

/**
 * @brief Homomorphic ReLU activation layer (CKKS polynomial approximation).
 *
 * @param encryptedInput Ciphertext to activate.
 * @param scaleValue Scaling factor.
 * @param vectorSize Number of elements in the vector.
 * @param polyDegree Degree of approximating polynomial.
 * @return Activated ciphertext.
 */
Ctext FHEONANNController::he_relu(Ctext& encryptedInput, double scaleValue,  int vectorSize, int polyDegree) {
    auto start_time = startTime();
    double lowerBound = -1;
    double upperBound = 1;

    // scaleValue = 2*scaleValue;
    
    auto encryptInn = encryptedInput->Clone();
    if(scaleValue > 1){
        // scaleValue = 2*scaleValue;
        auto mask_data = context->MakeCKKSPackedPlaintext(generate_scale_mask(scaleValue, vectorSize), 1, 0, nullptr, nextPowerOf2(vectorSize));
        encryptInn = context->EvalMult(encryptedInput, mask_data);
    }
    else{
        scaleValue = 1;
    }
    
    Ctext relu_result = context->EvalChebyshevFunction(
        [scaleValue](double x) -> double { if (x < 0) return 0; else return scaleValue*x; }, 
                                            encryptInn,
                                            lowerBound,
                                            upperBound, 
                                            polyDegree);
    printTiming("ReLU evaluated...", "ReLU", start_time);
    return relu_result;
}

/**
 * @brief Extract elements from ciphertext using basic striding.
 *
 * @param in_cipher Input ciphertext.
 * @param inputWidth Width of input.
 * @param widthOut Width of output.
 * @param stride Stride value.
 * @return Strided output ciphertext.
 */
Ctext  FHEONANNController::basic_striding(Ctext in_cipher, int inputWidth, int widthOut,  int stride){
    
    auto in_digits = context->EvalFastRotationPrecompute(in_cipher);
    vector<Ctext> chan_vec(widthOut);
    vector<Ctext> rotated_ciphertexts(widthOut);
    int i_rot = stride*inputWidth;
    
    for (int k =0; k<widthOut; k++){
        if(k!=0){
            in_cipher = context->EvalFastRotation(in_cipher, i_rot, context->GetCyclotomicOrder(), in_digits);
            in_digits = context->EvalFastRotationPrecompute(in_cipher);
        }
        for(int t=0; t<widthOut; t++){
            if(t==0){
                rotated_ciphertexts[t]= in_cipher;
            }
            else{
                rotated_ciphertexts[t] = context->EvalFastRotation(in_cipher, t*stride, context->GetCyclotomicOrder(), in_digits);
            }
        }
        Ctext merged_cipher = context->EvalMerge(rotated_ciphertexts);
        if(k == 0){
            chan_vec[k] = merged_cipher;
        }
        else{
            chan_vec[k] = context->EvalRotate(merged_cipher, -k*widthOut);
        }
    }
    return context->EvalAddMany(chan_vec);
}


// Apply convolution with batch channel processing
/**
 * @brief Perform batch convolution operations on rotated ciphertexts.
 *
 * Multiplies rotated ciphertexts with kernel weight masks and sums them up.
 *
 * @param rotatedInputs Rotated input ciphertexts.
 * @param kernelData Encoded kernel weight masks.
 * @param kernelSq Squared dimension of kernel (k^2).
 * @param inputSize Size of each input channel.
 * @param inputChannels Number of input channels.
 * @return Aggregated batch convolution output.
 */
Ctext  FHEONANNController::batch_convolution_operation(const vector<Ctext>& rotatedInputs, const vector<Ptext>& kernelData, int kernelSq, int inputSize,  int inputChannels) {
    
    // Apply kernel to each rotated cipher
    vector<Ctext> kernelSum(kernelSq);
    for (int j = 0; j < kernelSq; ++j) {
        kernelSum[j] = context->EvalMult(rotatedInputs[j], kernelData[j]);
    }
    return context->EvalAddMany(kernelSum);
}

/**
 * @brief Downsample ciphertext slots.
 *
 * @param input Input ciphertext.
 * @param inputWidth Width of input.
 * @param stride Downsample stride.
 * @return Downsampled ciphertext.
 */
Ctext FHEONANNController::downsample(const Ctext& input, int inputWidth, int stride) {
    
    Ctext result = input->Clone();
    const int outputWidth = inputWidth / stride;
    const int inputSize = inputWidth * inputWidth;
    
    // Step 1: Binary decomposition for row juxtaposition
    result = context->EvalMult(result, first_mask(inputWidth, inputSize, stride, input->GetLevel()));
    for (int s = 1; s < log2(outputWidth); s++) {
        result = context->EvalMult(
            context->EvalAdd(result, context->EvalRotate(result, pow(2, s-1))),
            generate_binary_mask(pow(2,s), inputSize, stride, input->GetLevel())
        );
    }
    result = context->EvalAdd(result, context->EvalRotate(result, pow(2, (log2(outputWidth)-1))));
    // Step 2: Row processing with optimized rotations
    Ctext downsampledrows = context->EvalMult(input, generate_zero_mask(inputSize, input->GetLevel()));
    
    for (int row = 0; row < outputWidth; ++row) {
        Ctext masked = context->EvalMult(result, generate_row_mask(row, outputWidth, inputSize, stride, input->GetLevel()));
        downsampledrows = context->EvalAdd(downsampledrows, masked);
        if(row < outputWidth-1){
            result = context->EvalRotate(result, (stride*inputWidth - outputWidth));
        }
    }
    return downsampledrows;
}

/**
 * @brief Downsample multiple channels packed within a ciphertext.
 *
 * @param input Input ciphertext.
 * @param inputWidth Input dimension.
 * @param stride Downsample stride.
 * @param numChannels Number of channels.
 * @return Downsampled ciphertext.
 */
Ctext FHEONANNController::downsample_with_multiple_channels(const Ctext& input, int inputWidth, int stride, int numChannels) {
    
    const int inputSize   = inputWidth * inputWidth; 
    const int outputWidth  = inputWidth / stride;
    const int level        = input->GetLevel();
    int outputSize = outputWidth * outputWidth;

    Ctext encryptedzeros =  context->EvalMult(input, generate_zero_mask_channels(inputSize, numChannels, input->GetLevel()));

    // 2) binary-row decomposition
    Ctext result = context->EvalMult(input, first_mask_with_channels(inputWidth, inputSize, stride, numChannels, level));

     // Step 1: Binary decomposition for row juxtaposition
    for (int s = 1; s < log2(outputWidth); s++) {
        result = context->EvalMult(
            context->EvalAdd(result, context->EvalRotate(result, pow(2, s-1))),
            generate_binary_mask_with_channels(pow(2,s), inputSize, stride,  numChannels, input->GetLevel())
        );
    }

    result = context->EvalAdd(result, context->EvalRotate(result, pow(2, (log2(outputWidth)-1))));

    // Step 2: Row processing with optimized rotations
    Ctext downsampledrows = encryptedzeros->Clone();
    for (int row = 0; row < outputWidth; row++) {
        Ctext masked = context->EvalMult(result, generate_row_mask_with_channels(row, outputWidth, inputSize, stride, numChannels, input->GetLevel()));
        downsampledrows = context->EvalAdd(downsampledrows, masked);
        if(row < outputWidth-1){
            result = context->EvalRotate(result, (stride*inputWidth - outputWidth));
        }
    }

    /***
    * step 3: process per channel
    ******/ 
    Ctext downsampledchannels =  encryptedzeros->Clone();
    for (int ch=0; ch < numChannels; ch++){
        Ctext masked = context->EvalMult(downsampledrows, generate_channel_mask_with_zeros(ch, outputSize, numChannels, input->GetLevel()));
        downsampledchannels = context->EvalAdd(downsampledchannels, masked);
        if(ch < numChannels-1){
            downsampledrows = context->EvalRotate(downsampledrows, (inputSize - outputSize));
        }
    }

    /*** step 3: process per channel */ 
    // int totalSize = numChannels * outputSize;
    // Ctext downsampledchannels = encryptedzeros->Clone();
    // for (int i = 0; i < numChannels; i++) {
    //     Ctext masked = context->EvalMult(downsampledrows, gen_channel_full_mask(i, inputSize, outputSize, numChannels, downsampledrows->GetLevel()));
    //     downsampledchannels = context->EvalAdd(downsampledchannels, masked);
    //     downsampledchannels = context->EvalRotate(downsampledchannels, -(inputSize - outputSize));
    // }

    // downsampledchannels = context->EvalRotate(downsampledchannels, (inputSize - outputSize) * numChannels);
    // downsampledchannels = context->EvalAdd(downsampledchannels, context->EvalRotate(downsampledchannels, -totalSize));

    return downsampledchannels;
}

/**
 * @brief Generate binary mask selecting elements after striding.
 *
 * @param width Width.
 * @param inputSize Input size.
 * @param stride Stride.
 * @param level Ciphertext level.
 * @return Plaintext mask.
 */
Ptext FHEONANNController::first_mask(int width, int inputSize, int stride, int level) {
    vector<double> mask(inputSize, 0);
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            if (j % stride == 0 && i % stride == 0) {
                int index = i * width + j;
                mask[index] = 1.0;
            }
        }
    }
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}

/**
 * @brief Generate binary mask selecting elements at stride positions.
 *
 * @param pattern Stride pattern index.
 * @param inputSize Size of input.
 * @param stride Stride size.
 * @param level Ciphertext level.
 * @return Plaintext mask.
 */
Ptext FHEONANNController::generate_binary_mask(int pattern, int inputSize, int stride, int level) {
    vector<double> mask;
    int copy_interval = pattern;
    for (int i = 0; i < inputSize; i++) {
        if (copy_interval > 0) {
            mask.push_back(1);
        } else {
            mask.push_back(0);
        }

        copy_interval--;

        if (copy_interval <= -pattern) {
            copy_interval = pattern;
        }
    }
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}

/**
 * @brief Generate mask for selecting specific row.
 *
 * @param row Row index to select.
 * @param width Width of the input.
 * @param inputSize Total number of elements in the input.
 * @param stride Stride size.
 * @param level Encryption level.
 * @return Plaintext mask.
 */
Ptext FHEONANNController::generate_row_mask(int row, int width, int inputSize, int stride, int level) {
    vector<double> mask;

    for (int j = 0; j < (row * width); j++) {
        mask.push_back(0);
    }
    for (int j = 0; j < width; j++) {
        mask.push_back(1);
    }
    for (int j = 0; j < (inputSize - width - (row * width)); j++) {
        mask.push_back(0);
    }
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}

/**
 * @brief Generate zero-valued mask of specified size.
 *
 * @param size Mask size.
 * @param level Ciphertext level.
 * @return Plaintext mask.
 */
Ptext FHEONANNController::generate_zero_mask(int size, int level) {
    vector<double> mask(size, 0.0);
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}

/**
 * @brief Generate full channel mask.
 *
 * @param n Mask index.
 * @param in_elements Input elements count.
 * @param out_elements Output elements count.
 * @param numChannels Channels count.
 * @param level Ciphertext level.
 * @return Plaintext mask.
 */
Ptext FHEONANNController::generate_channel_full_mask(int n, int in_elements, int out_elements, int numChannels, int level) {

    const int totalSlots = in_elements * numChannels;
    std::vector<double> mask(totalSlots, 0.0);
    const int base = n * in_elements;
    for (int i = 0; i < out_elements; ++i){
        mask[base + i] = 1.0;
    } 
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}

/**
 * @brief Generate zero mask across multiple channels.
 *
 * @param inputSize Input size.
 * @param numChannels Channels count.
 * @param level Ciphertext level.
 * @return Plaintext mask.
 */
Ptext FHEONANNController::generate_zero_mask_channels(int inputSize, int numChannels, int level) {
    
    int totalSlots = inputSize * numChannels;
    vector<double> mask(totalSlots, 0.0);
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}

/**
 * @brief Generate first mask for multiple channels.
 *
 * @param inputWidth Input width.
 * @param inputSize Input size.
 * @param stride Stride value.
 * @param numChannels Channels count.
 * @param level Ciphertext level.
 * @return Plaintext mask.
 */
Ptext FHEONANNController::first_mask_with_channels(int inputWidth, int inputSize, int stride, int numChannels, int level) {
    // int outputWidth = inputWidth / stride;
    vector<double> mask;
    vector<double> baseMask(inputSize, 0.0);
    for (int i = 0; i < inputWidth; i++) {
        for (int j = 0; j < inputWidth; j++) {
            if (j % stride == 0 && i % stride == 0) {
                int index =(i * inputWidth + j);
                baseMask[index] = 1.0;
            }
        }
    }

    for (int ch = 0; ch < numChannels; ch++) {
        mask.insert(mask.end(), baseMask.begin(), baseMask.end());
    }
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}

/**
 * @brief Generate binary mask supporting multiple channels.
 *
 * @param pattern Pattern index.
 * @param inputSize Input size.
 * @param stride Stride value.
 * @param numChannels Channels count.
 * @param level Ciphertext level.
 * @return Plaintext mask.
 */
Ptext FHEONANNController::generate_binary_mask_with_channels(int pattern, int inputSize, int stride,  int numChannels, int level) {
   
    vector<double> baseMask;
    int copy_interval = pattern;
    for (int i = 0; i < inputSize; i++) {
        if (copy_interval > 0) {
            baseMask.push_back(1);
        } else {
            baseMask.push_back(0);
        }

        copy_interval--;

        if (copy_interval <= -pattern) {
            copy_interval = pattern;
        }
    }

    // repeat baseMask n times
    vector<double> mask;
    mask.reserve(baseMask.size() * numChannels);
    for (int i = 0; i < numChannels; i++) {
        mask.insert(mask.end(), baseMask.begin(), baseMask.end());
    }

    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}

/**
 * @brief Generate row mask supporting multiple channels.
 *
 * @param row Row index.
 * @param width Width.
 * @param inputSize Input size.
 * @param stride Stride value.
 * @param numChannels Channels count.
 * @param level Ciphertext level.
 * @return Plaintext mask.
 */
Ptext FHEONANNController::generate_row_mask_with_channels(int row, int width, int inputSize, int stride, int numChannels, int level) {
    
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
    mask.reserve(baseMask.size() * numChannels);
    for (int i = 0; i < numChannels; i++) {
        mask.insert(mask.end(), baseMask.begin(), baseMask.end());
    }

    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}

/**
 * @brief Generate channel mask selecting one channel and zeroing others.
 *
 * @param channel Channel index.
 * @param outputSize Channel size.
 * @param numChannels Channels count.
 * @param level Ciphertext level.
 * @return Plaintext mask.
 */
Ptext FHEONANNController::generate_channel_mask_with_zeros(int channel, int outputSize, int numChannels, int level ){
    
    int totalSlots = outputSize * numChannels;
    vector<double> mask(totalSlots, 0.0);

    int pos = channel * outputSize;
    for (int i = 0; i < outputSize; i++) {
        mask[pos + i] = 1.0;
    }
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}

