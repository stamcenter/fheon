
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
#include "FHEONTrainSupportController.h"
#include "UtilsTraining.h"


namespace fs = std::filesystem;

void FHEONTrainSupportController::setContext(CryptoContext<DCRTPoly>& in_context){
    context = in_context;
}


/***
 * welcome message for training controller */
void FHEONTrainSupportController::welcomeToTrainingMessage(){
    cout << endl << "****************************************************************************************************" << endl;
    cout << endl << "******************************* WELCOME TO FHEON TRAINING CONTROLLER! *******************************" << endl;
    cout << "This controller is designed to manage and orchestrate the training process "
            "of homomorphically encrypted neural networks. It provides functionalities "
            "to set up the environment, manage encrypted data, and coordinate the "
            "training workflow." << endl;
    cout << "Let's get started on your secure and private machine learning journey!" << endl;
    cout << endl << "****************************************************************************************************" << endl
        << endl;
}


/**
 * @brief Generates rotation positions for the convolution operation in the encrypted domain. 
 * This function calculates the necessary rotation indices based on the batch size, input width, kernel width, padding, and stride. 
 * The generated rotation positions are used to align the encrypted data correctly during the convolution process. 
 * 
 * @param batchSize The number of batches encoded in the input ciphertexts.
 * @param inputWidth The width (and height) of the input feature maps (assumed square).
 * @param kernelWidth The width (and height) of the convolution kernel (assumed square).
 * @param padding The number of zero-padding slots applied around the input feature map.
 * @param stride The stride length used for the convolution operation.
 *  
 * @return vector<int> A vector of rotation positions required for the convolution operation.
 * 
*/
vector<int> FHEONTrainSupportController::generate_convolution_batch_rotation_positions(int batchSize, int inputWidth, int kernelWidth, int padding, int stride) {
        
    vector<int> keys_position;
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
vector<int> FHEONTrainSupportController::generate_avgpool_batch_optimized_rotation_positions(int batchSize, int inputWidth, int kernelWidth, int stride, 
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
vector<int> FHEONTrainSupportController::generate_linear_batch_rotation_positions(int batchSize, vector<int> outputSizes, vector<int> inputSizes, int rotationIndex) {
    
    vector<int> keys_position;
    // int chunkSize = 128;
    for(size_t k =0; k<inputSizes.size(); k++){
        keys_position.push_back(inputSizes[k]);
    }

    for(size_t i=0; i<outputSizes.size(); i++){
        for(int j=1; j<=outputSizes[i]; j++){
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
 * @brief Generate rotation positions for normalization layers.
 *
 * This function calculates the rotation positions required for performing
 * normalization operations (e.g., batch normalization) on encrypted data.
 * It generates indices based on the input size to optimize the rotations
 * needed for computing mean and variance across the feature maps.
 *
 * @param inputSize      Size of the input feature maps (width × height).
 *
 * @return vector<int>   Vector of rotation positions required for normalization.
 */
vector<int> FHEONTrainSupportController::generate_normalization_rotation_positions(int inputSize) {
    vector<int> keys_position;
    keys_position.push_back(-1);

   for(int j=1; j<log2(inputSize); j++){
        int powVal = pow(2, j);
        keys_position.push_back(-powVal);
    }
    std::sort(keys_position.begin(), keys_position.end());
    auto new_end = std::unique(keys_position.begin(), keys_position.end());
    keys_position.erase(new_end, keys_position.end());
    return keys_position;
}


/**
 * @brief Generate rotation positions for backpropagation in convolution layers.
 *
 * This function calculates the rotation positions required for performing
 * backpropagation operations in convolution layers on encrypted data. It
 * generates indices based on the input size and kernel width to optimize
 * the rotations needed for computing gradients during the training process.
 *
 * @param inputSize      Size of the input feature maps (width × height).
 * @param kernelWidth    Width (and height) of the convolution kernel (assumed square).
 *
 * @return vector<int>   Vector of rotation positions required for backpropagation in convolution layers.
 */
vector<int> FHEONTrainSupportController::generate_backpass_convolution_rotation_positions(int inputSize, int kernelWidth) {
    vector<int> keys_position;
    int kernelSize = kernelWidth * kernelWidth;

   for(int j=1; j<kernelSize; j++){
        keys_position.push_back(j);
    }
    std::sort(keys_position.begin(), keys_position.end());
    auto new_end = std::unique(keys_position.begin(), keys_position.end());
    keys_position.erase(new_end, keys_position.end());
    return keys_position;
}

/**
 * @brief Generate a zero mask of given size.
 *
 * @param size Number of elements in the mask.
 * @param level Encryption level for CKKS plaintext.
 * @return Packed plaintext mask with all zeros.
 */
Ptext FHEONTrainSupportController::gen_zero_mask(int size, int level) {
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
Ptext FHEONTrainSupportController::gen_row_mask_with_channels(int row, int width, int inputSize, int batchSize, int level) {
    
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
Ptext FHEONTrainSupportController::gen_channel_mask_with_zeros(int channel, int outputSize, int numChannels, int level ){
    
    int totalSlots = outputSize * numChannels;
    vector<double> mask(totalSlots, 0.0);

    int pos = channel * outputSize;
    for (int i = 0; i < outputSize; i++) {
        mask[pos + i] = 1.0;
    }
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}


/**
 * Serialization and storage function.
 * This function serializes an encrypted ciphertext and stores it in a file for later retrieval.
 *
 * Deliberately does NOT catch/swallow failures here (std::filesystem remove / create_directory
 * throwing, or SerializeToFile returning false all propagate/throw out of this call) --
 * a caller that silently ignores a failed write here leaves whatever ciphertext was
 * PREVIOUSLY at that path (from an earlier, possibly incompatible run) sitting there
 * unchanged, ready to be deserialize_and_load'd later as if it were fresh. On a shared
 * filesystem (NFS/scratch storage under concurrent jobs) that silent failure mode is
 * exactly what produced a real "Ciphertexts were not created in the same CryptoContext"
 * crash, hours into a run, in a spot with no obvious connection to this function --
 * failing loudly and immediately here, at the actual point of failure, is far cheaper
 * to diagnose than that.
 *
 * @param encryptedInput  The encrypted ciphertext to be serialized and stored.
 * @param fileName        The name of the file where the encrypted ciphertext will be stored.
 */
void FHEONTrainSupportController::serialize_and_store(Ctext& encryptedInput, string& fileName){
    string dirPath = training_data_dir;
    if (!fs::exists(dirPath)) {
        fs::create_directory(dirPath);
    }
    string sfileName = dirPath + fileName+ ".bin";
    if (fs::exists(sfileName)) {
        fs::remove(sfileName);
    }
    if (!lbcrypto::Serial::SerializeToFile(sfileName, encryptedInput, lbcrypto::SerType::BINARY)) {
        throw std::runtime_error("serialize_and_store: failed to write \"" + sfileName + "\"");
    }
}

/***
 * Encrypts and caches a single externally-supplied constant (see header doc comment).
 */
void FHEONTrainSupportController::import_scalar_param(double value, string fileName, int vectorSize){
    if (!encryptedOnes) {
        cerr << "Error in import_scalar_param: encryptedOnes is NULL!" << endl;
        std::abort();
    }
    // Values with a subnormal-range magnitude (e.g. ~1e-40 float export artifacts from
    // PyTorch) break CKKS's scaling-factor selection during Encode() ("Scaling factor
    // too small") -- exact 0.0 encodes fine (no precision to represent), so clamp near-
    // zero values down to it instead, matching UtilsBatchData.h's load_batch_weights's
    // own existing convention for the exact same reason.
    if (std::abs(value) < 1e-10) {
        value = 0.0;
    }
    size_t numSlots = (vectorSize > 0) ? (size_t)vectorSize : (context->GetRingDimension() / 2);
    vector<double> init_vec(numSlots, value);
    auto pt = context->MakeCKKSPackedPlaintext(init_vec, 1, encryptedOnes->GetLevel());
    Ctext paramCtxt = context->EvalMult(encryptedOnes, pt);
    serialize_and_store(paramCtxt, fileName);
}

/***
 * Deserialization and loading function.
 * This function loads an encrypted ciphertext from a file and deserializes it for use in computations.
 *
 * @param fileName  The name of the file from which to load the encrypted ciphertext.
 *
 * @return The deserialized encrypted ciphertext.
 */
Ctext FHEONTrainSupportController::deserialize_and_load(string& fileName, double fan_in, int vectorSize){
   
    string fullFileName = training_data_dir + fileName+ ".bin";
    Ctext deseriazedInput;
    
    // Try direct deserialization with exception handling
    // cout << "deserialized and loaded ctxt name: " << fullFileName << endl;
    try {
        if (lbcrypto::Serial::DeserializeFromFile(fullFileName, deseriazedInput, lbcrypto::SerType::BINARY)) {
            return deseriazedInput;
        }
    } catch (const std::exception& e) {
        // File corrupted or inaccessible; will create placeholder below
        cerr << "Warning: Deserialization failed for '" << fileName << "': " << e.what() << endl;
    }
    
    // If this IS a placeholder file (encryptedZeros/Ones), fail—don't recurse.
    if (fileName == encryptedZerosFile || fileName == encryptedOnesFile) {
        cerr << "Fatal: placeholder file '" << fileName << "' is missing/corrupted. Cannot recover." << endl;
        std::abort();
    }
    
    // File missing/corrupted; use the in-memory encrypted placeholder with dynamic initialization
    if (!encryptedOnes) {
        cerr << "Error in deserialize_and_load: encryptedOnes is NULL!" << endl;
        std::abort();
    }
    size_t numSlots = (vectorSize > 0) ? (size_t)vectorSize : (context->GetRingDimension() / 2);
    Ctext placeholderCtxt;
    try {
        if (fileName.find("bias") != string::npos) {
            std::vector<double> init_vec = utilstraining::init_zeros(numSlots);
            auto pt = context->MakeCKKSPackedPlaintext(init_vec, 1, encryptedOnes->GetLevel());
            placeholderCtxt = context->EvalMult(encryptedOnes, pt);
        } else if (fileName.find("batch_norm") != string::npos || fileName.find("bn") != string::npos) {
            std::vector<double> init_vec = utilstraining::init_ones(numSlots);
            auto pt = context->MakeCKKSPackedPlaintext(init_vec, 1, encryptedOnes->GetLevel());
            placeholderCtxt = context->EvalMult(encryptedOnes, pt);
        } else {
            std::vector<double> init_vec = utilstraining::kaiming_uniform(numSlots, fan_in);
            auto pt = context->MakeCKKSPackedPlaintext(init_vec, 1, encryptedOnes->GetLevel());
            placeholderCtxt = context->EvalMult(encryptedOnes, pt);
        }
    } catch (const std::exception& e) {
        cerr << "Error during fallback initialization for '" << fileName << "': " << e.what() << endl;
        throw;
    }
    
    serialize_and_store(placeholderCtxt, fileName);

    return placeholderCtxt;
}
