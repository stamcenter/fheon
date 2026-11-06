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

#ifndef FHEON_ANNTRAINSUPPORTCONTROLLER_H
#define FHEON_ANNTRAINSUPPORTCONTROLLER_H

#include <openfhe.h>
#include <thread>
// #include <cereal/types/polymorphic.hpp> // Include this header

#include "Utils.h"
#include "UtilsData.h"
#include "UtilsBatchData.h"

#include "FHEONHEController.h"
#include "FHEONANNController.h"

using namespace lbcrypto;
using namespace std;

/** secure_anncontroller defined utils */
using namespace utils;
using namespace utilsdata;
using namespace utilsbatchdata;

/**
 * @brief Shared services of the standard training controllers.
 *
 * Storage of the encrypted parameters and caches under training_data_dir (serialize_and_store, deserialize_and_load with
 * Kaiming / zero / one initialization of a missing parameter, import_scalar_param for pretrained values) and the
 * rotation-key generators of the training primitives.
 */
class FHEONTrainSupportController {


protected:
    string users_prefix = "./../encrypted_users/";
    string users_database = "./../encrypted_db/";
    string encryptedZerosFile = "00_encrypted_zeros";
    string encryptedOnesFile = "01_encrypted_ones";

    Ctext encryptedZeros;
    Ctext encryptedOnes;

protected:
    CryptoContext<DCRTPoly> context;

public:

    FHEONTrainSupportController( CryptoContext<DCRTPoly> ctx) {
        context = ctx;
    }
    void setContext(CryptoContext<DCRTPoly>& in_context);

    void setEncryptedZeros(Ctext& zero) {
        encryptedZeros = zero;
        serialize_and_store(encryptedZeros, encryptedZerosFile);
    }

    void setEncryptedOnes(Ctext& ones) {
        encryptedOnes = ones;
        serialize_and_store(encryptedOnes, encryptedOnesFile);
    }

    void welcomeToTrainingMessage();

    vector <int> generate_convolution_batch_rotation_positions(int batchSize, int inputWidth, int kernelWidth, int padding=0, int stride=1);
    vector <int> generate_avgpool_batch_optimized_rotation_positions(int batchSize, int inputWidth, int kernelWidth, int stride=2, 
                            bool globalPooling=false, int rotationIndex=16);
    vector <int> generate_linear_batch_rotation_positions(int batchSize, vector<int> outputSizes, vector<int> inputSizes, int rotationIndex=100);
    vector <int> generate_normalization_rotation_positions(int inputSize);
    vector<int> generate_backpass_convolution_rotation_positions(int inputSize, int kernelWidth);

    void serialize_and_store(Ctext& encryptedInput, string& fileName);
    Ctext deserialize_and_load(string& fileName, double fan_in = 1.0, int vectorSize = -1);

    // Encrypts a single externally-supplied constant (e.g. a pretrained PyTorch
    // weight/bias/BN-gamma/BN-beta scalar) and caches it under fileName using the
    // exact same broadcast-and-store construction deserialize_and_load's cache-miss
    // branch uses for its random/fallback init -- so a real forward/backward call
    // later reads this imported value on its first deserialize_and_load instead of
    // falling back to Kaiming/ones/zeros init. Over-provisions to numSlots (or
    // vectorSize, if given) uniformly since every value here is a constant
    // broadcast; unlike deserialize_and_load, no fan_in/bias/bn filename sniffing
    // is needed since the caller already knows exactly what value to write.
    void import_scalar_param(double value, string fileName, int vectorSize = -1);
    
private:
    Ptext gen_zero_mask(int size, int level); 
    Ptext gen_row_mask_with_channels(int row, int width, int inputSize, int batchSize, int level);
    Ptext gen_channel_mask_with_zeros(int channel, int outputSize, int numChannels, int level );
    

};

#endif // FHEON_ANNTRAINSUPPORTCONTROLLER_H