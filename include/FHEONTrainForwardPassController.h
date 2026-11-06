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

#ifndef FHEON_TRAINFORWARDPASSCONTROLLER_H
#define FHEON_TRAINFORWARDPASSCONTROLLER_H

#include <openfhe.h>
#include <thread>
// #include <cereal/types/polymorphic.hpp> // Include this header

#include "FHEONHEController.h"
#include "FHEONANNController.h"
#include "FHEONTrainSupportController.h"

#include "FHEONTrainForwardPassController.h"

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
 * @brief Forward pass of the training pipeline, standard layout (one channel of the batch per ciphertext).
 *
 * Convolution, 1x1 shortcut, batch normalization, ReLU, average and global average pooling, fully connected layer and
 * softmax with encrypted (trainable) parameters, one broadcast scalar per ciphertext; every primitive caches what the
 * backward pass needs. With pretrainedFromCsv, parameters without a trained ciphertext yet are read from the PyTorch CSV
 * exports. FHEONTrainStandardForwardPassController is the multithreaded, optimized version (same files and keys).
 */
class FHEONTrainForwardPassController : public FHEONTrainSupportController, protected FHEONANNController {

protected:
    CryptoContext<DCRTPoly> context;

public:
    int num_slots = 1 << 14;
    int baseIndex = 1024;

    // Constructor must forward CryptoContext to the base class
    FHEONTrainForwardPassController(CryptoContext<DCRTPoly> ctx)
    : FHEONTrainSupportController(ctx), FHEONANNController(ctx), context(ctx) {}
    
    void setContext(CryptoContext<DCRTPoly>& in_context);
    void setNumSlots(int numSlots){
        num_slots = 1<< numSlots;
    }

   
    // pretrainedFromCsv=true (opt-in, default false so TrainModel.cpp's ConvNet is
    // completely unaffected): a missing weight/bias is read from the matching CSV and
    // used as Ptext in the forward computation. The same scalar is also lazily encrypted
    // and stored under the parameter filename because backward_pass_convolution needs a
    // Ctext for dInput and optimizer updates. This avoids an up-front import step, but a
    // full fine-tuning pass still materializes one large ciphertext per parameter; see
    // ImportPretrainedResNet20Weights.cpp for the resulting storage cost. Existing
    // ciphertext parameters are used directly and remain the values backward updates.
    vector<Ctext> he_forward_convolution_optimized(vector<Ctext> &encryptedInputs, int batchSize, int inputWidth, int inputChannels, int outputChannels, int stride, string layerName="", bool pretrainedFromCsv = false);
    // pretrainedFromCsv: same lazy CSV/Ptext use and ciphertext persistence as
    // he_forward_convolution_optimized, applied to this layer's dense (oc,in) weight
    // matrix -- CSV read from weights_folder+layerName+"_weight.csv"/"_bias.csv",
    // idx = oc*inputChannels+inCh.
    vector<Ctext> he_forward_convolution_shortcut_optimized(vector<Ctext> &encryptedInputs, int batchSize, int inputWidth, int inputChannels, int outputChannels, int stride, string layerName="", bool pretrainedFromCsv = false);
    vector<Ctext> he_forward_global_avgpool(vector<Ctext>& encryptedInputs, int batchSize, int dataWidth, int inputChannels, string layerName="");
    vector<Ctext> he_forward_relu(vector<Ctext>& encryptedInputs, vector<int> scaleValues, int inputChannels, int vectorSize, int polyDegree=59, string layerName="");
    vector<Ctext> he_forward_avgPool_optimized(vector<Ctext>& encryptedInputs, int batchSize, int inputWidth, int kernelWidth, int stride=2, string layerName="");

    // pretrainedFromCsv: same lazy CSV/Ptext use and ciphertext persistence, CSV read
    // from weights_folder+layerName+"_weight.csv"/"_bias.csv", idx = ou*inputSize+inCh
    // (nn.Linear's own flat layout).
    vector<Ctext> he_forward_linear_multiple_outputs(vector<Ctext>& encryptedInput, int vectorSize, int inputSize, int outputSize, string layerName="", bool pretrainedFromCsv = false);
    vector<Ctext> he_forward_softmax(vector<Ctext>& encryptedInputs, const vector<int>& scalingValues, int inputSize, int appxPolyDegree=59);
    // pretrainedFromCsv: gamma/beta read from weights_folder+layerName+"_gamma.csv"/
    // "_beta.csv" (one value per channel, see python/training/export_tuneresnet20_weights.py's
    // export_bn) instead of identity init. Missing CSV-sourced values are also persisted
    // as ciphertext parameters for backward updates.
    vector<Ctext> he_forward_batch_norm(vector<Ctext> &encryptedInputs, int inputSize, double appxInitValue, int appxDegree = 3, string layerName = "", bool isBootstrap = true, bool pretrainedFromCsv = false);

protected:
    Ctext newtonraphson_approximate_inverse_sqrt(Ctext& encryptedInput, int inputSize, double initValue = 1.0, int appxDegree = 3, bool isBootstrap = true);
    Ctext repeat_encrypted_values(Ctext& encryptedInput, int vecSize, int batchSize);
    Ptext gen_zero_mask(int size, int level); 
    Ptext gen_row_mask_with_channels(int row, int width, int inputSize, int batchSize, int level);
    Ptext gen_channel_mask_with_zeros(int channel, int outputSize, int numChannels, int level );
    

};

#endif // FHEON_TRAINFORWARDPASSCONTROLLER_H