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

#ifndef FHEON_ANNTRAININGCONCROLLER_H
#define FHEON_ANNTRAININGCONCROLLER_H

#include <openfhe.h>
#include <thread>

#include "FHEONHEController.h"
#include "FHEONANNStandardBatchController.h"
#include "FHEONTrainSupportController.h"

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
 * @brief Backward pass of the training pipeline, standard layout.
 *
 * Gradients of the cross-entropy + softmax, fully connected, convolution (stride 1, and stride 2 through a backward
 * dilation), shortcut, batch normalization, ReLU and pooling layers, with the SGD update of every encrypted parameter written
 * back (optimizer_update). The backward dilation exists in three variants (backward_dilate, backward_dilate_optimized,
 * backward_dilate_logdepth) trading operations against levels. FHEONTrainStandardBackwardPassController is the multithreaded,
 * optimized version (same files and keys).
 */
class FHEONTrainBackwardPassController : public FHEONTrainSupportController{

private:
    // Non-owning pointer, set via setHEController — used only to decrypt-and-print
    // sample weight/bias/gradient scalars for comparison against testing.py's
    // backward-pass prints. Left null (all prints skipped) unless wired up.
    FHEONHEController* heController = nullptr;

public:
    int num_slots = 1 << 14;
    int baseIndex = 1024;

    void setHEController(FHEONHEController* controller) {
        heController = controller;
    }

    FHEONTrainBackwardPassController(CryptoContext<DCRTPoly>& ctx)
        : FHEONTrainSupportController(ctx) {}

    void setNumSlots(int numSlots){
        num_slots = 1<< numSlots;
    }

    void welcomeToTrainingMessage();
    
    vector<Ctext> softmax(vector<Ctext> &encryptedInput, int inputSize, int appxPolyDegree = 59);

    Ctext cross_entropy_loss(vector<Ctext> &encryptedInputs, vector<double> &labels, int batchSize, int numClasses, int appxDegree=3);
    vector<Ctext> backward_pass_cross_entropy_and_softmax(vector<Ctext> &encryptedInputs, vector<double> &labels, int batchSize, int vectorSize, string layerName="");

    vector<Ctext> backward_pass_relu(vector<Ctext> &gradInputs, vector<int>& scalers,int appxPolyDegree, string layerName = "");
    // pretrainedFromCsv: see he_forward_batch_norm's matching doc comment -- same
    // gamma/beta CSV fallback, only kicks in on a gamma/beta channel that has never
    // been trained (updated by this function) before.
    vector<Ctext> backward_pass_batch_norm(vector<Ctext> &encryptedInputs, vector<Ctext> &gradInputs, int inputSize, double learningRate, string layerName = "");

    // pretrainedFromCsv: see he_forward_linear_multiple_outputs's matching doc comment.
    vector<Ctext> backward_pass_fully_connected(vector<Ctext> &gradInputs, int vectorSize, int inputSize, int outputSize, double learningRate, string layerName);
    // pretrainedFromCsv: see FHEONTrainForwardPassController::he_forward_convolution_optimized's
    // matching doc comment -- same opt-in, same default-false (no effect on TrainModel.cpp).
    // A tap without a trained ciphertext yet is read from weights_folder+layerName+
    // "_weight.csv" as a plaintext for the dInput computation, and its optimizer_update
    // is done against that same plaintext (see the Ptext overload below) -- this call
    // is what actually creates that tap's FIRST trained ciphertext on disk.
    vector<Ctext> backward_pass_convolution(vector<Ctext> &gradInputs, int batchSize, int inputWidth,
                    int inputChannels, int outputChannels, int kernelWidth, double learningRate, string layerName, int stride = 1);
    vector<Ctext> backward_pass_avgpool(vector<Ctext> &encryptedInputs, int batchSize, int inputWidth, int poolSize);
    vector<Ctext> backward_pass_avgpool_legacy(vector<Ctext> &encryptedInputs, int batchSize, int inputWidth, int poolSize);  // deprecated backup, do not call
    // pretrainedFromCsv: see he_forward_convolution_shortcut_optimized's matching doc comment.
    vector<Ctext> backward_pass_convolution_shortcut(vector<Ctext> &gradInputs, int batchSize, int inputWidth,
                    int inputChannels, int outputChannels, int stride, double learningRate, string layerName);
    vector<Ctext> backward_pass_global_avgpool(vector<Ctext> &gradInputs, int batchSize, int dataWidth);

    // Optimized variants (see backward_dilate_optimized's doc comment for the
    // saving) -- added alongside, not in place of, the originals above so the
    // originals stay available as the known-good reference; see
    // validate_avgpool_optimized/validate_global_avgpool_optimized in
    // HETrainingModels/TrainingModules.cpp for a direct numeric comparison of
    // old vs new on identical inputs. Not yet wired into tuneResNet20.cpp's
    // real per-stage key sets -- see backward_dilate_optimized_indices's own
    // doc comment for what a caller needs to add before using these for real.
    vector<Ctext> backward_pass_avgpool_optimized(vector<Ctext> &encryptedInputs, int batchSize, int inputWidth, int dilateFactor);
    vector<Ctext> backward_pass_global_avgpool_optimized(vector<Ctext> &gradInputs, int batchSize, int dataWidth);
    Ctext backward_dilate_optimized(Ctext& encryptedInput, int batchSize, int inputWidth, int dilateFactor);
    vector<int> backward_dilate_optimized_indices(int batchSize, int inputWidth, int dilateFactor);

    // Log-depth variant: same output as backward_dilate/backward_dilate_optimized,
    // built by transposing (adjointing) forward's downsample_with_multiple_channels
    // masked-doubling compaction network (FHEONANNController.cpp) instead of handling
    // each column independently -- see backward_dilate_logdepth's own doc comment for
    // the derivation and the real tradeoff (fewer total ops, but more consumed levels
    // than backward_dilate_optimized -- chained mask-multiplies, not independent
    // branches). inputWidth must be a power of two.
    vector<Ctext> backward_pass_avgpool_logdepth(vector<Ctext> &encryptedInputs, int batchSize, int inputWidth, int dilateFactor);
    vector<Ctext> backward_pass_global_avgpool_logdepth(vector<Ctext> &gradInputs, int batchSize, int dataWidth);
    Ctext backward_dilate_logdepth(Ctext& encryptedInput, int batchSize, int inputWidth, int dilateFactor);
    vector<int> backward_dilate_logdepth_indices(int batchSize, int inputWidth, int dilateFactor);

    // Helper functions for training primitives
    Ctext newtonraphson_approximate_inverse_sqrt(Ctext& encryptedInput, int inputSize, double initValue = 1.0, int appxDegree = 3);
    Ctext taylors_approximation(Ctext& encryptedInput, int appxDegree);
    
    Ctext repeat_encrypted_values(Ctext& encryptedInput, int vecSize, int batchSize);
    Ctext optimizer_update(Ctext& encryptedParam, Ctext& encryptedGrad, double learningRate);
    // For a param that has never been trained yet (still just a pretrained plaintext
    // constant, not a ciphertext) -- EvalSub(Plaintext, Ciphertext) produces the
    // first-ever trained Ctext for that param, matching the non-Ptext overload's formula.
    Ctext optimizer_update(Ptext& constParam, Ctext& encryptedGrad, double learningRate);
    Ctext backward_dilate(Ctext& encryptedInput, int batchSize, int inputWidth, int dilateFactor);
   
   
    vector<int> backward_dilate_indices(int batchSize, int inputWidth, int dilateFactor);

    // Plaintext mask with a single 1 at slot 0, else 0 -- isolates "the value
    // currently rotated into slot 0" from everything else packed alongside it.
    // Used by backward_dilate_optimized (see its doc comment).
    Ptext gen_slot0_mask(uint32_t level);

    // Mask helpers for backward_dilate_logdepth (see its doc comment for the
    // derivation) -- mirror FHEONANNController's first_mask_with_channels /
    // generate_binary_mask_with_channels, reimplemented here (rather than
    // reused across classes) since they're plain periodic 0/1 plaintext
    // patterns with no dependency on that class's state.
    Ptext gen_prefix_mask(int prefixLen, uint32_t level);        // 1 at slots [0, prefixLen), else 0
    Ptext gen_binary_run_mask(int pattern, uint32_t level);      // period 2*pattern, 1 for the first `pattern` of each period
    Ptext gen_stride_mask(int period, uint32_t level);           // 1 at slots that are multiples of `period`, else 0
    Ctext column_scatter_log_depth(Ctext& rowInput, int inputWidth, int dilateFactor);

};

#endif