
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
 * Per-module backward-pass validation harness.
 *
 * TrainModel.cpp chains all layers back-to-back, so a bug in any one backward
 * function is only visible several (very expensive) layers downstream, mixed
 * in with every other layer's approximation noise. This program instead runs
 * each backward_pass_* function ONCE, in isolation, against small
 * deterministically-generated inputs/weights/upstream-gradients (fixed
 * sin()/cos() formulas, not random) and prints the results.
 *
 * python/training/validate_modules.py regenerates the exact same
 * deterministic values (same formulas, same constants) and computes each
 * module's forward+backward pass independently in plain numpy — no shared
 * code path with the C++ side. Diffing the two programs' printed output is
 * the actual validation step (run both, compare by eye/script); this file
 * does not attempt to assert correctness itself since there is no ground
 * truth available on the C++ side alone.
 *
 * All test dimensions deliberately reuse EXACT (batchSize, width) pairs from
 * the real network (see TrainModel.cpp's channels/dataWidths/batchSizes) so
 * every rotation key these modules need already exists in the real,
 * already-generated keyset — only channel counts are shrunk (2 in / 2 out)
 * to keep each isolated test fast. Every cache file this program reads or
 * writes is prefixed "validate_" so it can never collide with real training
 * data in encrypted_training_data/.
 */

#include <iostream>
#include <cmath>
#include <sys/stat.h>

#include "FHEONHEController.h"
#include "FHEONANNController.h"
#include "FHEONTrainSupportController.h"
#include "FHEONTrainForwardPassController.h"
#include "FHEONTrainBackwardPassController.h"

using namespace std;

#ifndef DEFAULT_TEST_SIZE
#define DEFAULT_TEST_SIZE 256
#endif

#ifndef INDEX_VALUE
#define INDEX_VALUE 0
#endif

CryptoContext<DCRTPoly> context;
FHEONHEController fheonHEController(context);
FHEONTrainSupportController fheonTrainSupportController(context);
FHEONTrainForwardPassController fheonTrainForwardPassController(context);
FHEONTrainBackwardPassController fheonTrainBackwardPassController(context);

// Real-network dimensions reused so existing rotation keys apply (see file header).
const int BATCH_SIZE = 16;
const int KERNEL_WIDTH = 3;
const int POOL_WIDTH = 2;
const double LEARNING_RATE = 0.01;
const int TEST_CH_IN = 2;
const int TEST_CH_OUT = 2;
const int CONV_WIDTH = 16;      // == dataWidths[1] (conv1/relu1/bn1/avgpool1 scope)
const int FC_VECSIZE = 1024;    // == bn2NormSize = batchSize * dataSizes[2]
const int AVGPOOL_SRC_WIDTH = 8; // == dataWidths[2], pooled/source width for the avgpool1-style call

/*** Deterministic synthetic value generator — NOT random. Python's
 * validate_modules.py regenerates the identical sequence from the same
 * (freq, phase, scale) constants, so results are reproducible across runs
 * and independently computable without shipping data between the two sides. */
vector<double> synth_vec(int size, double freq, double phase, double scale) {
    vector<double> v(size);
    for (int i = 0; i < size; i++) {
        v[i] = scale * sin(freq * (i + 1) + phase);
    }
    return v;
}

Ctext encrypt_constant(int size, double value) {
    vector<double> v(size, value);
    return fheonHEController.encrypt_input(v, false);
}

Ctext encrypt_synth(int size, double freq, double phase, double scale) {
    auto v = synth_vec(size, freq, phase, scale);
    return fheonHEController.encrypt_input(v, false);
}

void print_row(const string& label, Ctext& ctxt, int n = 8) {
    cout << "[VALIDATE] " << label << ": ";
    fheonHEController.decrypt_and_print(ctxt, n);
}

// Ground-truth-free correctness check for the backward_dilate_optimized family:
// unlike every other validator in this file (which needs python/training/validate_modules.py
// as an independent reference), the *_optimized functions are supposed to compute the exact
// same mathematical quantity as their originals, so decrypting both on identical inputs and
// diffing is a real pass/fail check, self-contained in this program.
double max_abs_diff(const vector<double>& a, const vector<double>& b) {
    double m = 0.0;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; i++) {
        double d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

/*** ============================== ReLU ============================== ***
 * Forward: y = ReLU(x) (Chebyshev approx over [-1,1]).
 * Backward: dInput = dOutput * (x > 0).
 * Python reference: exact step-function derivative (not the Chebyshev
 * approximation) — expect close-but-not-bitwise-identical agreement,
 * consistent with this project's established approximation-tolerance.
 */
void validate_relu() {
    cout << "\n===== VALIDATE: ReLU backward =====" << endl;
    int vecSize = BATCH_SIZE * CONV_WIDTH * CONV_WIDTH;
    string layerName = "validate_relu";

    vector<Ctext> preAct(TEST_CH_IN);
    for (int c = 0; c < TEST_CH_IN; c++) {
        preAct[c] = encrypt_synth(vecSize, 0.031, c * 1.7, 0.8); // in [-0.8, 0.8], inside Chebyshev's fitted domain
    }
    vector<int> scaleOnes(TEST_CH_IN, 1);
    // he_forward_relu caches preAct to disk under layerName — that cache is what backward_pass_relu reads back.
    fheonTrainForwardPassController.he_forward_relu(preAct, scaleOnes, TEST_CH_IN, vecSize, 59, layerName);

    vector<Ctext> gradOut(TEST_CH_IN);
    for (int c = 0; c < TEST_CH_IN; c++) {
        gradOut[c] = encrypt_synth(vecSize, 0.047, c * 0.9 + 3.0, 0.05);
    }

    cout << "  Params: vecSize=" << vecSize << ", channels=" << TEST_CH_IN
         << ", preAct=synth(freq=0.031,phase=1.7*c,scale=0.8), gradOut=synth(freq=0.047,phase=0.9*c+3.0,scale=0.05)" << endl;

    auto gradIn = fheonTrainBackwardPassController.backward_pass_relu(gradOut, scaleOnes, 59, layerName);
    for (int c = 0; c < TEST_CH_IN; c++) {
        print_row("relu gradInput channel " + to_string(c), gradIn[c]);
    }
}

/*** =========================== BatchNorm ============================ ***
 * Forward: xHat=(x-mean)/std, y=gamma*xHat+beta (mean/var over the WHOLE
 * per-channel vector, per this codebase's design — not per-batch-element).
 * Backward: standard BN backward formula; dGamma=sum(dOut*xHat), dBeta=sum(dOut).
 */
void validate_batch_norm() {
    cout << "\n===== VALIDATE: BatchNorm backward =====" << endl;
    int vecSize = BATCH_SIZE * CONV_WIDTH * CONV_WIDTH; // == bn1NormSize scope
    string layerName = "validate_bn";

    // Pre-seed deterministic gamma/beta so results don't depend on kaiming_uniform's process-order-dependent counter.
    for (int c = 0; c < TEST_CH_IN; c++) {
        string gammaFile = "batch_norm_" + layerName + "_channel_" + to_string(c);
        string betaFile = "batch_norm_bias_" + layerName + "_channel_" + to_string(c);
        auto gammaCt = encrypt_constant(vecSize, 1.0 + 0.1 * c);
        auto betaCt = encrypt_constant(vecSize, 0.05 * c);
        fheonTrainForwardPassController.serialize_and_store(gammaCt, gammaFile);
        fheonTrainForwardPassController.serialize_and_store(betaCt, betaFile);
    }

    vector<Ctext> preAct(TEST_CH_IN);
    for (int c = 0; c < TEST_CH_IN; c++) {
        preAct[c] = encrypt_synth(vecSize, 0.017, c * 2.1, 1.2);
    }
    // he_forward_batch_norm caches xHat + gamma*invstd under layerName — backward_pass_batch_norm reads both back.
    fheonTrainForwardPassController.he_forward_batch_norm(preAct, vecSize, 2.5, 3, layerName, true);

    vector<Ctext> gradOut(TEST_CH_IN);
    for (int c = 0; c < TEST_CH_IN; c++) {
        gradOut[c] = encrypt_synth(vecSize, 0.053, c * 1.3 + 2.0, 0.05);
    }

    cout << "  Params: vecSize=" << vecSize << ", channels=" << TEST_CH_IN
         << ", gamma[c]=1.0+0.1*c, beta[c]=0.05*c, preAct=synth(freq=0.017,phase=2.1*c,scale=1.2),"
         << " gradOut=synth(freq=0.053,phase=1.3*c+2.0,scale=0.05)" << endl;

    // First arg is unused by the implementation (see backward_pass_batch_norm) — reusing gradOut satisfies the signature.
    auto gradIn = fheonTrainBackwardPassController.backward_pass_batch_norm(gradOut, gradOut, vecSize, LEARNING_RATE, layerName);
    for (int c = 0; c < TEST_CH_IN; c++) {
        print_row("bn gradInput channel " + to_string(c), gradIn[c]);
    }
}

/*** ============= BatchNorm, sparse gradOut (real-network-density probe) ==== ***
 * Diagnostic for the "increasing divergence past BN2/BN1" question: in the
 * real chain, backward_pass_cross_entropy_and_softmax's gradient is nonzero
 * in only the first BATCH_SIZE=16 of vecSize slots (the rest were masked to
 * zero — see its own doc comment) — 1.5% real signal at BN2's 1024-wide
 * scope, 0.4% at BN1's 4096-wide scope. CKKS approximation noise is roughly
 * fixed in absolute magnitude regardless of how sparse the real signal is,
 * so if that dilution alone (not a second formula bug) explains the growing
 * mismatch against the plaintext reference deeper in the chain, this same
 * isolated backward_pass_batch_norm call — already proven formula-correct by
 * validate_batch_norm() above with a DENSE gradOut — should show comparably
 * degraded (small-signal, noise-dominated, possibly sign-flipped) agreement
 * here, at BN1's exact scope, without any other layer involved.
 */
void validate_batch_norm_sparse_grad() {
    cout << "\n===== VALIDATE: BatchNorm backward (sparse gradOut, BN1-density probe) =====" << endl;
    int vecSize = BATCH_SIZE * CONV_WIDTH * CONV_WIDTH; // 4096 == bn1NormSize scope
    int realCount = BATCH_SIZE; // 16 nonzero slots out of 4096, matching the real chain's masked density
    string layerName = "validate_bn_sparse";

    for (int c = 0; c < TEST_CH_IN; c++) {
        string gammaFile = "batch_norm_" + layerName + "_channel_" + to_string(c);
        string betaFile = "batch_norm_bias_" + layerName + "_channel_" + to_string(c);
        auto gammaCt = encrypt_constant(vecSize, 1.0 + 0.1 * c);
        auto betaCt = encrypt_constant(vecSize, 0.05 * c);
        fheonTrainForwardPassController.serialize_and_store(gammaCt, gammaFile);
        fheonTrainForwardPassController.serialize_and_store(betaCt, betaFile);
    }

    vector<Ctext> preAct(TEST_CH_IN);
    for (int c = 0; c < TEST_CH_IN; c++) {
        preAct[c] = encrypt_synth(vecSize, 0.017, c * 2.1, 1.2);
    }
    fheonTrainForwardPassController.he_forward_batch_norm(preAct, vecSize, 2.5, 3, layerName, true);

    vector<Ctext> gradOut(TEST_CH_IN);
    vector<vector<double>> gradOutVecs(TEST_CH_IN);
    for (int c = 0; c < TEST_CH_IN; c++) {
        auto v = synth_vec(vecSize, 0.053, c * 1.3 + 2.0, 0.05);
        for (int j = realCount; j < vecSize; j++) v[j] = 0.0;
        gradOutVecs[c] = v;
        gradOut[c] = fheonHEController.encrypt_input(v, false);
    }

    cout << "  Params: vecSize=" << vecSize << ", realCount=" << realCount << " (rest zeroed), channels=" << TEST_CH_IN
         << ", gamma[c]=1.0+0.1*c, beta[c]=0.05*c, preAct=synth(freq=0.017,phase=2.1*c,scale=1.2),"
         << " gradOut=synth(freq=0.053,phase=1.3*c+2.0,scale=0.05) zeroed past slot " << realCount << endl;

    auto gradIn = fheonTrainBackwardPassController.backward_pass_batch_norm(gradOut, gradOut, vecSize, LEARNING_RATE, layerName);
    for (int c = 0; c < TEST_CH_IN; c++) {
        print_row("bn(sparse) gradInput channel " + to_string(c), gradIn[c]);
    }
}

/*** ======================== Fully Connected ========================== ***
 * Forward: output[ou] = sum_in(weight[ou][in]*input[in]) + bias[ou].
 * Backward: dInput[in]=sum_ou(dOut[ou]*W[ou][in]); dW[ou][in]=sum_vec(dOut[ou]*input[in]); dBias[ou]=sum_vec(dOut[ou]).
 */
void validate_fully_connected() {
    cout << "\n===== VALIDATE: FullyConnected backward =====" << endl;
    int vecSize = FC_VECSIZE;
    int inputSize = TEST_CH_IN;
    int outputSize = TEST_CH_OUT;
    string layerName = "validate_fc";

    for (int ou = 0; ou < outputSize; ou++) {
        for (int inCh = 0; inCh < inputSize; inCh++) {
            string wFile = layerName + "_out_neuron_" + to_string(ou) + "_in_" + to_string(inCh);
            double val = 0.1 * sin(0.7 * (ou * 10 + inCh) + 1.0);
            auto wCt = encrypt_constant(vecSize, val);
            fheonTrainForwardPassController.serialize_and_store(wCt, wFile);
        }
        string bFile = layerName + "_out" + "_neuron_" + to_string(ou) + "_bias";
        auto bCt = encrypt_constant(vecSize, 0.01 * ou);
        fheonTrainForwardPassController.serialize_and_store(bCt, bFile);
    }

    vector<Ctext> input(inputSize);
    for (int inCh = 0; inCh < inputSize; inCh++) {
        input[inCh] = encrypt_synth(vecSize, 0.023, inCh * 1.9, 1.0);
    }
    fheonTrainForwardPassController.he_forward_linear_multiple_outputs(input, vecSize, inputSize, outputSize, layerName);

    vector<Ctext> gradOut(outputSize);
    for (int ou = 0; ou < outputSize; ou++) {
        gradOut[ou] = encrypt_synth(vecSize, 0.061, ou * 1.1 + 4.0, 0.05);
    }

    cout << "  Params: vecSize=" << vecSize << ", inputSize=" << inputSize << ", outputSize=" << outputSize
         << ", W[ou][in]=0.1*sin(0.7*(10*ou+in)+1.0), bias[ou]=0.01*ou,"
         << " input=synth(freq=0.023,phase=1.9*in,scale=1.0), gradOut=synth(freq=0.061,phase=1.1*ou+4.0,scale=0.05)" << endl;

    auto gradIn = fheonTrainBackwardPassController.backward_pass_fully_connected(gradOut, vecSize, inputSize, outputSize, LEARNING_RATE, layerName);
    for (int inCh = 0; inCh < inputSize; inCh++) {
        print_row("fc gradInput channel " + to_string(inCh), gradIn[inCh]);
    }
}

/*** ========================== Convolution ============================ ***
 * Forward (circular/wraparound, no zero-padding): out[oc][idx] =
 *   sum_in sum_k w[oc][in][k]*input[in][idx+offset_k], offset_k in
 *   {-(W+1),-W,-(W-1),-1,0,1,(W-1),W,(W+1)}.
 * Backward: dW[oc][in][k]=sum_idx(dOut[oc][idx]*input[in][idx+offset_k]);
 *   dBias[oc]=sum_idx(dOut[oc][idx]); dInput via full correlation with dOut.
 */
void validate_convolution() {
    cout << "\n===== VALIDATE: Convolution backward =====" << endl;
    int width = CONV_WIDTH;
    int kernelSize = KERNEL_WIDTH * KERNEL_WIDTH;
    int vecSize = BATCH_SIZE * width * width;
    string layerName = "validate_conv";

    for (int oc = 0; oc < TEST_CH_OUT; oc++) {
        for (int in = 0; in < TEST_CH_IN; in++) {
            for (int k = 0; k < kernelSize; k++) {
                string wFile = layerName + "_weights_" + to_string(oc) + "_" + to_string(in) + "_" + to_string(k);
                double val = 0.05 * sin(0.37 * (oc * 100 + in * 10 + k) + 1.0);
                auto wCt = encrypt_constant(vecSize, val);
                fheonTrainForwardPassController.serialize_and_store(wCt, wFile);
            }
        }
        string bFile = layerName + "_bias_" + to_string(oc);
        auto bCt = encrypt_constant(vecSize, 0.01 * oc);
        fheonTrainForwardPassController.serialize_and_store(bCt, bFile);
    }

    vector<Ctext> input(TEST_CH_IN);
    for (int in = 0; in < TEST_CH_IN; in++) {
        input[in] = encrypt_synth(vecSize, 0.013, in * 1.5, 1.0);
    }
    fheonTrainForwardPassController.he_forward_convolution_optimized(input, BATCH_SIZE, width, TEST_CH_IN, TEST_CH_OUT, 1, layerName);

    vector<Ctext> gradOut(TEST_CH_OUT);
    for (int oc = 0; oc < TEST_CH_OUT; oc++) {
        gradOut[oc] = encrypt_synth(vecSize, 0.043, oc * 1.2 + 5.0, 0.05);
    }

    cout << "  Params: width=" << width << ", batch=" << BATCH_SIZE << ", inCh=" << TEST_CH_IN << ", outCh=" << TEST_CH_OUT
         << ", W[oc][in][k]=0.05*sin(0.37*(100*oc+10*in+k)+1.0), bias[oc]=0.01*oc,"
         << " input=synth(freq=0.013,phase=1.5*in,scale=1.0), gradOut=synth(freq=0.043,phase=1.2*oc+5.0,scale=0.05)" << endl;

    auto gradIn = fheonTrainBackwardPassController.backward_pass_convolution(gradOut, BATCH_SIZE, width, TEST_CH_IN, TEST_CH_OUT, KERNEL_WIDTH, LEARNING_RATE, layerName);
    for (int in = 0; in < TEST_CH_IN; in++) {
        print_row("conv gradInput channel " + to_string(in), gradIn[in]);
    }
}

/*** ============================ AvgPool =============================== ***
 * Backward is self-contained (no forward cache dependency): every position
 * in a 2x2 window gets pooledGrad/4.
 */
void validate_avgpool() {
    cout << "\n===== VALIDATE: AvgPool backward =====" << endl;
    int srcWidth = AVGPOOL_SRC_WIDTH; // pooled/source resolution
    int vecSize = BATCH_SIZE * srcWidth * srcWidth;
    string layerName = "validate_avgpool";

    vector<Ctext> gradOut(TEST_CH_IN);
    for (int c = 0; c < TEST_CH_IN; c++) {
        gradOut[c] = encrypt_synth(vecSize, 0.029, c * 1.6, 0.1);
    }

    cout << "  Params: srcWidth=" << srcWidth << ", batch=" << BATCH_SIZE << ", channels=" << TEST_CH_IN
         << ", dilateFactor=" << POOL_WIDTH << ", gradOut=synth(freq=0.029,phase=1.6*c,scale=0.1)" << endl;

    auto gradIn = fheonTrainBackwardPassController.backward_pass_avgpool(gradOut, BATCH_SIZE, srcWidth, POOL_WIDTH);
    for (int c = 0; c < TEST_CH_IN; c++) {
        print_row("avgpool gradInput channel " + to_string(c) + " (width " + to_string(srcWidth * POOL_WIDTH) + ")", gradIn[c], 20);
    }
    (void)layerName;
}

/*** ============== AvgPool backward, optimized vs original =============== ***
 * backward_pass_avgpool_optimized is supposed to be mathematically identical
 * to backward_pass_avgpool, just built on the cheaper backward_dilate_optimized
 * (see that function's doc comment). Same synthetic inputs fed to both, both
 * decrypted, and the max abs difference reported directly -- a real pass/fail
 * check, not an eyeballed print (see max_abs_diff's doc comment).
 *
 * Requires backward_dilate_optimized_indices(BATCH_SIZE, AVGPOOL_SRC_WIDTH,
 * POOL_WIDTH)'s rotation keys to already be present -- main() adds them to
 * layer1.bin via add_and_persist_rotation_keys before calling this.
 */
void validate_avgpool_optimized() {
    cout << "\n===== VALIDATE: AvgPool backward (optimized vs original) =====" << endl;
    int srcWidth = AVGPOOL_SRC_WIDTH;
    int vecSize = BATCH_SIZE * srcWidth * srcWidth;

    vector<Ctext> gradOutOrig(TEST_CH_IN), gradOutOpt(TEST_CH_IN);
    for (int c = 0; c < TEST_CH_IN; c++) {
        auto v = synth_vec(vecSize, 0.029, c * 1.6, 0.1);
        gradOutOrig[c] = fheonHEController.encrypt_input(v, false);
        gradOutOpt[c] = fheonHEController.encrypt_input(v, false);
    }

    cout << "  Params: srcWidth=" << srcWidth << ", batch=" << BATCH_SIZE << ", channels=" << TEST_CH_IN
         << ", dilateFactor=" << POOL_WIDTH << ", gradOut=synth(freq=0.029,phase=1.6*c,scale=0.1)" << endl;

    auto start = startTime();
    auto gradInOrig = fheonTrainBackwardPassController.backward_pass_avgpool(gradOutOrig, BATCH_SIZE, srcWidth, POOL_WIDTH);
    printDuration(start, "  backward_pass_avgpool (original)", false);
    start = startTime();
    auto gradInOpt = fheonTrainBackwardPassController.backward_pass_avgpool_optimized(gradOutOpt, BATCH_SIZE, srcWidth, POOL_WIDTH);
    printDuration(start, "  backward_pass_avgpool_optimized (optimized)", false);

    int outWidth = srcWidth * POOL_WIDTH;
    int checkSlots = std::min(BATCH_SIZE * outWidth * outWidth, 512);
    for (int c = 0; c < TEST_CH_IN; c++) {
        cout << "  channel " << c << " (width " << outWidth << "):" << endl;
        cout << "    original : ";
        auto orig = fheonHEController.decrypt_and_print(gradInOrig[c], checkSlots);
        cout << "    optimized: ";
        auto opt = fheonHEController.decrypt_and_print(gradInOpt[c], checkSlots);
        double diff = max_abs_diff(orig, opt);
        cout << "    max abs diff (" << checkSlots << " slots): " << diff
             << (diff < 1e-6 ? "  [MATCH]" : "  [MISMATCH]") << endl;
    }
}

/*** ========== GlobalAvgPool backward, optimized vs original ============= ***
 * Same idea as validate_avgpool_optimized, for backward_pass_global_avgpool /
 * backward_pass_global_avgpool_optimized. Reuses AVGPOOL_SRC_WIDTH as the
 * dilateFactor (matches tuneResNet20.cpp's real globalpool_fc usage, where
 * dilateFactor == dataWidths[2] == 8) -- the shape where
 * backward_dilate_optimized's saving over backward_dilate is largest (8x
 * fewer per-row ops instead of 2x, see that function's doc comment).
 *
 * Requires backward_dilate_optimized_indices(BATCH_SIZE, 1, AVGPOOL_SRC_WIDTH)'s
 * rotation keys to already be present -- main() adds them to layer1.bin via
 * add_and_persist_rotation_keys before calling this.
 */
void validate_global_avgpool_optimized() {
    cout << "\n===== VALIDATE: GlobalAvgPool backward (optimized vs original) =====" << endl;
    int dataWidth = AVGPOOL_SRC_WIDTH;
    int vecSize = BATCH_SIZE; // compact: one value per image

    vector<Ctext> gradOutOrig(TEST_CH_IN), gradOutOpt(TEST_CH_IN);
    for (int c = 0; c < TEST_CH_IN; c++) {
        auto v = synth_vec(vecSize, 0.037, c * 2.1, 0.2);
        gradOutOrig[c] = fheonHEController.encrypt_input(v, false);
        gradOutOpt[c] = fheonHEController.encrypt_input(v, false);
    }

    cout << "  Params: dataWidth(=dilateFactor)=" << dataWidth << ", batch=" << BATCH_SIZE << ", channels=" << TEST_CH_IN
         << ", gradOut=synth(freq=0.037,phase=2.1*c,scale=0.2)" << endl;

    auto start = startTime();
    auto gradInOrig = fheonTrainBackwardPassController.backward_pass_global_avgpool(gradOutOrig, BATCH_SIZE, dataWidth);
    printDuration(start, "  backward_pass_global_avgpool (original)", false);
    start = startTime();
    auto gradInOpt = fheonTrainBackwardPassController.backward_pass_global_avgpool_optimized(gradOutOpt, BATCH_SIZE, dataWidth);
    printDuration(start, "  backward_pass_global_avgpool_optimized (optimized)", false);

    int outVecSize = BATCH_SIZE * dataWidth * dataWidth;
    for (int c = 0; c < TEST_CH_IN; c++) {
        cout << "  channel " << c << ":" << endl;
        cout << "    original : ";
        auto orig = fheonHEController.decrypt_and_print(gradInOrig[c], outVecSize);
        cout << "    optimized: ";
        auto opt = fheonHEController.decrypt_and_print(gradInOpt[c], outVecSize);
        double diff = max_abs_diff(orig, opt);
        cout << "    max abs diff (" << outVecSize << " slots): " << diff
             << (diff < 1e-6 ? "  [MATCH]" : "  [MISMATCH]") << endl;
    }
}

/*** ============== AvgPool backward, log-depth vs original =============== ***
 * Same idea as validate_avgpool_optimized, for backward_pass_avgpool_logdepth
 * -- see backward_dilate_logdepth's doc comment for the derivation and the
 * real tradeoff (fewer ops, more consumed levels). Prints each result's
 * ciphertext level alongside the values so that tradeoff is a measured
 * number here, not just an estimate.
 *
 * Requires backward_dilate_logdepth_indices(BATCH_SIZE, AVGPOOL_SRC_WIDTH,
 * POOL_WIDTH)'s rotation keys -- main() adds them to
 * modules_optimized_dilate.bin alongside the *_optimized indices.
 */
void validate_avgpool_logdepth() {
    cout << "\n===== VALIDATE: AvgPool backward (logdepth vs original) =====" << endl;
    int srcWidth = AVGPOOL_SRC_WIDTH;
    int vecSize = BATCH_SIZE * srcWidth * srcWidth;

    vector<Ctext> gradOutOrig(TEST_CH_IN), gradOutLd(TEST_CH_IN);
    for (int c = 0; c < TEST_CH_IN; c++) {
        auto v = synth_vec(vecSize, 0.029, c * 1.6, 0.1);
        gradOutOrig[c] = fheonHEController.encrypt_input(v, false);
        gradOutLd[c] = fheonHEController.encrypt_input(v, false);
    }

    cout << "  Params: srcWidth=" << srcWidth << ", batch=" << BATCH_SIZE << ", channels=" << TEST_CH_IN
         << ", dilateFactor=" << POOL_WIDTH << ", gradOut=synth(freq=0.029,phase=1.6*c,scale=0.1)" << endl;
    cout << "  Input level: " << gradOutOrig[0]->GetLevel() << endl;

    auto start = startTime();
    auto gradInOrig = fheonTrainBackwardPassController.backward_pass_avgpool(gradOutOrig, BATCH_SIZE, srcWidth, POOL_WIDTH);
    printDuration(start, "  backward_pass_avgpool (original)", false);
    start = startTime();
    auto gradInLd = fheonTrainBackwardPassController.backward_pass_avgpool_logdepth(gradOutLd, BATCH_SIZE, srcWidth, POOL_WIDTH);
    printDuration(start, "  backward_pass_avgpool_logdepth (logdepth)", false);

    int outWidth = srcWidth * POOL_WIDTH;
    int checkSlots = std::min(BATCH_SIZE * outWidth * outWidth, 512);
    for (int c = 0; c < TEST_CH_IN; c++) {
        cout << "  channel " << c << " (width " << outWidth << "):" << endl;
        cout << "    original level: " << gradInOrig[c]->GetLevel() << ", logdepth level: " << gradInLd[c]->GetLevel() << endl;
        cout << "    original : ";
        auto orig = fheonHEController.decrypt_and_print(gradInOrig[c], checkSlots);
        cout << "    logdepth : ";
        auto ld = fheonHEController.decrypt_and_print(gradInLd[c], checkSlots);
        double diff = max_abs_diff(orig, ld);
        cout << "    max abs diff (" << checkSlots << " slots): " << diff
             << (diff < 1e-6 ? "  [MATCH]" : "  [MISMATCH]") << endl;
    }
}

/*** ========== GlobalAvgPool backward, log-depth vs original ============= ***
 * Same idea as validate_global_avgpool_optimized, for
 * backward_pass_global_avgpool_logdepth. inputWidth=1 here means
 * column_scatter_log_depth takes its L=0 fast path (see that function's doc
 * comment) -- expect only a 1-level difference vs the *_optimized variant
 * here, not the bigger gap validate_avgpool_logdepth should show.
 */
void validate_global_avgpool_logdepth() {
    cout << "\n===== VALIDATE: GlobalAvgPool backward (logdepth vs original) =====" << endl;
    int dataWidth = AVGPOOL_SRC_WIDTH;
    int vecSize = BATCH_SIZE;

    vector<Ctext> gradOutOrig(TEST_CH_IN), gradOutLd(TEST_CH_IN);
    for (int c = 0; c < TEST_CH_IN; c++) {
        auto v = synth_vec(vecSize, 0.037, c * 2.1, 0.2);
        gradOutOrig[c] = fheonHEController.encrypt_input(v, false);
        gradOutLd[c] = fheonHEController.encrypt_input(v, false);
    }

    cout << "  Params: dataWidth(=dilateFactor)=" << dataWidth << ", batch=" << BATCH_SIZE << ", channels=" << TEST_CH_IN
         << ", gradOut=synth(freq=0.037,phase=2.1*c,scale=0.2)" << endl;

    auto gradInOrig = fheonTrainBackwardPassController.backward_pass_global_avgpool(gradOutOrig, BATCH_SIZE, dataWidth);
    auto gradInLd = fheonTrainBackwardPassController.backward_pass_global_avgpool_logdepth(gradOutLd, BATCH_SIZE, dataWidth);

    int outVecSize = BATCH_SIZE * dataWidth * dataWidth;
    for (int c = 0; c < TEST_CH_IN; c++) {
        cout << "  channel " << c << ":" << endl;
        cout << "    original level: " << gradInOrig[c]->GetLevel() << ", logdepth level: " << gradInLd[c]->GetLevel() << endl;
        cout << "    original : ";
        auto orig = fheonHEController.decrypt_and_print(gradInOrig[c], outVecSize);
        cout << "    logdepth : ";
        auto ld = fheonHEController.decrypt_and_print(gradInLd[c], outVecSize);
        double diff = max_abs_diff(orig, ld);
        cout << "    max abs diff (" << outVecSize << " slots): " << diff
             << (diff < 1e-6 ? "  [MATCH]" : "  [MISMATCH]") << endl;
    }
}

/*** ==================== Cross-Entropy + Softmax ======================= ***
 * Backward is exact (no approximation): dInput[c] = (pred[c]-label_onehot[c])/batch.
 */
void validate_cross_entropy_softmax() {
    cout << "\n===== VALIDATE: CrossEntropy+Softmax backward =====" << endl;
    int numClasses = 3;
    int batch = BATCH_SIZE;
    string layerName = "validate_ce";

    // "predictions" here stand in for softmax's output directly (this backward function takes
    // predictions, not logits) — values in [0,1], not necessarily summing to 1 per slot since
    // each class is a distinct ciphertext; that's fine, the formula is exact per-slot arithmetic.
    vector<Ctext> preds(numClasses);
    vector<vector<double>> predVecs(numClasses, vector<double>(batch));
    for (int c = 0; c < numClasses; c++) {
        auto v = synth_vec(batch, 0.4, c * 0.8, 0.15);
        for (auto& x : v) x = 0.33 + x; // center around 1/numClasses
        predVecs[c] = v;
        preds[c] = fheonHEController.encrypt_input(v, false);
    }

    vector<double> labels(batch);
    for (int i = 0; i < batch; i++) labels[i] = i % numClasses;

    cout << "  Params: numClasses=" << numClasses << ", batch=" << batch
         << ", pred[c]=0.33+synth(freq=0.4,phase=0.8*c,scale=0.15), labels[i]=i%numClasses" << endl;

    auto gradIn = fheonTrainBackwardPassController.backward_pass_cross_entropy_and_softmax(preds, labels, batch, batch, layerName);
    for (int c = 0; c < numClasses; c++) {
        print_row("cross-entropy gradInput class " + to_string(c), gradIn[c], batch);
    }
}

/*** ============================ Softmax (forward) ======================= ***
 * softmax[c][i] = exp(x[c][i]) / sum_c' exp(x[c'][i]), with exp(x)
 * approximated by a degree-2 Taylor polynomial (1 + x + x^2/2) and the
 * reciprocal of the summed exponentials approximated by a Chebyshev fit of
 * 1/x over [-1,1] — see he_forward_softmax's own doc comment for the
 * calibration-rescaling trick (scalingValues) that keeps the summed
 * exponentials inside that interval. scalingValues here is computed the
 * same way FHEONHEController::read_batch_scaling_values does in the real
 * pipeline (ceil of each channel's max-abs value) — just read directly off
 * the known plaintext instead of decrypting, since we already have it.
 */
void validate_softmax() {
    cout << "\n===== VALIDATE: Softmax forward =====" << endl;
    int numClasses = 3;
    int batch = BATCH_SIZE;

    vector<Ctext> logits(numClasses);
    vector<int> scalingValues(numClasses);
    for (int c = 0; c < numClasses; c++) {
        auto v = synth_vec(batch, 0.4, c * 0.8, 1.0);
        logits[c] = fheonHEController.encrypt_input(v, false);
        double maxAbs = 0.0;
        for (double x : v) {
            if (fabs(x) > maxAbs) maxAbs = fabs(x);
        }
        scalingValues[c] = (int)ceil(maxAbs);
    }

    cout << "  Params: numClasses=" << numClasses << ", batch=" << batch
         << ", logit[c]=synth(freq=0.4,phase=0.8*c,scale=1.0)" << endl;

    auto softMaxOut = fheonTrainForwardPassController.he_forward_softmax(logits, scalingValues, numClasses, 59);
    for (int c = 0; c < numClasses; c++) {
        print_row("softmax output class " + to_string(c), softMaxOut[c], batch);
    }
}

/*** ============ Cross-Entropy Loss (forward, chained with Softmax) ======= ***
 * loss = -(1/batch) * sum_i log(pred[true_class(i)][i]), pred = softmax
 * output straight out of he_forward_softmax above, and log(x) approximated
 * by a degree-taylorApproxDegree Taylor series around x=1 (see
 * taylors_approximation) — matches TrainModel.cpp's Forward 4.2/4.3 chain
 * exactly. NOTE: TrainModel.cpp's Forward 4.2 he_forward_softmax call is
 * currently commented out, so it feeds raw FC2 logits straight into
 * cross_entropy_loss's log-Taylor-around-1 approximation instead of softmax
 * probabilities — that series only converges for inputs in roughly (0,2),
 * so raw logits (which can be arbitrarily large/negative) will make both the
 * printed loss and the backward_pass_cross_entropy_and_softmax gradient
 * meaningless. This validator exercises the full, intended chain so that gap
 * is visible/testable in isolation.
 */
void validate_cross_entropy_loss() {
    cout << "\n===== VALIDATE: Softmax + CrossEntropy loss (forward) =====" << endl;
    int numClasses = 3;
    int batch = BATCH_SIZE;
    int taylorApproxDegree = 3; // matches TrainModel.cpp's taylorApproxDegree

    vector<Ctext> logits(numClasses);
    vector<int> scalingValues(numClasses);
    for (int c = 0; c < numClasses; c++) {
        auto v = synth_vec(batch, 0.4, c * 0.8, 1.0);
        logits[c] = fheonHEController.encrypt_input(v, false);
        double maxAbs = 0.0;
        for (double x : v) {
            if (fabs(x) > maxAbs) maxAbs = fabs(x);
        }
        scalingValues[c] = (int)ceil(maxAbs);
    }

    vector<double> labels(batch);
    for (int i = 0; i < batch; i++) labels[i] = i % numClasses;

    cout << "  Params: numClasses=" << numClasses << ", batch=" << batch
         << ", logit[c]=synth(freq=0.4,phase=0.8*c,scale=1.0), labels[i]=i%numClasses,"
         << " taylorApproxDegree=" << taylorApproxDegree << endl;

    auto softMaxOut = fheonTrainForwardPassController.he_forward_softmax(logits, scalingValues, numClasses, 59);
    auto loss = fheonTrainBackwardPassController.cross_entropy_loss(softMaxOut, labels, batch, numClasses, taylorApproxDegree);

    cout << "  Cross-entropy loss (decrypted, averaged scalar broadcast across all " << batch << " slots): ";
    fheonHEController.decrypt_and_print(loss, batch);
}

int main(int argc, char *argv[]) {

    auto begin_time = startTime();
    printWelcomeMessage();
    cout << "Initializing FHEON per-module backward-pass validation harness..." << endl;

    auto runtime_config = parse_runtime_args(argc, argv, DEFAULT_TEST_SIZE, INDEX_VALUE);

    FHEONHEController::HEConfig heinitConfig;
    heinitConfig.ringDim = 15;
    heinitConfig.numSlots = 14;
    heinitConfig.mlevelBootstrap = 12;
    heinitConfig.dcrtBits = 46;
    heinitConfig.firstMod = 50;
    heinitConfig.numDigits = 4;
    heinitConfig.levelBudget = {3, 3};
    heinitConfig.serialize = true;
    if (!runtime_config.keys_folder.empty()) {
        heinitConfig.keysFolder = runtime_config.keys_folder;
    }

    // Always load — this harness relies entirely on the crypto context/keypair
    // already generated by a real trainModel run (see file header). Run
    // trainModel at least once first if AllHEKeys/ is empty.
    fheonHEController.load_context(heinitConfig, true);
    printDuration(begin_time, "Context Loaded", false);

    context = fheonHEController.getContext();
    fheonTrainSupportController.setContext(context);
    fheonTrainForwardPassController.setContext(context);
    fheonTrainBackwardPassController.setContext(context);
    fheonTrainBackwardPassController.setHEController(&fheonHEController);

    vector<double> zeroVec(1 << heinitConfig.numSlots, 0.0);
    vector<double> onesVec(1 << heinitConfig.numSlots, 1.0);
    auto encrytedZero = fheonHEController.encrypt_input(zeroVec);
    auto encrytedOne = fheonHEController.encrypt_input(onesVec);
    fheonTrainSupportController.setEncryptedZeros(encrytedZero);
    fheonTrainSupportController.setEncryptedOnes(encrytedOne);
    fheonTrainForwardPassController.setEncryptedZeros(encrytedZero);
    fheonTrainForwardPassController.setEncryptedOnes(encrytedOne);
    fheonTrainBackwardPassController.setEncryptedZeros(encrytedZero);
    fheonTrainBackwardPassController.setEncryptedOnes(encrytedOne);

    cout << "\nRun python/training/validate_modules.py independently and diff its printed"
         << " values against the [VALIDATE] lines below (close, not bit-identical, agreement"
         << " is expected for ReLU due to the Chebyshev approximation)." << endl;

    // validate_cross_entropy_softmax();
    // validate_softmax();
    // validate_cross_entropy_loss();
    // validate_relu();
    // validate_batch_norm();
    // validate_batch_norm_sparse_grad();
    validate_fully_connected();
    // validate_avgpool();
    // validate_convolution();

    // This harness now only exercises the optimized-dilate validators below
    // (every original per-primitive validator above is disabled), which need
    // a much smaller rotation-key set than the old layer1.bin -- so this no
    // longer depends on that file (or a prior trainModel run having produced
    // it) at all. Generated fresh into a dedicated "modules_optimized_dilate.bin"
    // on a normal run; pass --load-context to reuse one already generated by
    // a prior run instead of regenerating it (same generate-vs-load split
    // tuneResNet20.cpp's own per-stage key files use, just made explicit here
    // since load_context() above always loads the shared secret key/context
    // regardless, so FHEONHEController's own context_loaded can't be used to
    // pick generate-vs-load automatically the way it does for a first-ever
    // run of that file).
    //
    // Indices needed, beyond backward_dilate_optimized_indices' own set (see
    // that function's doc comment for why those alone aren't enough):
    //   - backward_pass_avgpool_optimized's 4-term smear: -1, -dilatedWidth, -(dilatedWidth+1)
    //   - backward_pass_global_avgpool_optimized's repeat_encrypted_values doubling smear: -1, -2, -4, ..., -(spatialSize/2)
    //   - backward_dilate_logdepth_indices' own doubling rotations (see its doc
    //     comment) -- overlaps heavily with the above (harmless, EvalRotateKeyGen
    //     is idempotent) but adds the couple of indices (e.g. -2, -4 for
    //     AVGPOOL_SRC_WIDTH=8) the *_optimized-only set above doesn't cover.
    auto optDilateAvgpoolKeys = fheonTrainBackwardPassController.backward_dilate_optimized_indices(BATCH_SIZE, AVGPOOL_SRC_WIDTH, POOL_WIDTH);
    auto optDilateGlobalpoolKeys = fheonTrainBackwardPassController.backward_dilate_optimized_indices(BATCH_SIZE, 1, AVGPOOL_SRC_WIDTH);
    auto ldAvgpoolKeys = fheonTrainBackwardPassController.backward_dilate_logdepth_indices(BATCH_SIZE, AVGPOOL_SRC_WIDTH, POOL_WIDTH);
    auto ldGlobalpoolKeys = fheonTrainBackwardPassController.backward_dilate_logdepth_indices(BATCH_SIZE, 1, AVGPOOL_SRC_WIDTH);
    vector<int> modulesKeys = optDilateAvgpoolKeys;
    modulesKeys.insert(modulesKeys.end(), optDilateGlobalpoolKeys.begin(), optDilateGlobalpoolKeys.end());
    modulesKeys.insert(modulesKeys.end(), ldAvgpoolKeys.begin(), ldAvgpoolKeys.end());
    modulesKeys.insert(modulesKeys.end(), ldGlobalpoolKeys.begin(), ldGlobalpoolKeys.end());

    int dilatedWidth = POOL_WIDTH * AVGPOOL_SRC_WIDTH;
    modulesKeys.push_back(-1);
    modulesKeys.push_back(-dilatedWidth);
    modulesKeys.push_back(-(dilatedWidth + 1));

    int spatialSize = AVGPOOL_SRC_WIDTH * AVGPOOL_SRC_WIDTH;
    for (int p = 1; p < spatialSize; p *= 2) {
        modulesKeys.push_back(-p);
    }

    if (runtime_config.load_context) {
        fheonHEController.load_rotation_keys("modules_optimized_dilate.bin");
        printDuration(begin_time, "modules_optimized_dilate.bin Rotation Keys Loaded", false);
    } else {
        fheonHEController.add_and_persist_rotation_keys(modulesKeys, "modules_optimized_dilate.bin");
        printDuration(begin_time, "modules_optimized_dilate.bin Rotation Keys Generated", false);
    }

    validate_avgpool_optimized();
    validate_global_avgpool_optimized();
    validate_avgpool_logdepth();
    validate_global_avgpool_logdepth();

    printDuration(begin_time, "Total Validation Time", false);
    return 0;
}
