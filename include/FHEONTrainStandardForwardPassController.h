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

#ifndef FHEON_TRAINSTANDARDFORWARDPASSCONTROLLER_H
#define FHEON_TRAINSTANDARDFORWARDPASSCONTROLLER_H

#include <functional>
#include "FHEONTrainForwardPassController.h"

/**
 * @brief Multithreaded forward pass of the training pipeline, standard layout (one channel of the batch per ciphertext).
 *
 * Same primitives, arguments, cache files and results as FHEONTrainForwardPassController (it derives from it and keeps
 * its helpers), with:
 *   - ciphertext weights multiplied without relinearization and relinearized once per output (the original
 *     relinearizes every product: inputChannels * 9 key switchings per output channel of a convolution, now 1);
 *   - every weight read from disk once (the original reads every convolution weight twice and writes the linear and
 *     shortcut weights back after reading them), streamed per output instead of held for the whole layer;
 *   - the global average pooling sums the images of a channel with one rotate-and-add (log2(w^2) rotations) instead of
 *     one EvalSum per image;
 *   - the batch norm keeps EvalBootstrap on the calling thread and runs the Newton iterations of all channels in lock
 *     step, with the first iteration in one level (y0 is a known constant);
 *   - the number of workers capped with setWorkerThreads, and the bootstrapping iterations set with
 *     setBootstrapIterations (2, as the original, by default).
 * The rotations are those of the original, so the existing training key files apply unchanged.
 */
class FHEONTrainStandardForwardPassController : public FHEONTrainForwardPassController {

protected:
    /** workers of the primitives: 0 = all hardware threads (or --threads), n = at most n, 1 = sequential */
    int worker_thread_limit = 0;
    /** iterations of every EvalBootstrap (2 = the precision of the original controller) */
    int bootstrap_iterations = 2;
    /** precision (bits) of the first of 2 iterations; 0 (as the original) makes the second iteration useless on deep
     *  ciphertexts (~8e-3 error at ring 2^15 instead of ~3e-7 with 15), see FHEONHEController::bootstrap_precision */
    int bootstrap_precision = 0;
    /** bootstrapping of a group of ciphertexts (e.g. FHEONHEController::batch_bootstrap_function); default: EvalBootstrap
     *  one after the other on the calling thread */
    std::function<vector<Ctext>(vector<Ctext>&)> bootstrap_function;

public:
    FHEONTrainStandardForwardPassController(CryptoContext<DCRTPoly> ctx) : FHEONTrainForwardPassController(ctx) {}

    void setWorkerThreads(int numThreads) { worker_thread_limit = numThreads; }
    int getWorkerThreads() const { return worker_thread_limit; }
    void setBootstrapIterations(int iterations) { bootstrap_iterations = std::max(1, iterations); }
    void setBootstrapPrecision(int bits) { bootstrap_precision = bits; }
    void setBootstrapFunction(const std::function<vector<Ctext>(vector<Ctext>&)>& function) { bootstrap_function = function; }

    vector<Ctext> he_forward_convolution_optimized(vector<Ctext> &encryptedInputs, int batchSize, int inputWidth, int inputChannels, int outputChannels, int stride, string layerName="", bool pretrainedFromCsv = false);
    vector<Ctext> he_forward_convolution_shortcut_optimized(vector<Ctext> &encryptedInputs, int batchSize, int inputWidth, int inputChannels, int outputChannels, int stride, string layerName="", bool pretrainedFromCsv = false);
    vector<Ctext> he_forward_global_avgpool(vector<Ctext>& encryptedInputs, int batchSize, int dataWidth, int inputChannels, string layerName="");
    vector<Ctext> he_forward_relu(vector<Ctext>& encryptedInputs, vector<int> scaleValues, int inputChannels, int vectorSize, int polyDegree=59, string layerName="");
    vector<Ctext> he_forward_avgPool_optimized(vector<Ctext>& encryptedInputs, int batchSize, int inputWidth, int kernelWidth, int stride=2, string layerName="");
    vector<Ctext> he_forward_linear_multiple_outputs(vector<Ctext>& encryptedInput, int vectorSize, int inputSize, int outputSize, string layerName="", bool pretrainedFromCsv = false);
    vector<Ctext> he_forward_softmax(vector<Ctext>& encryptedInputs, const vector<int>& scalingValues, int inputSize, int appxPolyDegree=59);
    vector<Ctext> he_forward_batch_norm(vector<Ctext> &encryptedInputs, int inputSize, double appxInitValue, int appxDegree = 3, string layerName = "", bool isBootstrap = true, bool pretrainedFromCsv = false);
    /**
     * Batch norm with a per-channel start of Newton's 1/sqrt(var + epsilon): channelInverseStdInit[c] (e.g.
     * 1 / sqrt(running_var[c] + eps) of the pretrained model). A constant start (appxInitValue) only converges near
     * var = 1 / start^2, and the batch variances of a trained ResNet20 span 1e-13 ... 2: 3 iterations leave 1/std up to
     * 1000x too small, which shrinks the activations layer after layer. Newton runs on the normalized variance
     * (var + eps) start^2 ~ 1 from 1 and the start enters only as scalar products, so the bootstrapped values stay small
     * (1/std reaches 1/sqrt(eps) ~ 316 on dead channels). Same caches and parameter files as the overload above.
     */
    vector<Ctext> he_forward_batch_norm(vector<Ctext> &encryptedInputs, int inputSize, const vector<double>& channelInverseStdInit, int appxDegree,
                                        string layerName, bool isBootstrap, bool pretrainedFromCsv, double epsilon = 1e-5);
    /**
     * Softmax over the class ciphertexts (one per class, image i in slot i < batchSize) by repeated squaring, for logits
     * within [-logitBound, logitBound]: q = exp(z / 2^r) (4th-order Taylor), then r times q <- q^2 / sum(q^2). Values stay
     * within [0, 1] and nothing is decrypted. he_forward_softmax's 1 + x + x^2 / 2 is not monotonic for x < -1 and picks
     * wrong classes on real logits (-8 ... 15 in ResNet20). One group bootstrapping of the class ciphertexts per squaring.
     */
    vector<Ctext> he_forward_softmax_normalized(vector<Ctext>& logits, int batchSize, double logitBound = 32.0, int finalNewtonIterations = 1);

protected:
    /** Newton iterations for 1/sqrt(x) of every ciphertext, in lock step (bootstrapping y every 2 iterations) */
    vector<Ctext> newton_inverse_sqrt_all(vector<Ctext>& encryptedInputs, int inputSize, double initValue, int appxDegree, bool isBootstrap);
    /** bootstrap every ciphertext (bootstrap_function when set, else EvalBootstrap with the iterations and precision) */
    void bootstrap_all(vector<Ctext>& ciphertexts);
    /** class probabilities q / sum(q): Chebyshev 1/x (degree) on [low, high], then newtonIterations Newton steps */
    vector<Ctext> normalize_classes(vector<Ctext>& values, int batchSize, double low, double high, int newtonIterations, int degree);
    /** cache the inputs of a layer under layerName + infix + channel */
    void store_inputs(vector<Ctext>& encryptedInputs, int count, const string& prefix);
};

#endif // FHEON_TRAINSTANDARDFORWARDPASSCONTROLLER_H
