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

#ifndef FHEON_TRAINSTANDARDBACKWARDPASSCONTROLLER_H
#define FHEON_TRAINSTANDARDBACKWARDPASSCONTROLLER_H

#include <functional>
#include "FHEONTrainBackwardPassController.h"

/**
 * @brief Multithreaded backward pass of the training pipeline, standard layout (one channel of the batch per ciphertext).
 *
 * Same primitives, arguments, cache files, parameter files and results as FHEONTrainBackwardPassController (it derives
 * from it; the dilations, pooling variants, losses and helpers are inherited), with:
 *   - convolution, shortcut and fully connected layers fused per input channel: every weight is read once, used for the
 *     input gradient (old weight, as the original) and updated and written in the same worker. The original reads
 *     every convolution weight twice and holds a whole outputs x inputs matrix of weight gradients (fully connected);
 *   - the input-gradient products accumulated without relinearization (one relinearization per input channel instead
 *     of outputChannels * 9);
 *   - every EvalBootstrap on the calling thread, with setBootstrapIterations (2 by default) and
 *     setBootstrapLevelThreshold (batch norm refresh above level 10 by default, as the original);
 *   - the number of workers capped with setWorkerThreads.
 * The rotations are those of the original, so the existing training key files apply unchanged.
 */
class FHEONTrainStandardBackwardPassController : public FHEONTrainBackwardPassController {

protected:
    /** workers of the primitives: 0 = all hardware threads (or --threads), n = at most n, 1 = sequential */
    int worker_thread_limit = 0;
    /** iterations of every EvalBootstrap (2 = the precision of the original controller) */
    int bootstrap_iterations = 2;
    /** batch norm: the cached xhat and gamma / std are refreshed above this level */
    int bootstrap_level_threshold = 10;
    /** precision (bits) of the first of 2 iterations (0 as the original, see FHEONHEController::bootstrap_precision) */
    int bootstrap_precision = 0;
    /** bootstrapping of a group of ciphertexts; default: EvalBootstrap one after the other on the calling thread */
    std::function<vector<Ctext>(vector<Ctext>&)> bootstrap_function;

public:
    FHEONTrainStandardBackwardPassController(CryptoContext<DCRTPoly>& ctx) : FHEONTrainBackwardPassController(ctx) {}

    void setWorkerThreads(int numThreads) { worker_thread_limit = numThreads; }
    int getWorkerThreads() const { return worker_thread_limit; }
    void setBootstrapIterations(int iterations) { bootstrap_iterations = std::max(1, iterations); }
    void setBootstrapLevelThreshold(int level) { bootstrap_level_threshold = level; }
    void setBootstrapPrecision(int bits) { bootstrap_precision = bits; }
    void setBootstrapFunction(const std::function<vector<Ctext>(vector<Ctext>&)>& function) { bootstrap_function = function; }

    vector<Ctext> backward_pass_cross_entropy_and_softmax(vector<Ctext> &encryptedInputs, vector<double> &labels, int batchSize, int vectorSize, string layerName="");
    vector<Ctext> backward_pass_relu(vector<Ctext> &gradInputs, vector<int>& scalers, int appxPolyDegree, string layerName = "");
    vector<Ctext> backward_pass_batch_norm(vector<Ctext> &encryptedInputs, vector<Ctext> &gradInputs, int inputSize, double learningRate, string layerName = "");
    vector<Ctext> backward_pass_fully_connected(vector<Ctext> &gradInputs, int vectorSize, int inputSize, int outputSize, double learningRate, string layerName);
    vector<Ctext> backward_pass_convolution(vector<Ctext> &gradInputs, int batchSize, int inputWidth,
                    int inputChannels, int outputChannels, int kernelWidth, double learningRate, string layerName, int stride = 1);
    vector<Ctext> backward_pass_convolution_shortcut(vector<Ctext> &gradInputs, int batchSize, int inputWidth,
                    int inputChannels, int outputChannels, int stride, double learningRate, string layerName);
    vector<Ctext> backward_pass_global_avgpool(vector<Ctext> &gradInputs, int batchSize, int dataWidth);
    vector<Ctext> backward_pass_global_avgpool_logdepth(vector<Ctext> &gradInputs, int batchSize, int dataWidth);
    vector<Ctext> backward_pass_avgpool_logdepth(vector<Ctext> &encryptedInputs, int batchSize, int inputWidth, int dilateFactor);

protected:
    /** bootstrap every ciphertext (bootstrap_function when set, else EvalBootstrap with the iterations and precision) */
    void bootstrap_all(vector<Ctext>& ciphertexts);
    /** stride-2 gradients upsampled to the input resolution (backward_dilate_logdepth), then refreshed on this thread */
    vector<Ctext> dilated_gradients(vector<Ctext>& gradInputs, int batchSize, int outputWidth, int stride);
};

#endif // FHEON_TRAINSTANDARDBACKWARDPASSCONTROLLER_H
