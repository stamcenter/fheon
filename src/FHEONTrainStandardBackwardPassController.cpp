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

#include "FHEONTrainStandardBackwardPassController.h"

/** add a product to an accumulator (the first product becomes the accumulator) */
static inline void accumulate_term(CryptoContext<DCRTPoly>& context, Ctext& accumulator, const Ctext& product) {
    if (accumulator) context->EvalAddInPlace(accumulator, product);
    else accumulator = product;
}

/** relinearize a sum of products taken without relinearization */
static inline Ctext relinearize_sum(CryptoContext<DCRTPoly>& context, const Ctext& sum) {
    return (sum->NumberCiphertextElements() > 2) ? context->Relinearize(sum) : sum;
}

/**
 * @brief (predictions - labels) / batchSize on the first batchSize slots, see
 *        FHEONTrainBackwardPassController::backward_pass_cross_entropy_and_softmax (labels and mask encoded once per class).
 */
vector<Ctext> FHEONTrainStandardBackwardPassController::backward_pass_cross_entropy_and_softmax(vector<Ctext> &encryptedInputs, vector<double> &labels,
                                                                                            int batchSize, int vectorSize, string layerName) {
    int numInputs = (int)encryptedInputs.size();
    double scale = (1.0 / batchSize);
    vector<double> cleanMaskVec(vectorSize, 0.0);
    for (int j = 0; j < batchSize && j < vectorSize; j++) {
        cleanMaskVec[j] = scale;
    }
    vector<Ctext> initGrad(numInputs);
    auto start_time = startTime();
    run_parallel_for(numInputs, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            vector<double> labelVec(batchSize, 0.0);
            for (int j = 0; j < batchSize; j++) {
                if ((int)labels[j] == i) labelVec[j] = 1.0;
            }
            int level = encryptedInputs[i]->GetLevel();
            Ptext labelsPxt = context->MakeCKKSPackedPlaintext(labelVec, 1, level, nullptr, batchSize);
            Ptext cleanMaskPtxt = context->MakeCKKSPackedPlaintext(cleanMaskVec, 1, level, nullptr, vectorSize);
            initGrad[i] = context->EvalMult(context->EvalSub(encryptedInputs[i], labelsPxt), cleanMaskPtxt);
            string gradFileName = "grad_" + layerName + "_channel_" + to_string(i);
            serialize_and_store(initGrad[i], gradFileName);
        }
    }, worker_thread_limit);
    printTiming("Cross entropy and softmax backward evaluating...", "Softmax", start_time);
    return initGrad;
}

/**
 * @brief dInput = dOutput * step(x), step approximated with Chebyshev on the cached input, see
 *        FHEONTrainBackwardPassController::backward_pass_relu.
 */
vector<Ctext> FHEONTrainStandardBackwardPassController::backward_pass_relu(vector<Ctext> &gradInputs, vector<int> &scalers, int appxPolyDegree, string layerName) {
    double lowerBound = -1;
    double upperBound = 1;
    int numChannels = (int)gradInputs.size();
    vector<Ctext> reluGradients(numChannels);
    auto start_time = startTime();
    run_parallel_for(numChannels, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            string inputFileName = layerName + "_channel_" + to_string(i);
            Ctext inputCtxt = deserialize_and_load(inputFileName);
            if (scalers[i] > 1) {
                inputCtxt = context->EvalMult(inputCtxt, 1.0 / (2 * scalers[i]));
            }
            Ctext reluDerivative = context->EvalChebyshevFunction([](double x) -> double { return (x < 0) ? 0.0 : 1.0; },
                                                                  inputCtxt, lowerBound, upperBound, appxPolyDegree);
            reluGradients[i] = context->EvalMult(gradInputs[i], reluDerivative);
            string gradFileName = "grad_" + layerName + "_channel_" + to_string(i);
            serialize_and_store(reluGradients[i], gradFileName);
        }
    }, worker_thread_limit);
    printTiming("ReLU backward evaluating...", "relu", start_time);
    return reluGradients;
}

/**
 * @brief Batch norm backward, see FHEONTrainBackwardPassController::backward_pass_batch_norm. The refresh bootstrappings
 *        run on this thread, the gradients of all channels in parallel, each channel updating its gamma and beta in the
 *        same worker.
 */
vector<Ctext> FHEONTrainStandardBackwardPassController::backward_pass_batch_norm(vector<Ctext> &encryptedInputs, vector<Ctext> &gradInputs, int inputSize,
                                                                              double learningRate, string layerName) {
    (void)encryptedInputs;
    double scaler = (-1.0 / inputSize);
    auto cleaningMaskVec = generate_scale_and_clean_mask(scaler, inputSize);
    int numChannels = (int)gradInputs.size();
    vector<Ctext> batchNormGradients(numChannels);

    vector<Ctext> xHatAll(numChannels);
    vector<Ctext> learnedandinvstdAll(numChannels);
    run_parallel_for(numChannels, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            string xHatFileName = "batch_norm_xhat_" + layerName + "_channel_" + to_string(i);
            xHatAll[i] = deserialize_and_load(xHatFileName);
            string invstdFileName = "batch_norm_invstd" + layerName + "_channel_" + to_string(i);
            learnedandinvstdAll[i] = deserialize_and_load(invstdFileName);
        }
    }, worker_thread_limit);
    {
        vector<int> deep;
        vector<Ctext> group;
        for (int i = 0; i < numChannels; i++) {
            if ((int)xHatAll[i]->GetLevel() > bootstrap_level_threshold) { deep.push_back(i); group.push_back(xHatAll[i]); }
            if ((int)learnedandinvstdAll[i]->GetLevel() > bootstrap_level_threshold) { deep.push_back(numChannels + i); group.push_back(learnedandinvstdAll[i]); }
        }
        bootstrap_all(group);
        for (size_t k = 0; k < deep.size(); k++) {
            if (deep[k] < numChannels) xHatAll[deep[k]] = group[k];
            else learnedandinvstdAll[deep[k] - numChannels] = group[k];
        }
    }

    auto gradients_start_time = startTime();
    run_parallel_for(numChannels, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            const Ctext& xHat = xHatAll[i];
            Ctext sumedLoss = context->EvalSum(gradInputs[i], inputSize);
            Ctext meanVal = context->EvalMult(sumedLoss, context->MakeCKKSPackedPlaintext(cleaningMaskVec, 1, sumedLoss->GetLevel(), nullptr, inputSize));
            meanVal = repeat_encrypted_values(meanVal, 1, inputSize);

            Ctext sumedInnGamma = context->EvalSum(context->EvalMult(gradInputs[i], xHat), inputSize);
            Ctext meanInnGamma = context->EvalMult(sumedInnGamma, context->MakeCKKSPackedPlaintext(cleaningMaskVec, 1, sumedInnGamma->GetLevel(), nullptr, inputSize));
            meanInnGamma = repeat_encrypted_values(meanInnGamma, 1, inputSize);

            Ctext innTerm = context->EvalAdd(gradInputs[i], context->EvalAdd(context->EvalMult(xHat, meanInnGamma), meanVal));
            batchNormGradients[i] = context->EvalMult(innTerm, learnedandinvstdAll[i]);

            string fileName = "batch_norm_" + layerName + "_channel_" + to_string(i);
            Ctext gamma = deserialize_and_load(fileName);
            gamma = optimizer_update(gamma, sumedInnGamma, learningRate);
            serialize_and_store(gamma, fileName);
            string biasFileName = "batch_norm_bias_" + layerName + "_channel_" + to_string(i);
            Ctext beta = deserialize_and_load(biasFileName);
            beta = optimizer_update(beta, sumedLoss, learningRate);
            serialize_and_store(beta, biasFileName);
            string backwardGradFileName = "batch_norm_grad_" + layerName + "_channel_" + to_string(i);
            serialize_and_store(batchNormGradients[i], backwardGradFileName);
        }
    }, worker_thread_limit);
    measureTiming("batch norm gradients...", "decrypt_batch_norm", gradients_start_time);
    return batchNormGradients;
}

/**
 * @brief Fully connected backward, see FHEONTrainBackwardPassController::backward_pass_fully_connected.
 *
 * One worker per input neuron reads each of its weights once: the old weight goes into dInput (accumulated without
 * relinearization), then the weight is updated with sum(dOutput * input) and written. The biases are updated per
 * output neuron afterwards. No outputs x inputs matrix of gradients is held.
 */
vector<Ctext> FHEONTrainStandardBackwardPassController::backward_pass_fully_connected(vector<Ctext> &gradInputs, int vectorSize, int inputSize, int outputSize,
                                double learningRate, string layerName) {
    double fan_in = inputSize;
    vector<Ctext> gradients(inputSize);
    auto start_time = startTime();
    run_parallel_for(inputSize, [&](int start, int end) {
        for (int inCh = start; inCh < end; inCh++) {
            string inputFileName = layerName + "_in_neuron_" + to_string(inCh);
            Ctext inputCtxt = deserialize_and_load(inputFileName);
            Ctext inputGradient;
            for (int ou = 0; ou < outputSize; ou++) {
                string weightsFileName = layerName + "_out_neuron_" + to_string(ou) + "_in_" + to_string(inCh);
                Ctext weightCtxt = deserialize_and_load(weightsFileName, fan_in, vectorSize);
                accumulate_term(context, inputGradient, context->EvalMultNoRelin(gradInputs[ou], weightCtxt));
                Ctext sumedWeightGrad = context->EvalSum(context->EvalMult(gradInputs[ou], inputCtxt), vectorSize);
                weightCtxt = optimizer_update(weightCtxt, sumedWeightGrad, learningRate);
                serialize_and_store(weightCtxt, weightsFileName);
            }
            gradients[inCh] = relinearize_sum(context, inputGradient);
        }
    }, worker_thread_limit);
    run_parallel_for(outputSize, [&](int start, int end) {
        for (int ou = start; ou < end; ou++) {
            Ctext biasGradSum = context->EvalSum(gradInputs[ou], vectorSize);
            string biasFileName = layerName + "_out_neuron_" + to_string(ou) + "_bias";
            Ctext inputbiasCtxt = deserialize_and_load(biasFileName);
            inputbiasCtxt = optimizer_update(inputbiasCtxt, biasGradSum, learningRate);
            serialize_and_store(inputbiasCtxt, biasFileName);
        }
    }, worker_thread_limit);
    printTiming("Fully Connected backward evaluating...", "fc", start_time);
    return gradients;
}

vector<Ctext> FHEONTrainStandardBackwardPassController::dilated_gradients(vector<Ctext>& gradInputs, int batchSize, int outputWidth, int stride) {
    vector<Ctext> dilated(gradInputs.size());
    run_parallel_for(gradInputs.size(), [&](int start, int end) {
        for (int oc = start; oc < end; oc++) {
            dilated[oc] = backward_dilate_logdepth(gradInputs[oc], batchSize, outputWidth, stride);
        }
    }, worker_thread_limit);
    bootstrap_all(dilated);
    return dilated;
}

/**
 * @brief 3x3 convolution backward, see FHEONTrainBackwardPassController::backward_pass_convolution.
 *
 * The 9 rotations of every (dilated) output gradient are computed in parallel, as in the original. Then one worker
 * per input channel builds the 9 taps of its cached input and, for every output channel and tap k, reads weight k once:
 *   dInput[in] += rotatedGradOut[oc][8 - k] * W[oc][in][k]          (old weight, no relinearization)
 *   W[oc][in][k] -= lr * EvalSum(gradOut[oc] * tap[in][k])          (written right away)
 * which is the original's arithmetic with each weight read once instead of twice.
 */
vector<Ctext> FHEONTrainStandardBackwardPassController::backward_pass_convolution(vector<Ctext> &gradInputs, int batchSize, int inputWidth,
                    int inputChannels, int outputChannels, int kernelWidth, double learningRate, string layerName, int stride) {
    int kernelSize = kernelWidth * kernelWidth;
    int totalInputSize = inputWidth * inputWidth * batchSize;
    int outputWidth = inputWidth / stride;
    int totalOutputSize = outputWidth * outputWidth * batchSize;
    double fan_in = inputChannels * kernelSize;

    vector<Ctext> biasGradAll(outputChannels);
    run_parallel_for(outputChannels, [&](int start, int end) {
        for (int oc = start; oc < end; oc++) {
            biasGradAll[oc] = context->EvalSum(gradInputs[oc], totalOutputSize);
        }
    }, worker_thread_limit);

    vector<Ctext> gradOut = (stride > 1) ? dilated_gradients(gradInputs, batchSize, outputWidth, stride) : gradInputs;

    auto start_time = startTime();
    auto nine_taps = [&](const Ctext& source, vector<Ctext>& taps) {
        auto cyclotomicOrder = context->GetCyclotomicOrder();
        Ctext replicated = replicate_across_ring(context, source, totalInputSize, num_slots);
        auto digits = context->EvalFastRotationPrecompute(replicated);
        Ctext first_shot = context->EvalRotate(replicated, -1);
        Ctext second_shot = context->EvalFastRotation(replicated, 1, cyclotomicOrder, digits);
        taps[0] = context->EvalRotate(first_shot, -inputWidth);
        taps[1] = context->EvalRotate(replicated, -inputWidth);
        taps[2] = context->EvalRotate(second_shot, -inputWidth);
        taps[3] = first_shot;
        taps[4] = replicated;
        taps[5] = second_shot;
        taps[6] = context->EvalRotate(first_shot, inputWidth);
        taps[7] = context->EvalFastRotation(replicated, inputWidth, cyclotomicOrder, digits);
        taps[8] = context->EvalRotate(second_shot, inputWidth);
    };

    vector<vector<Ctext>> rotatedGradOut(outputChannels, vector<Ctext>(kernelSize));
    run_parallel_for(outputChannels, [&](int start, int end) {
        for (int oc = start; oc < end; oc++) {
            nine_taps(gradOut[oc], rotatedGradOut[oc]);
        }
    }, worker_thread_limit);

    vector<Ctext> gradiantsCtxts(inputChannels);
    run_parallel_for(inputChannels, [&](int start, int end) {
        vector<Ctext> rotatedInput(kernelSize);
        for (int in = start; in < end; in++) {
            string inputFileName = layerName + "_channel_" + to_string(in);
            nine_taps(deserialize_and_load(inputFileName), rotatedInput);
            Ctext inputGradient;
            for (int oc = 0; oc < outputChannels; oc++) {
                for (int k = 0; k < kernelSize; k++) {
                    string weightsFileName = layerName + "_weights_" + to_string(oc) + "_" + to_string(in) + "_" + to_string(k);
                    Ctext weightCtxt = deserialize_and_load(weightsFileName, fan_in, totalInputSize);
                    accumulate_term(context, inputGradient, context->EvalMultNoRelin(rotatedGradOut[oc][kernelSize - 1 - k], weightCtxt));
                    Ctext sumedWeightGrad = context->EvalSum(context->EvalMult(gradOut[oc], rotatedInput[k]), totalInputSize);
                    weightCtxt = optimizer_update(weightCtxt, sumedWeightGrad, learningRate);
                    serialize_and_store(weightCtxt, weightsFileName);
                }
            }
            gradiantsCtxts[in] = relinearize_sum(context, inputGradient);
        }
    }, worker_thread_limit);

    run_parallel_for(outputChannels, [&](int start, int end) {
        for (int oc = start; oc < end; oc++) {
            string biasFileName = layerName + "_bias_" + to_string(oc);
            Ctext biasCtxt = deserialize_and_load(biasFileName, 1.0, num_slots);
            biasCtxt = optimizer_update(biasCtxt, biasGradAll[oc], learningRate);
            serialize_and_store(biasCtxt, biasFileName);
        }
    }, worker_thread_limit);
    printTiming("Convolution backward evaluating...", "convolution", start_time);
    return gradiantsCtxts;
}

/**
 * @brief 1x1 shortcut backward, see FHEONTrainBackwardPassController::backward_pass_convolution_shortcut: fused per
 *        input channel like backward_pass_fully_connected (each weight read once, old weight for dInput).
 */
vector<Ctext> FHEONTrainStandardBackwardPassController::backward_pass_convolution_shortcut(vector<Ctext> &gradInputs, int batchSize, int inputWidth,
                    int inputChannels, int outputChannels, int stride, double learningRate, string layerName) {
    int outputWidth = inputWidth / stride;
    int inVecSize = inputWidth * inputWidth * batchSize;
    int outVecSize = outputWidth * outputWidth * batchSize;
    double fan_in = inputChannels;

    vector<Ctext> biasGradAll(outputChannels);
    run_parallel_for(outputChannels, [&](int start, int end) {
        for (int oc = start; oc < end; oc++) {
            biasGradAll[oc] = context->EvalSum(gradInputs[oc], outVecSize);
        }
    }, worker_thread_limit);

    auto start_time = startTime();
    vector<Ctext> dilatedGrad = (stride > 1) ? dilated_gradients(gradInputs, batchSize, outputWidth, stride) : gradInputs;

    vector<Ctext> gradientsCtxts(inputChannels);
    run_parallel_for(inputChannels, [&](int start, int end) {
        for (int inCh = start; inCh < end; inCh++) {
            string inputFileName = layerName + "_in_neuron_" + to_string(inCh);
            Ctext cachedInput = deserialize_and_load(inputFileName);
            Ctext inputGradient;
            for (int oc = 0; oc < outputChannels; oc++) {
                string fileName = layerName + "_out_neuron_" + to_string(oc) + "_in_" + to_string(inCh);
                Ctext weightCtxt = deserialize_and_load(fileName, fan_in, inVecSize);
                accumulate_term(context, inputGradient, context->EvalMultNoRelin(dilatedGrad[oc], weightCtxt));
                Ctext sumedWeightGrad = context->EvalSum(context->EvalMult(dilatedGrad[oc], cachedInput), inVecSize);
                weightCtxt = optimizer_update(weightCtxt, sumedWeightGrad, learningRate);
                serialize_and_store(weightCtxt, fileName);
            }
            gradientsCtxts[inCh] = relinearize_sum(context, inputGradient);
        }
    }, worker_thread_limit);

    run_parallel_for(outputChannels, [&](int start, int end) {
        for (int oc = start; oc < end; oc++) {
            string biasFileName = layerName + "_out_neuron_" + to_string(oc) + "_bias";
            Ctext biasCtxt = deserialize_and_load(biasFileName, 1.0, num_slots);
            biasCtxt = optimizer_update(biasCtxt, biasGradAll[oc], learningRate);
            serialize_and_store(biasCtxt, biasFileName);
        }
    }, worker_thread_limit);
    printTiming("Convolution Shortcut backward evaluating...", "shortcut", start_time);
    return gradientsCtxts;
}

/** see FHEONTrainBackwardPassController::backward_pass_global_avgpool (same keys), with the worker cap */
vector<Ctext> FHEONTrainStandardBackwardPassController::backward_pass_global_avgpool(vector<Ctext> &gradInputs, int batchSize, int dataWidth) {
    int spatialSize = dataWidth * dataWidth;
    double scaler = 1.0 / spatialSize;
    vector<Ctext> result(gradInputs.size());
    auto start_time = startTime();
    run_parallel_for(gradInputs.size(), [&](int start, int end) {
        for (int oc = start; oc < end; oc++) {
            Ctext dilated = backward_dilate(gradInputs[oc], batchSize, 1, dataWidth);
            result[oc] = context->EvalMult(repeat_encrypted_values(dilated, 1, spatialSize), scaler);
        }
    }, worker_thread_limit);
    printTiming("Backward global average pooling evaluated...", "backward_global_avgpool", start_time);
    return result;
}

/** see FHEONTrainBackwardPassController::backward_pass_global_avgpool_logdepth, with the worker cap */
vector<Ctext> FHEONTrainStandardBackwardPassController::backward_pass_global_avgpool_logdepth(vector<Ctext> &gradInputs, int batchSize, int dataWidth) {
    int spatialSize = dataWidth * dataWidth;
    double scaler = 1.0 / spatialSize;
    vector<Ctext> result(gradInputs.size());
    auto start_time = startTime();
    run_parallel_for(gradInputs.size(), [&](int start, int end) {
        for (int oc = start; oc < end; oc++) {
            Ctext dilated = backward_dilate_logdepth(gradInputs[oc], batchSize, 1, dataWidth);
            result[oc] = context->EvalMult(repeat_encrypted_values(dilated, 1, spatialSize), scaler);
        }
    }, worker_thread_limit);
    printTiming("Backward global average pooling (log-depth) evaluated...", "backward_global_avgpool", start_time);
    return result;
}

/**
 * @brief 2x2 average pooling backward, see FHEONTrainBackwardPassController::backward_pass_avgpool_logdepth (the 3
 *        window shifts hoisted on one precomputation).
 */
vector<Ctext> FHEONTrainStandardBackwardPassController::backward_pass_avgpool_logdepth(vector<Ctext> &encryptedInputs, int batchSize, int inputWidth, int dilateFactor) {
    if (dilateFactor != 2) {
        throw std::runtime_error("backward_pass_avgpool_logdepth: only 2x2 pooling windows (dilateFactor = 2) are supported");
    }
    int dilatedWidth = dilateFactor * inputWidth;
    double scaler = 1.0 / (dilateFactor * dilateFactor);
    vector<Ctext> pooledGradients(encryptedInputs.size());
    auto start_time = startTime();
    run_parallel_for(encryptedInputs.size(), [&](int start, int end) {
        auto cyclotomicOrder = context->GetCyclotomicOrder();
        for (int i = start; i < end; i++) {
            Ctext dilatedCtxt = backward_dilate_logdepth(encryptedInputs[i], batchSize, inputWidth, dilateFactor);
            auto digits = context->EvalFastRotationPrecompute(dilatedCtxt);
            Ctext smeared = context->EvalAddMany({dilatedCtxt,
                                                  context->EvalFastRotation(dilatedCtxt, -1, cyclotomicOrder, digits),
                                                  context->EvalFastRotation(dilatedCtxt, -dilatedWidth, cyclotomicOrder, digits),
                                                  context->EvalFastRotation(dilatedCtxt, -(dilatedWidth + 1), cyclotomicOrder, digits)});
            pooledGradients[i] = context->EvalMult(smeared, scaler);
        }
    }, worker_thread_limit);
    printTiming("Backward average pooling (log-depth) evaluated...", "backward_avgpool", start_time);
    return pooledGradients;
}

void FHEONTrainStandardBackwardPassController::bootstrap_all(vector<Ctext>& ciphertexts) {
    if (ciphertexts.empty()) return;
    if (bootstrap_function) {
        ciphertexts = bootstrap_function(ciphertexts);
        return;
    }
    for (auto &ciphertext : ciphertexts) {
        ciphertext = context->EvalBootstrap(ciphertext, bootstrap_iterations, bootstrap_iterations > 1 ? bootstrap_precision : 0);
    }
}
