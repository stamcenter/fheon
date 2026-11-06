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

#include <filesystem>
#include <mutex>
#include "FHEONTrainStandardForwardPassController.h"

namespace fs = std::filesystem;

/** pretrained CSV files, read once per run (see FHEONTrainForwardPassController's pretrainedFromCsv) */
static map<string, vector<double>> g_standardPretrainedCsvCache;
static std::mutex g_standardPretrainedCsvMutex;

static const vector<double>& standard_pretrained_csv(const string& csvPath) {
    std::lock_guard<std::mutex> lock(g_standardPretrainedCsvMutex);
    auto it = g_standardPretrainedCsvCache.find(csvPath);
    if (it != g_standardPretrainedCsvCache.end()) {
        return it->second;
    }
    auto data = utilsbatchdata::flatten_csv_data(utilsbatchdata::loadCSV(csvPath));
    return g_standardPretrainedCsvCache.emplace(csvPath, std::move(data)).first->second;
}

/** pretrained value idx of a CSV (0 for a missing or denormal value), see import_scalar_param */
static inline double standard_csv_value(const vector<double>* values, size_t idx, double fallback) {
    double value = (values && idx < values->size()) ? (*values)[idx] : fallback;
    return (std::abs(value) < 1e-10) ? 0.0 : value;
}

/** true when the parameter is still only in the pretrained CSV (no trained ciphertext on disk yet) */
static inline bool standard_from_csv(bool pretrainedFromCsv, const string& fileName) {
    return pretrainedFromCsv && !fs::exists(training_data_dir + fileName + ".bin");
}

/** add a product to an accumulator (the first product becomes the accumulator) */
static inline void accumulate(CryptoContext<DCRTPoly>& context, Ctext& accumulator, const Ctext& product) {
    if (accumulator) context->EvalAddInPlace(accumulator, product);
    else accumulator = product;
}

/** relinearize a sum of products taken without relinearization */
static inline Ctext relinearized(CryptoContext<DCRTPoly>& context, const Ctext& sum) {
    return (sum->NumberCiphertextElements() > 2) ? context->Relinearize(sum) : sum;
}

void FHEONTrainStandardForwardPassController::store_inputs(vector<Ctext>& encryptedInputs, int count, const string& prefix) {
    run_parallel_for(count, [&](int start, int end) {
        for (int inCh = start; inCh < end; inCh++) {
            string inputFileName = prefix + to_string(inCh);
            FHEONTrainSupportController::serialize_and_store(encryptedInputs[inCh], inputFileName);
        }
    }, worker_thread_limit);
}

/**
 * @brief 3x3 convolution (padding 1, stride 1 or 2) with ciphertext weights, see
 *        FHEONTrainForwardPassController::he_forward_convolution_optimized.
 *
 * The taps of every input channel are computed in parallel (same rotations as the original), then every output channel
 * accumulates its inputChannels * 9 products without relinearization and relinearizes once. Every weight is read once,
 * in the worker of its output channel.
 */
vector<Ctext> FHEONTrainStandardForwardPassController::he_forward_convolution_optimized(vector<Ctext> &encryptedInputs,
                int batchSize, int inputWidth, int inputChannels, int outputChannels, int stride, string layerName, bool pretrainedFromCsv) {

    constexpr int kernelSq = 9;
    int outputWidth = inputWidth / stride;
    int inVecSize = batchSize * inputWidth * inputWidth;
    int outVecSize = batchSize * outputWidth * outputWidth;
    vector<double> cleaningoutputVec;
    if (stride > 1) {
        cleaningoutputVec = generate_mixed_mask(outVecSize, inVecSize);
    }

    store_inputs(encryptedInputs, inputChannels, layerName + "_channel_");

    const vector<double>* csvWeights = nullptr;
    const vector<double>* csvBias = nullptr;
    if (pretrainedFromCsv) {
        csvWeights = &standard_pretrained_csv(weights_folder + layerName + "_weight.csv");
        csvBias = &standard_pretrained_csv(weights_folder + layerName + "_bias.csv");
    }
    double fan_in = inputChannels * kernelSq;

    auto start_time = startTime();
    vector<vector<Ctext>> taps(inputChannels, vector<Ctext>(kernelSq));
    run_parallel_for(inputChannels, [&](int start, int end) {
        auto cyclotomicOrder = context->GetCyclotomicOrder();
        for (int i = start; i < end; i++) {
            Ctext encryptedInput = replicate_across_ring(context, encryptedInputs[i]->Clone(), inVecSize, num_slots);
            auto digits = context->EvalFastRotationPrecompute(encryptedInput);
            Ctext first_shot = context->EvalRotate(encryptedInput, -1);
            Ctext second_shot = context->EvalFastRotation(encryptedInput, 1, cyclotomicOrder, digits);
            taps[i][0] = context->EvalRotate(first_shot, -inputWidth);
            taps[i][1] = context->EvalRotate(encryptedInput, -inputWidth);
            taps[i][2] = context->EvalRotate(second_shot, -inputWidth);
            taps[i][3] = first_shot;
            taps[i][4] = encryptedInput;
            taps[i][5] = second_shot;
            taps[i][6] = context->EvalRotate(first_shot, inputWidth);
            taps[i][7] = context->EvalFastRotation(encryptedInput, inputWidth, cyclotomicOrder, digits);
            taps[i][8] = context->EvalRotate(second_shot, inputWidth);
        }
    }, worker_thread_limit);

    vector<Ctext> output_ciphertexts(outputChannels);
    run_parallel_for(outputChannels, [&](int start, int end) {
        for (int outCh = start; outCh < end; outCh++) {
            Ctext result;
            for (int inCh = 0; inCh < inputChannels; inCh++) {
                for (int k = 0; k < kernelSq; k++) {
                    string weightsFileName = layerName + "_weights_" + to_string(outCh) + "_" + to_string(inCh) + "_" + to_string(k);
                    const Ctext& dataCtxt = taps[inCh][k];
                    if (standard_from_csv(pretrainedFromCsv, weightsFileName)) {
                        double val = standard_csv_value(csvWeights, (outCh * inputChannels + inCh) * kernelSq + k, 0.0);
                        import_scalar_param(val, weightsFileName);
                        vector<double> broadcastVec(inVecSize, val);
                        Ptext weightPtxt = context->MakeCKKSPackedPlaintext(broadcastVec, 1, dataCtxt->GetLevel());
                        accumulate(context, result, context->EvalMult(dataCtxt, weightPtxt));
                    } else {
                        Ctext kernelCtxt = deserialize_and_load(weightsFileName, fan_in);
                        accumulate(context, result, context->EvalMultNoRelin(dataCtxt, kernelCtxt));
                    }
                }
            }
            result = relinearized(context, result);

            if (stride > 1) {
                result = FHEONANNController::downsample_with_multiple_channels(result, inputWidth, stride, batchSize);
                result = context->EvalMult(result, context->MakeCKKSPackedPlaintext(cleaningoutputVec, 1, result->GetLevel()));
            }
            string biasFileName = layerName + "_bias_" + to_string(outCh);
            if (standard_from_csv(pretrainedFromCsv, biasFileName)) {
                double val = standard_csv_value(csvBias, outCh, 0.0);
                import_scalar_param(val, biasFileName);
                vector<double> broadcastVec(outVecSize, val);
                Ptext constantPtxt = context->MakeCKKSPackedPlaintext(broadcastVec, 1, result->GetLevel());
                output_ciphertexts[outCh] = context->EvalAdd(result, constantPtxt);
            } else {
                Ctext biasCtxt = deserialize_and_load(biasFileName);
                output_ciphertexts[outCh] = context->EvalAdd(result, biasCtxt);
            }
        }
    }, worker_thread_limit);
    printTiming("Convolution evaluated...", "conv", start_time);
    return output_ciphertexts;
}

/**
 * @brief 1x1 shortcut convolution (stride 1 or 2), see FHEONTrainForwardPassController::he_forward_convolution_shortcut_optimized.
 *
 * Weights streamed per output channel and multiplied without relinearization; the weights are no longer written back
 * after being read (deserialize_and_load already stores a weight it initializes).
 */
vector<Ctext> FHEONTrainStandardForwardPassController::he_forward_convolution_shortcut_optimized(vector<Ctext> &encryptedInputs,
                int batchSize, int inputWidth, int inputChannels, int outputChannels, int stride, string layerName, bool pretrainedFromCsv) {

    int outputWidth = inputWidth / stride;
    int inVecSize = batchSize * inputWidth * inputWidth;
    int outVecSize = batchSize * outputWidth * outputWidth;
    vector<double> cleaningoutputVec;
    if (stride > 1) {
        cleaningoutputVec = generate_mixed_mask(outVecSize, inVecSize);
    }

    store_inputs(encryptedInputs, inputChannels, layerName + "_in_neuron_");

    const vector<double>* csvWeights = nullptr;
    const vector<double>* csvBias = nullptr;
    if (pretrainedFromCsv) {
        csvWeights = &standard_pretrained_csv(weights_folder + layerName + "_weight.csv");
        csvBias = &standard_pretrained_csv(weights_folder + layerName + "_bias.csv");
    }
    double fan_in = inputChannels;

    vector<Ctext> output_ciphertexts(outputChannels);
    auto start_time = startTime();
    run_parallel_for(outputChannels, [&](int start, int end) {
        for (int oc = start; oc < end; oc++) {
            Ctext result;
            for (int inCh = 0; inCh < inputChannels; inCh++) {
                string fileName = layerName + "_out_neuron_" + to_string(oc) + "_in_" + to_string(inCh);
                if (standard_from_csv(pretrainedFromCsv, fileName)) {
                    double val = standard_csv_value(csvWeights, oc * inputChannels + inCh, 0.0);
                    import_scalar_param(val, fileName, inVecSize);
                    vector<double> broadcastVec(inVecSize, val);
                    Ptext weightPtxt = context->MakeCKKSPackedPlaintext(broadcastVec, 1, encryptedInputs[inCh]->GetLevel());
                    accumulate(context, result, context->EvalMult(encryptedInputs[inCh], weightPtxt));
                } else {
                    Ctext weight = deserialize_and_load(fileName, fan_in, inVecSize);
                    accumulate(context, result, context->EvalMultNoRelin(encryptedInputs[inCh], weight));
                }
            }
            result = relinearized(context, result);

            if (stride > 1) {
                result = FHEONANNController::downsample_with_multiple_channels(result, inputWidth, stride, batchSize);
                result = context->EvalMult(result, context->MakeCKKSPackedPlaintext(cleaningoutputVec, 1, result->GetLevel()));
            }

            string biasFileName = layerName + "_out_neuron_" + to_string(oc) + "_bias";
            if (standard_from_csv(pretrainedFromCsv, biasFileName)) {
                double val = standard_csv_value(csvBias, oc, 0.0);
                import_scalar_param(val, biasFileName, outVecSize);
                vector<double> broadcastVec(outVecSize, val);
                Ptext constantPtxt = context->MakeCKKSPackedPlaintext(broadcastVec, 1, result->GetLevel());
                output_ciphertexts[oc] = context->EvalAdd(result, constantPtxt);
            } else {
                Ctext bias = deserialize_and_load(biasFileName, 1.0, outVecSize);
                output_ciphertexts[oc] = context->EvalAdd(result, bias);
            }
        }
    }, worker_thread_limit);
    printTiming("Convolution shortcut evaluated...", "shortcut", start_time);
    return output_ciphertexts;
}

/**
 * @brief Global average pooling, see FHEONTrainForwardPassController::he_forward_global_avgpool.
 *
 * One rotate-and-add over the spatial positions (log2(w^2) rotations, the EvalSum keys) leaves the sum of image b in
 * slot b * w^2 for every image at once; the per-image sums are then brought to slot 0 with the rotations by w^2 of the
 * original and merged. The original computed one EvalSum per image (batchSize * log2(w^2) rotations).
 */
vector<Ctext> FHEONTrainStandardForwardPassController::he_forward_global_avgpool(vector<Ctext>& encryptedInputs, int batchSize, int dataWidth, int inputChannels, string layerName) {

    int spatialSize = dataWidth * dataWidth;
    vector<double> reducingMask = generate_scale_mask(spatialSize, batchSize);

    store_inputs(encryptedInputs, inputChannels, layerName + "_channel_");

    vector<Ctext> pooled_ciphertexts(inputChannels);
    auto start_time = startTime();
    run_parallel_for(inputChannels, [&](int start, int end) {
        for (int ch = start; ch < end; ch++) {
            Ctext sums = encryptedInputs[ch];
            for (int step = 1; step < spatialSize; step <<= 1) {
                sums = context->EvalAdd(sums, context->EvalRotate(sums, step));
            }
            vector<Ctext> result_ciphers(batchSize);
            for (int b = 0; b < batchSize; b++) {
                if (b != 0) {
                    sums = context->EvalRotate(sums, spatialSize);
                }
                result_ciphers[b] = sums;
            }
            Ctext merged = context->EvalMerge(result_ciphers);
            pooled_ciphertexts[ch] = context->EvalMult(merged, context->MakeCKKSPackedPlaintext(reducingMask, 1, merged->GetLevel()));
        }
    }, worker_thread_limit);
    printTiming("Global pooling Optimized...", "pooling", start_time);
    return pooled_ciphertexts;
}

/**
 * @brief ReLU of every channel (Chebyshev approximation of the input normalized by 2 * scale), see
 *        FHEONTrainForwardPassController::he_forward_relu. The unused mask plaintext of the original is not encoded.
 */
vector<Ctext> FHEONTrainStandardForwardPassController::he_forward_relu(vector<Ctext>& encryptedInputs, vector<int> scaleValues, int inputChannels, int vectorSize, int polyDegree, string layerName) {
    (void)vectorSize;
    double lowerBound = -1;
    double upperBound = 1;
    vector<Ctext> relu_results(inputChannels);

    store_inputs(encryptedInputs, inputChannels, layerName + "_channel_");

    auto start_time = startTime();
    run_parallel_for(inputChannels, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            int scaleVal = scaleValues[i];
            if (scaleVal <= 1) {
                relu_results[i] = context->EvalChebyshevFunction([](double x) -> double { return (x < 0) ? 0 : x; },
                                                                 encryptedInputs[i], lowerBound, upperBound, polyDegree);
            } else {
                Ctext normalized = context->EvalMult(encryptedInputs[i], 1.0 / (2.0 * scaleVal));
                relu_results[i] = context->EvalChebyshevFunction([](double x) -> double { return (x < 0) ? 0 : x; },
                                                                 normalized, lowerBound, upperBound, polyDegree);
                relu_results[i] = context->EvalMult(relu_results[i], (double)(2 * scaleVal));
            }
        }
    }, worker_thread_limit);
    printTiming("ReLU evaluated...", "relue", start_time);
    return relu_results;
}

/**
 * @brief 2x2 average pooling, see FHEONTrainForwardPassController::he_forward_avgPool_optimized (window sums with
 *        hoisted rotations, then the downsampling of every channel in parallel).
 */
vector<Ctext> FHEONTrainStandardForwardPassController::he_forward_avgPool_optimized(vector<Ctext>& encryptedInputs, int batchSize, int inputWidth, int kernelWidth, int stridingLen, string layerName){
    int kernelSize = kernelWidth * kernelWidth;
    int inputSize = inputWidth * inputWidth;
    int outputWidth = inputWidth / stridingLen;
    int inVecSize = batchSize * inputSize;
    int outVecSize = batchSize * outputWidth * outputWidth;
    int inputChannels = encryptedInputs.size();
    int encode_level = encryptedInputs[0]->GetLevel();

    Ptext reducingMask = context->MakeCKKSPackedPlaintext(generate_scale_mask(kernelSize, inVecSize), 1, encode_level);
    Ptext cleaningInMask = context->MakeCKKSPackedPlaintext(generate_mixed_mask(inputSize, inVecSize), 1, encode_level);
    vector<double> cleaningoutputVec = generate_mixed_mask(outVecSize, inVecSize);

    store_inputs(encryptedInputs, inputChannels, layerName + "_channel_");

    auto start_time = startTime();
    vector<Ctext> pooled_ciphertexts(inputChannels);
    run_parallel_for(inputChannels, [&](int start, int end) {
        auto cyclotomicOrder = context->GetCyclotomicOrder();
        for (int i = start; i < end; i++) {
            const Ctext& encryptedInput = encryptedInputs[i];
            auto digits = context->EvalFastRotationPrecompute(encryptedInput);
            Ctext down = context->EvalFastRotation(encryptedInput, inputWidth, cyclotomicOrder, digits);
            Ctext sum_cipher = context->EvalAddMany({encryptedInput,
                                                     context->EvalFastRotation(encryptedInput, 1, cyclotomicOrder, digits),
                                                     down, context->EvalRotate(down, 1)});
            Ctext windows = context->EvalMult(sum_cipher, reducingMask);
            if (inputWidth <= 2) {
                vector<Ctext> inn_ciphers;
                for (int b = 0; b < batchSize; b++) {
                    inn_ciphers.push_back(context->EvalSum(context->EvalMult(windows, cleaningInMask), inputSize));
                    windows = context->EvalRotate(windows, inputSize);
                }
                pooled_ciphertexts[i] = context->EvalMerge(inn_ciphers);
            } else {
                Ctext result = FHEONANNController::downsample_with_multiple_channels(windows, inputWidth, stridingLen, batchSize);
                pooled_ciphertexts[i] = context->EvalMult(result, context->MakeCKKSPackedPlaintext(cleaningoutputVec, 1, result->GetLevel()));
            }
        }
    }, worker_thread_limit);
    printTiming("Average pooling Optimized...", "pooling", start_time);
    return pooled_ciphertexts;
}

/**
 * @brief Fully connected layer with one ciphertext per neuron, see FHEONTrainForwardPassController::he_forward_linear_multiple_outputs.
 *
 * Weights streamed per output neuron (the original holds the whole outputSize x inputSize matrix and writes every
 * weight back after reading it), products accumulated without relinearization.
 */
vector<Ctext> FHEONTrainStandardForwardPassController::he_forward_linear_multiple_outputs(vector<Ctext>& encryptedInputs,
                                                int vectorSize, int inputSize, int outputSize, string layerName, bool pretrainedFromCsv){

    store_inputs(encryptedInputs, inputSize, layerName + "_in_neuron_");

    const vector<double>* csvWeights = nullptr;
    const vector<double>* csvBias = nullptr;
    if (pretrainedFromCsv) {
        csvWeights = &standard_pretrained_csv(weights_folder + layerName + "_weight.csv");
        csvBias = &standard_pretrained_csv(weights_folder + layerName + "_bias.csv");
    }
    double fan_in = inputSize;

    vector<Ctext> results(outputSize);
    auto start_time = startTime();
    run_parallel_for(outputSize, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            Ctext temp;
            for (int inCh = 0; inCh < inputSize; inCh++) {
                string fileName = layerName + "_out_neuron_" + to_string(i) + "_in_" + to_string(inCh);
                if (standard_from_csv(pretrainedFromCsv, fileName)) {
                    double val = standard_csv_value(csvWeights, i * inputSize + inCh, 0.0);
                    import_scalar_param(val, fileName, vectorSize);
                    vector<double> broadcastVec(vectorSize, val);
                    Ptext weightPtxt = context->MakeCKKSPackedPlaintext(broadcastVec, 1, encryptedInputs[inCh]->GetLevel());
                    accumulate(context, temp, context->EvalMult(encryptedInputs[inCh], weightPtxt));
                } else {
                    Ctext weight = deserialize_and_load(fileName, fan_in, vectorSize);
                    accumulate(context, temp, context->EvalMultNoRelin(encryptedInputs[inCh], weight));
                }
            }
            temp = relinearized(context, temp);
            string biasFileName = layerName + "_out_neuron_" + to_string(i) + "_bias";
            if (standard_from_csv(pretrainedFromCsv, biasFileName)) {
                double val = standard_csv_value(csvBias, i, 0.0);
                import_scalar_param(val, biasFileName, vectorSize);
                vector<double> broadcastVec(vectorSize, val);
                Ptext constantPtxt = context->MakeCKKSPackedPlaintext(broadcastVec, 1, temp->GetLevel());
                results[i] = context->EvalAdd(temp, constantPtxt);
            } else {
                Ctext bias = deserialize_and_load(biasFileName, 1.0, vectorSize);
                results[i] = context->EvalAdd(temp, bias);
            }
        }
    }, worker_thread_limit);
    printTiming("Linear multiple outputs...", "linear", start_time);
    return results;
}

/**
 * @brief Softmax exp(x_i) / sum_j exp(x_j) with exp(x) ~ 1 + x + x^2 / 2 and the reciprocal from Newton-Raphson on
 *        1/sqrt (then squared), see FHEONTrainForwardPassController::he_forward_softmax.
 */
vector<Ctext> FHEONTrainStandardForwardPassController::he_forward_softmax(vector<Ctext> &encryptedInputs, const vector<int>& scalingValues, int inputSize, int appxPolyDegree) {
    (void)appxPolyDegree;
    (void)inputSize;
    int softmaxInputSize = encryptedInputs.size();
    vector<Ctext> expInputs(softmaxInputSize);
    auto start_time = startTime();
    run_parallel_for(softmaxInputSize, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            expInputs[i] = context->EvalMult(context->EvalSquare(encryptedInputs[i]), 0.5);
            expInputs[i] = context->EvalAdd(expInputs[i], encryptedInputs[i]);
            expInputs[i] = context->EvalAdd(expInputs[i], 1.0);
        }
    }, worker_thread_limit);

    double sumVal = 0.0;
    for (int i = 0; i < softmaxInputSize; i++) {
        sumVal += (pow(scalingValues[i], 2.0) / 2.0) + (double)scalingValues[i] + 1.0;
    }
    cout << "Worst-case upper bound on sum of exponentials: " << sumVal << endl;

    vector<Ctext> sumedExp = {context->EvalAddMany(expInputs)};
    bootstrap_all(sumedExp);
    double initValue = 1.0 / sqrt(sumVal / 2.0);
    Ctext invSqrt = newton_inverse_sqrt_all(sumedExp, num_slots, initValue, 4, true)[0];
    Ctext reciprocal = context->EvalSquare(invSqrt);

    vector<Ctext> softMaxCtxts(softmaxInputSize);
    run_parallel_for(softmaxInputSize, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            softMaxCtxts[i] = context->EvalMult(expInputs[i], reciprocal);
        }
    }, worker_thread_limit);
    printTiming("Softmax evaluated...", "Softmax", start_time);
    return softMaxCtxts;
}

/**
 * @brief Batch normalization of every channel over its inputSize slots, see FHEONTrainForwardPassController::he_forward_batch_norm.
 *
 * Statistics of all channels in parallel; the bootstrappings on the calling thread (EvalBootstrap parallelizes itself
 * with OpenMP); Newton iterations of all channels in lock step; scale, shift and caches in parallel. Same cache files.
 */
vector<Ctext> FHEONTrainStandardForwardPassController::he_forward_batch_norm(vector<Ctext> &encryptedInputs, int inputSize, double appxInitValue, int appxDegree,
                                                                           string layerName, bool isBootstrap, bool pretrainedFromCsv) {
    double scaler = (1.0 / inputSize);
    auto cleaningMaskVec = generate_scale_and_clean_mask(scaler, inputSize);
    int numChannels = encryptedInputs.size();
    vector<Ctext> normResults(numChannels);
    vector<Ctext> topVecs(numChannels);
    vector<Ctext> varVals(numChannels);

    const vector<double>* csvGamma = nullptr;
    const vector<double>* csvBeta = nullptr;
    if (pretrainedFromCsv) {
        csvGamma = &standard_pretrained_csv(weights_folder + layerName + "_gamma.csv");
        csvBeta = &standard_pretrained_csv(weights_folder + layerName + "_beta.csv");
    }

    auto start_time = startTime();
    run_parallel_for(numChannels, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            Ctext sumVal = context->EvalSum(encryptedInputs[i], inputSize);
            Ctext meanVal = context->EvalMult(sumVal, context->MakeCKKSPackedPlaintext(cleaningMaskVec, 1, sumVal->GetLevel(), nullptr, inputSize));
            meanVal = repeat_encrypted_values(meanVal, 1, inputSize);
            topVecs[i] = context->EvalSub(encryptedInputs[i], meanVal);
            Ctext varValSum = context->EvalSum(context->EvalSquare(topVecs[i]), inputSize);
            Ctext varVal = context->EvalMult(varValSum, context->MakeCKKSPackedPlaintext(cleaningMaskVec, 1, varValSum->GetLevel(), nullptr, inputSize));
            varVals[i] = repeat_encrypted_values(varVal, 1, inputSize);
        }
    }, worker_thread_limit);
    measureTiming("Batch norm evaluating...", "batch_norm", start_time);

    if (isBootstrap) bootstrap_all(varVals);
    double initVal = (appxInitValue > 0.0) ? appxInitValue : 0.005;
    vector<Ctext> approxStd = newton_inverse_sqrt_all(varVals, inputSize, initVal, appxDegree, isBootstrap);

    run_parallel_for(numChannels, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            Ctext xHat = context->EvalMult(topVecs[i], approxStd[i]);
            string xHatFileName = "batch_norm_xhat_" + layerName + "_channel_" + to_string(i);
            serialize_and_store(xHat, xHatFileName);

            string fileName = "batch_norm_" + layerName + "_channel_" + to_string(i);
            string biasFileName = "batch_norm_bias_" + layerName + "_channel_" + to_string(i);
            Ctext learnedandinvstd;
            if (standard_from_csv(pretrainedFromCsv, fileName)) {
                double val = standard_csv_value(csvGamma, i, 1.0);
                import_scalar_param(val, fileName, inputSize);
                vector<double> gammaVec(inputSize, val);
                learnedandinvstd = context->EvalMult(approxStd[i], context->MakeCKKSPackedPlaintext(gammaVec, 1, approxStd[i]->GetLevel()));
            } else {
                Ctext learnedCtxt = deserialize_and_load(fileName, 1.0, inputSize);
                learnedandinvstd = context->EvalMult(learnedCtxt, approxStd[i]);
            }
            Ctext scaledXhat = context->EvalMult(learnedandinvstd, topVecs[i]);
            if (standard_from_csv(pretrainedFromCsv, biasFileName)) {
                double val = standard_csv_value(csvBeta, i, 0.0);
                import_scalar_param(val, biasFileName, inputSize);
                vector<double> betaVec(inputSize, val);
                Ptext constantPtxt = context->MakeCKKSPackedPlaintext(betaVec, 1, scaledXhat->GetLevel());
                normResults[i] = context->EvalAdd(scaledXhat, constantPtxt);
            } else {
                Ctext learnedbiasCtxt = deserialize_and_load(biasFileName, 1.0, inputSize);
                normResults[i] = context->EvalAdd(scaledXhat, learnedbiasCtxt);
            }
            string invstdFileName = "batch_norm_invstd" + layerName + "_channel_" + to_string(i);
            serialize_and_store(learnedandinvstd, invstdFileName);
        }
    }, worker_thread_limit);
    measureTiming("Batch norm evaluating...", "batch_norm", start_time);
    return normResults;
}

/**
 * @brief y <- y * (1.5 - 0.5 * x * y^2) for every ciphertext x, from the constant y0 = initValue.
 *
 * The first iteration is evaluated on the constant: y1 = 1.5 * y0 - 0.5 * y0^3 * x (one level instead of three, and no
 * encrypted zeros). As in the original, y is bootstrapped before every second iteration when isBootstrap; the
 * bootstrappings run on the calling thread and the iterations of all ciphertexts in parallel.
 */
vector<Ctext> FHEONTrainStandardForwardPassController::newton_inverse_sqrt_all(vector<Ctext>& encryptedInputs, int inputSize, double initValue,
                                                                            int appxDegree, bool isBootstrap) {
    (void)inputSize;
    int count = encryptedInputs.size();
    vector<Ctext> y(count);
    if (appxDegree <= 0) {
        for (int i = 0; i < count; i++) {
            y[i] = context->EvalAdd(context->EvalMult(encryptedInputs[i], 0.0), initValue);
        }
        return y;
    }
    run_parallel_for(count, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            y[i] = context->EvalAdd(context->EvalMult(encryptedInputs[i], -0.5 * initValue * initValue * initValue), 1.5 * initValue);
        }
    }, worker_thread_limit);
    const int itersPerBootstrap = 2;
    for (int n = 1; n < appxDegree; n++) {
        if (n % itersPerBootstrap == 0 && isBootstrap) bootstrap_all(y);
        run_parallel_for(count, [&](int start, int end) {
            for (int i = start; i < end; i++) {
                Ctext g = context->EvalMult(encryptedInputs[i], context->EvalSquare(y[i]));
                Ctext t = context->EvalSub(1.5, context->EvalMult(g, 0.5));
                y[i] = context->EvalMult(y[i], t);
            }
        }, worker_thread_limit);
    }
    return y;
}

void FHEONTrainStandardForwardPassController::bootstrap_all(vector<Ctext>& ciphertexts) {
    if (ciphertexts.empty()) return;
    if (bootstrap_function) {
        ciphertexts = bootstrap_function(ciphertexts);
        return;
    }
    for (auto &ciphertext : ciphertexts) {
        ciphertext = context->EvalBootstrap(ciphertext, bootstrap_iterations, bootstrap_iterations > 1 ? bootstrap_precision : 0);
    }
}

/** the rescaling FLEXIBLEAUTO leaves pending after a multiplication */
static Ctext standard_rescaled(CryptoContext<DCRTPoly>& context, const Ctext& input) {
    if (input->GetNoiseScaleDeg() < 2) return input;
    Ctext result = input->Clone();
    context->GetScheme()->ModReduceInternalInPlace(result, BASE_NUM_LEVELS_TO_DROP);
    return result;
}

vector<Ctext> FHEONTrainStandardForwardPassController::he_forward_batch_norm(vector<Ctext> &encryptedInputs, int inputSize,
                                        const vector<double>& channelInverseStdInit, int appxDegree, string layerName, bool isBootstrap,
                                        bool pretrainedFromCsv, double epsilon) {
    double scaler = (1.0 / inputSize);
    auto cleaningMaskVec = generate_scale_and_clean_mask(scaler, inputSize);
    int numChannels = encryptedInputs.size();
    vector<Ctext> normResults(numChannels), topVecs(numChannels), normalizedVars(numChannels);
    vector<double> y0(numChannels);
    for (int i = 0; i < numChannels; i++) y0[i] = (i < (int)channelInverseStdInit.size()) ? channelInverseStdInit[i] : 1.0;

    const vector<double>* csvGamma = nullptr;
    const vector<double>* csvBeta = nullptr;
    if (pretrainedFromCsv) {
        csvGamma = &standard_pretrained_csv(weights_folder + layerName + "_gamma.csv");
        csvBeta = &standard_pretrained_csv(weights_folder + layerName + "_beta.csv");
    }

    auto start_time = startTime();
    run_parallel_for(numChannels, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            Ctext sumVal = context->EvalSum(encryptedInputs[i], inputSize);
            Ctext meanVal = context->EvalMult(sumVal, context->MakeCKKSPackedPlaintext(cleaningMaskVec, 1, sumVal->GetLevel(), nullptr, inputSize));
            meanVal = repeat_encrypted_values(meanVal, 1, inputSize);
            topVecs[i] = context->EvalSub(encryptedInputs[i], meanVal);
            Ctext varValSum = context->EvalSum(context->EvalSquare(topVecs[i]), inputSize);
            Ctext varVal = context->EvalMult(varValSum, context->MakeCKKSPackedPlaintext(cleaningMaskVec, 1, varValSum->GetLevel(), nullptr, inputSize));
            varVal = context->EvalAdd(repeat_encrypted_values(varVal, 1, inputSize), epsilon);
            /** u = (var + eps) y0^2 ~ 1 */
            normalizedVars[i] = context->EvalMult(varVal, y0[i] * y0[i]);
        }
    }, worker_thread_limit);
    measureTiming("Batch norm evaluating...", "batch_norm", start_time);

    if (isBootstrap) bootstrap_all(normalizedVars);
    vector<Ctext> w = newton_inverse_sqrt_all(normalizedVars, inputSize, 1.0, appxDegree, isBootstrap);

    run_parallel_for(numChannels, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            /** 1/std = w y0: xhat = (centered y0) w (y0 on the shallow centered values) */
            Ctext xHat = context->EvalMult(context->EvalMult(topVecs[i], y0[i]), w[i]);
            string xHatFileName = "batch_norm_xhat_" + layerName + "_channel_" + to_string(i);
            serialize_and_store(xHat, xHatFileName);
            Ctext inverseStd = context->EvalMult(w[i], y0[i]);

            string fileName = "batch_norm_" + layerName + "_channel_" + to_string(i);
            string biasFileName = "batch_norm_bias_" + layerName + "_channel_" + to_string(i);
            Ctext learnedandinvstd;
            if (standard_from_csv(pretrainedFromCsv, fileName)) {
                double val = standard_csv_value(csvGamma, i, 1.0);
                import_scalar_param(val, fileName, inputSize);
                learnedandinvstd = context->EvalMult(inverseStd, val);
            } else {
                Ctext learnedCtxt = deserialize_and_load(fileName, 1.0, inputSize);
                learnedandinvstd = context->EvalMult(learnedCtxt, inverseStd);
            }
            Ctext scaledXhat = context->EvalMult(learnedandinvstd, topVecs[i]);
            if (standard_from_csv(pretrainedFromCsv, biasFileName)) {
                double val = standard_csv_value(csvBeta, i, 0.0);
                import_scalar_param(val, biasFileName, inputSize);
                vector<double> betaVec(inputSize, val);
                Ptext betaPtxt = context->MakeCKKSPackedPlaintext(betaVec, 1, scaledXhat->GetLevel());
                normResults[i] = context->EvalAdd(scaledXhat, betaPtxt);
            } else {
                Ctext learnedbiasCtxt = deserialize_and_load(biasFileName, 1.0, inputSize);
                normResults[i] = context->EvalAdd(scaledXhat, learnedbiasCtxt);
            }
            string invstdFileName = "batch_norm_invstd" + layerName + "_channel_" + to_string(i);
            serialize_and_store(learnedandinvstd, invstdFileName);
        }
    }, worker_thread_limit);
    measureTiming("Batch norm evaluating...", "batch_norm", start_time);
    return normResults;
}

vector<Ctext> FHEONTrainStandardForwardPassController::normalize_classes(vector<Ctext>& values, int batchSize, double low, double high,
                                                                       int newtonIterations, int degree) {
    int numClasses = values.size();
    Ctext sums = standard_rescaled(context, context->EvalAddMany(values));
    /** the slots past the batch hold no image: their sum (0) is outside [low, high], where the Chebyshev polynomial is
     *  huge and ruins the precision of the whole ciphertext; it is moved to the middle (plaintext addition, no level) */
    int numSlots = context->GetRingDimension() / 2;
    vector<double> offsets(numSlots, 0.5 * (low + high));
    for (int i = 0; i < batchSize && i < numSlots; i++) offsets[i] = 0.0;
    Ptext offsetPtxt = context->MakeCKKSPackedPlaintext(offsets, 1, sums->GetLevel(), nullptr, numSlots);
    sums = context->EvalAdd(sums, offsetPtxt);
    Ctext reciprocal = context->EvalChebyshevFunction([](double x) -> double { return 1.0 / x; }, sums, low, high, degree);
    if (newtonIterations > 0) {
        vector<Ctext> group = values;
        group.push_back(reciprocal);
        bootstrap_all(group);
        reciprocal = group.back();
        group.pop_back();
        values = group;
        for (int n = 0; n < newtonIterations; n++) {
            reciprocal = context->EvalMult(reciprocal, context->EvalSub(2.0, context->EvalMult(sums, reciprocal)));
        }
    }
    vector<Ctext> normalized(numClasses);
    run_parallel_for(numClasses, [&](int start, int end) {
        for (int k = start; k < end; k++) normalized[k] = context->EvalMult(values[k], reciprocal);
    }, worker_thread_limit);
    return normalized;
}

vector<Ctext> FHEONTrainStandardForwardPassController::he_forward_softmax_normalized(vector<Ctext>& logits, int batchSize, double logitBound,
                                                                                   int finalNewtonIterations) {
    auto start_time = startTime();
    int numClasses = logits.size();
    int squarings = std::max(0, (int)std::ceil(std::log2(logitBound / 0.25)));
    double inputScale = std::ldexp(1.0, -squarings);
    int numSlots = context->GetRingDimension() / 2;
    vector<double> batchMask(numSlots, 0.0);
    for (int i = 0; i < batchSize && i < numSlots; i++) batchMask[i] = 1.0;

    /** q = exp(z / 2^r) on the image slots: 1 + x + x^2/2 + x^3/6 + x^4/24, |x| <= 1/4 */
    vector<Ctext> q(numClasses);
    run_parallel_for(numClasses, [&](int start, int end) {
        for (int k = start; k < end; k++) {
            Ctext x = context->EvalMult(logits[k], inputScale);
            Ctext x2 = context->EvalSquare(x);
            Ctext x3 = context->EvalMult(x2, x);
            Ctext x4 = context->EvalSquare(x2);
            Ctext poly = context->EvalAdd(context->EvalAdd(x, 1.0), context->EvalMult(x2, 0.5));
            poly = standard_rescaled(context, context->EvalAdd(poly, context->EvalAdd(context->EvalMult(x3, 1.0 / 6.0), context->EvalMult(x4, 1.0 / 24.0))));
            Ptext maskPtxt = context->MakeCKKSPackedPlaintext(batchMask, 1, poly->GetLevel(), nullptr, numSlots);
            q[k] = context->EvalMult(poly, maskPtxt);
        }
    }, worker_thread_limit);

    double n = numClasses;
    bootstrap_all(q);
    bool last = (squarings == 0);
    q = normalize_classes(q, batchSize, 0.75 * n * std::exp(-0.25), 1.25 * n * std::exp(0.25), last ? finalNewtonIterations : 0, last ? 31 : 15);
    for (int step = 1; step <= squarings; step++) {
        bootstrap_all(q);
        run_parallel_for(numClasses, [&](int start, int end) {
            for (int k = start; k < end; k++) q[k] = context->EvalSquare(q[k]);
        }, worker_thread_limit);
        last = (step == squarings);
        /** intermediate normalizations only bound the values (softmax is invariant to a per-image scale); the last is exact */
        q = normalize_classes(q, batchSize, 0.5 / n, 1.5, last ? finalNewtonIterations : 0, last ? 31 : 15);
    }
    printTiming("Softmax (normalized squaring) evaluated...", "Softmax", start_time);
    return q;
}
