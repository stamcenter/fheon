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
#include "FHEONTrainForwardPassController.h"
#include "UtilsTraining.h"

namespace fs = std::filesystem;

// Per-run cache of pretrained CSV weight files, keyed by full path -- avoids
// re-reading and re-parsing a whole layer's CSV (up to ~590,000 values) on every
// individual weight-tap lookup. See he_forward_convolution_optimized's
// pretrainedFromCsv doc comment for why this exists at all.
static map<string, vector<double>> g_pretrainedCsvCache;

static const vector<double>& load_pretrained_csv_cached(const string& csvPath) {
    auto it = g_pretrainedCsvCache.find(csvPath);
    if (it != g_pretrainedCsvCache.end()) {
        return it->second;
    }
    auto data = utilsbatchdata::flatten_csv_data(utilsbatchdata::loadCSV(csvPath));
    auto inserted = g_pretrainedCsvCache.emplace(csvPath, std::move(data));
    return inserted.first->second;
}

static inline double csv_scalar_or_default(const vector<double>& values, size_t idx, double fallback) {
    if (idx < values.size()) {
        return values[idx];
    }
    return fallback;
}

void FHEONTrainForwardPassController::setContext(CryptoContext<DCRTPoly>& in_context){
    context = in_context;
    FHEONANNController::setContext(in_context);
    FHEONTrainSupportController::setContext(in_context);
}

/**
 * @brief 3x3 convolution (padding 1, stride 1 or 2) of the training forward pass, with ciphertext (trainable) weights.
 *
 * Every input channel is replicated over the ring and rotated into its 9 taps; every output channel sums, over the
 * input channels and taps, the products with its encrypted weights, then (stride 2) downsamples all images at once
 * (downsample_with_multiple_channels) and adds its encrypted bias. The inputs are cached for the backward pass.
 *
 * Parameter files (training_data_dir): layerName_weights_{out}_{in}_{tap} (one broadcast scalar per file, created with
 * Kaiming init on first use) and layerName_bias_{out}. Cached inputs: layerName_channel_{in}.
 *
 * @param encryptedInputs   One ciphertext per input channel, each holding the batch of feature maps
 *                          (batchSize x inputWidth x inputWidth, image after image).
 * @param batchSize         Number of images in every ciphertext.
 * @param inputWidth        Width (and height) of the input feature maps.
 * @param inputChannels     Number of input channels.
 * @param outputChannels    Number of output channels.
 * @param stride            1 or 2.
 * @param layerName         Prefix of the parameter and cache files.
 * @param pretrainedFromCsv If true, a weight or bias without a trained ciphertext yet is read from
 *                          weights_folder + layerName + "_weight.csv" / "_bias.csv" and used as a plaintext (and stored as
 *                          its first ciphertext for the backward pass).
 *
 * @return One ciphertext per output channel, the batch packed image after image at the output width.
 */
vector<Ctext> FHEONTrainForwardPassController::he_forward_convolution_optimized(vector<Ctext> &encryptedInputs,
                int batchSize, int inputWidth, int inputChannels, int outputChannels, int stride, string layerName, bool pretrainedFromCsv) {

    constexpr int kernelSq = 9;
    int outputWidth = inputWidth / stride;
    int inputSize = inputWidth * inputWidth;
    int outputSize = outputWidth * outputWidth;
    int inVecSize = batchSize * inputSize;
    int outVecSize = batchSize * outputSize;
    vector<double> cleaningoutputVec;
    if (stride > 1) {
        cleaningoutputVec = generate_mixed_mask(outVecSize, inVecSize);
    }

    run_parallel_for(inputChannels, [&](int start, int end) {
        for(int inCh = start; inCh < end; inCh++){
            string inputFileName = layerName+ "_channel_" + to_string(inCh);
            FHEONTrainSupportController::serialize_and_store(encryptedInputs[inCh], inputFileName);
        }
    });

    const vector<double>* csvWeights = nullptr;
    const vector<double>* csvBias = nullptr;
    if (pretrainedFromCsv) {
        csvWeights = &load_pretrained_csv_cached(weights_folder + layerName + "_weight.csv");
        csvBias = &load_pretrained_csv_cached(weights_folder + layerName + "_bias.csv");
    }
    {
        double fan_in = inputChannels * kernelSq;
        auto in_worker = [&](int start, int end) {
            for (int outCh = start; outCh < end; outCh++) {
                for (int inCh = 0; inCh < inputChannels; inCh++) {
                    for (int k = 0; k < kernelSq; k++) {
                        string weightsFileName = layerName + "_weights_" + to_string(outCh) + "_" + to_string(inCh) + "_" + to_string(k);
                        if (pretrainedFromCsv && !fs::exists(training_data_dir + weightsFileName + ".bin")) {
                            continue;
                        }
                        deserialize_and_load(weightsFileName, fan_in);
                    }
                }
                string biasFileName = layerName + "_bias_" + to_string(outCh);
                if (pretrainedFromCsv && !fs::exists(training_data_dir + biasFileName + ".bin")) {
                    continue;
                }

                deserialize_and_load(biasFileName);
            }
        };
        run_parallel_for(outputChannels, in_worker);
    }

    int numThreads = std::min(inputChannels, (int)thread::hardware_concurrency());
    vector<thread> threads(numThreads);

    // Precompute rotations (same as before)
    vector<vector<Ctext>> batched_rotated_ciphertexts(inputChannels);


    auto start_time = startTime();
    auto in_worker = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            vector<Ctext> rotated_ciphertexts;
            Ctext encryptedInput = encryptedInputs[i]->Clone();
            encryptedInput = replicate_across_ring(context, encryptedInput, inVecSize, num_slots);
            auto digits = context->EvalFastRotationPrecompute(encryptedInput);

            Ctext first_shot = context->EvalRotate(encryptedInput, -1);
            Ctext second_shot = context->EvalFastRotation(encryptedInput, 1, context->GetCyclotomicOrder(), digits);
            rotated_ciphertexts.push_back(context->EvalRotate(first_shot, -inputWidth));
            rotated_ciphertexts.push_back(context->EvalRotate(encryptedInput, -inputWidth));
            rotated_ciphertexts.push_back(context->EvalRotate(second_shot, -inputWidth));
            rotated_ciphertexts.push_back(first_shot);
            rotated_ciphertexts.push_back(encryptedInput);
            rotated_ciphertexts.push_back(second_shot);
            rotated_ciphertexts.push_back(context->EvalRotate(first_shot, inputWidth));
            rotated_ciphertexts.push_back(context->EvalFastRotation(encryptedInput, inputWidth, context->GetCyclotomicOrder(), digits));
            rotated_ciphertexts.push_back(context->EvalRotate(second_shot, inputWidth));
            batched_rotated_ciphertexts[i] = rotated_ciphertexts;
            rotated_ciphertexts.clear();
        }
    };
    run_parallel_for(inputChannels, in_worker);

    vector<Ctext> output_ciphertexts(outputChannels);
    auto worker = [&](int start, int end) {
        for (int outCh = start; outCh < end; outCh++) {
            vector<Ctext> batched_results(inputChannels);
            vector<Ctext> mult_results(kernelSq);

            for (int inCh = 0; inCh < inputChannels; inCh++) {
                for (int k = 0; k < kernelSq; k++) {

                    string weightsFileName = layerName + "_weights_" + to_string(outCh) + "_" + to_string(inCh) + "_" + to_string(k);
                    Ctext dataCtxt = batched_rotated_ciphertexts[inCh][k];
                    if (pretrainedFromCsv && !fs::exists(training_data_dir + weightsFileName + ".bin")) {
                        int idx = (outCh * inputChannels + inCh) * kernelSq + k;
                        double val = csv_scalar_or_default(*csvWeights, idx, 0.0);
                        if (std::abs(val) < 1e-10) val = 0.0;
                        import_scalar_param(val, weightsFileName);
                        vector<double> broadcastVec(inVecSize, val);
                        Ptext weightPtxt = context->MakeCKKSPackedPlaintext(broadcastVec, 1, dataCtxt->GetLevel());
                        mult_results[k] = context->EvalMult(dataCtxt, weightPtxt);
                    } 
                    else {
                        Ctext kernelCtxt = deserialize_and_load(weightsFileName);
                        mult_results[k] = context->EvalMult(dataCtxt, kernelCtxt);
                    }
                }
                batched_results[inCh] = context->EvalAddMany(mult_results);
            }

            // Sum input-channel results
            Ctext result = context->EvalAddMany(batched_results);

            if (stride > 1) {
                result = FHEONANNController::downsample_with_multiple_channels(result, inputWidth, stride, batchSize);
                result = context->EvalMult(result, context->MakeCKKSPackedPlaintext(cleaningoutputVec, 1, result->GetLevel()));
            }
            string biasFileName = layerName + "_bias_" + to_string(outCh);
            if (pretrainedFromCsv && !fs::exists(training_data_dir + biasFileName + ".bin")) {
                double val = csv_scalar_or_default(*csvBias, outCh, 0.0);
                if (std::abs(val) < 1e-10) val = 0.0;
                import_scalar_param(val, biasFileName);
                vector<double> broadcastVec(outVecSize, val);
                Ptext biasPtxt = context->MakeCKKSPackedPlaintext(broadcastVec, 1, result->GetLevel());
                output_ciphertexts[outCh] = context->EvalAdd(result, biasPtxt);
            } else {
                Ctext biasCtxt = deserialize_and_load(biasFileName);
                output_ciphertexts[outCh] = context->EvalAdd(result, biasCtxt);
            }
            batched_results.clear();
            mult_results.clear();
        }
    };
    run_parallel_for(outputChannels, worker);
    printTiming("Convolution evaluated...", "conv", start_time);

    batched_rotated_ciphertexts.clear();

    return output_ciphertexts;
}

/**
 * @brief 1x1 (no spatial tap), stride-capable convolution -- used for ResNet's
 * "shortcut"/downsample residual path.
 *
 * Since a 1x1 kernel has no spatial neighbors, this is structurally identical to
 * he_forward_linear_multiple_outputs's per-position linear combination
 * (output[oc] = sum_in(weight[oc][in]*input[in]) + bias[oc], one distinct scalar
 * weight per (in,out) pair, broadcast over the whole vector) -- just evaluated at
 * full input resolution and, when stride>1, followed by the same
 * downsample_with_multiple_channels + bias-after-downsample treatment
 * he_forward_convolution_optimized uses for its strided 3x3 case (matches the
 * already-correct inference-only FHEONANNStandardBatchController::
 * he_batch_convolution_shortcut_optimized's ordering: dense combine -> downsample
 * -> bias).
 *
 * Caches its inputs under FC-style "_in_neuron_" naming (not "_channel_", since
 * backward_pass_convolution_shortcut's dWeight formula is the FC-style
 * sum(gradOutput*input), not a tap-rotation correlation).
 */
vector<Ctext> FHEONTrainForwardPassController::he_forward_convolution_shortcut_optimized(vector<Ctext> &encryptedInputs,
                int batchSize, int inputWidth, int inputChannels, int outputChannels, int stride, string layerName, bool pretrainedFromCsv) {

    int outputWidth = inputWidth / stride;
    int inputSize = inputWidth * inputWidth;
    int outputSize = outputWidth * outputWidth;
    int inVecSize = batchSize * inputSize;
    int outVecSize = batchSize * outputSize;
    vector<double> cleaningoutputVec;
    if (stride > 1) {
        cleaningoutputVec = generate_mixed_mask(outVecSize, inVecSize);
    }

    run_parallel_for(inputChannels, [&](int start, int end) {
        for(int inCh = start; inCh < end; inCh++){
            string inputFileName = layerName+ "_in_neuron_" + to_string(inCh);
            FHEONTrainSupportController::serialize_and_store(encryptedInputs[inCh], inputFileName);
        }
    });

    const vector<double>* csvWeights = nullptr;
    const vector<double>* csvBias = nullptr;
    if (pretrainedFromCsv) {
        csvWeights = &load_pretrained_csv_cached(weights_folder + layerName + "_weight.csv");
        csvBias = &load_pretrained_csv_cached(weights_folder + layerName + "_bias.csv");
    }

    vector<vector<Ctext>> weightCiphertexts(outputChannels, vector<Ctext>(inputChannels));
    vector<Ctext> biasCiphertexts(outputChannels);
    double fan_in = inputChannels;


    auto in_worker = [&](int start, int end) {
        for (int oc = start; oc < end; oc++) {
            for (int inCh = 0; inCh < inputChannels; inCh++) {
                string fileName = layerName + "_out_neuron_" + to_string(oc) + "_in_" + to_string(inCh);
                if (pretrainedFromCsv && !fs::exists(training_data_dir + fileName + ".bin")) continue;
                weightCiphertexts[oc][inCh] = deserialize_and_load(fileName, fan_in, inVecSize);
                serialize_and_store(weightCiphertexts[oc][inCh], fileName);
            }
            string biasFileName = layerName + "_out_neuron_" + to_string(oc) + "_bias";
            if (pretrainedFromCsv && !fs::exists(training_data_dir + biasFileName + ".bin")) continue;
            biasCiphertexts[oc] = deserialize_and_load(biasFileName, 1.0, outVecSize);
            serialize_and_store(biasCiphertexts[oc], biasFileName);
        }
    };
    run_parallel_for(outputChannels, in_worker);

    vector<Ctext> output_ciphertexts(outputChannels);
    auto start_time = startTime();
    auto worker = [&](int start, int end) {
        for (int oc = start; oc < end; oc++) {
            vector<Ctext> temp_results(inputChannels);
            for (int inCh = 0; inCh < inputChannels; inCh++) {
                string fileName = layerName + "_out_neuron_" + to_string(oc) + "_in_" + to_string(inCh);
                if (pretrainedFromCsv && !fs::exists(training_data_dir + fileName + ".bin")) {
                    int idx = oc * inputChannels + inCh;
                    double val = csv_scalar_or_default(*csvWeights, idx, 0.0);
                    if (std::abs(val) < 1e-10) val = 0.0;
                    import_scalar_param(val, fileName, inVecSize);
                    vector<double> broadcastVec(inVecSize, val);
                    Ptext weightPtxt = context->MakeCKKSPackedPlaintext(broadcastVec, 1, encryptedInputs[inCh]->GetLevel());
                    temp_results[inCh] = context->EvalMult(encryptedInputs[inCh], weightPtxt);
                } else {
                    temp_results[inCh] = context->EvalMult(encryptedInputs[inCh], weightCiphertexts[oc][inCh]);
                }
            }
            Ctext result = context->EvalAddMany(temp_results);

            temp_results.clear();

            if (stride > 1) {
                result = FHEONANNController::downsample_with_multiple_channels(result, inputWidth, stride, batchSize);
                result = context->EvalMult(result, context->MakeCKKSPackedPlaintext(cleaningoutputVec, 1, result->GetLevel()));
            }

            string biasFileName = layerName + "_out_neuron_" + to_string(oc) + "_bias";
            if (pretrainedFromCsv && !fs::exists(training_data_dir + biasFileName + ".bin")) {
                double val = csv_scalar_or_default(*csvBias, oc, 0.0);
                if (std::abs(val) < 1e-10) val = 0.0;
                import_scalar_param(val, biasFileName, outVecSize);
                vector<double> broadcastVec(outVecSize, val);
                Ptext biasPtxt = context->MakeCKKSPackedPlaintext(broadcastVec, 1, result->GetLevel());
                output_ciphertexts[oc] = context->EvalAdd(result, biasPtxt);
            } else {
                output_ciphertexts[oc] = context->EvalAdd(result, biasCiphertexts[oc]);
            }
        }
    };
    run_parallel_for(outputChannels, worker);
    printTiming("Convolution shortcut evaluated...", "shortcut", start_time);

    weightCiphertexts.clear();
    biasCiphertexts.clear();

    return output_ciphertexts;
}

/**
 * @brief Global average pool: collapses each channel's whole (batchSize x
 * dataWidth x dataWidth) feature map down to one scalar per image, packed
 * compactly (contiguous, one slot per image) -- matches
 * FHEONANNStandardBatchController::he_batch_globalpool's construction (rotate the
 * whole channel by spatialSize to bring each image's block to the front,
 * EvalSum that block, EvalMerge the per-image sums into a compact vector),
 * simplified since this codebase's batchSize (16) never needs
 * he_batch_globalpool's chunked-merge optimization (that only kicks in once
 * batchSize exceeds its rotatePositions parameter, e.g. 32 there).
 *
 * Caches inputs under plain "_channel_" naming (matches backward_pass_
 * global_avgpool's expectations, and every other spatial-map forward
 * function's convention).
 */
vector<Ctext> FHEONTrainForwardPassController::he_forward_global_avgpool(vector<Ctext>& encryptedInputs, int batchSize, int dataWidth, int inputChannels, string layerName) {

    int spatialSize = dataWidth * dataWidth;
    vector<double> reducingMask = generate_scale_mask(spatialSize, batchSize);

    run_parallel_for(inputChannels, [&](int start, int end) {
        for(int inCh = start; inCh < end; inCh++){
            string inputFileName = layerName+ "_channel_" + to_string(inCh);
            FHEONTrainSupportController::serialize_and_store(encryptedInputs[inCh], inputFileName);
        }
    });

    vector<Ctext> pooled_ciphertexts(inputChannels);

    auto start_time = startTime();
    auto worker = [&](int start, int end) {
        for (int ch = start; ch < end; ch++) {
            vector<Ctext> result_ciphers(batchSize);
            Ctext encryptedInput = encryptedInputs[ch]->Clone();
            for (int b = 0; b < batchSize; b++) {
                if (b != 0) {
                    encryptedInput = context->EvalRotate(encryptedInput, spatialSize);
                }
                result_ciphers[b] = context->EvalSum(encryptedInput, spatialSize);
            }
            Ctext merged = context->EvalMerge(result_ciphers);
            pooled_ciphertexts[ch] = context->EvalMult(merged, context->MakeCKKSPackedPlaintext(reducingMask, 1, merged->GetLevel()));
        }
    };
    run_parallel_for(inputChannels, worker);

    printTiming("Global pooling Optimized...", "pooling", start_time);

    return pooled_ciphertexts;
}




/**
 * @brief Apply a homomorphic ReLU activation using Chebyshev polynomial approximation.
 *
 * This function evaluates a ReLU activation on encrypted inputs using a
 * Chebyshev polynomial approximation over a bounded interval. Each input
 * channel is optionally rescaled using a plaintext mask to control the output
 * magnitude before applying the polynomial.
 *
 * The ReLU function is approximated as:
 *   f(x) = 0              if x < 0
 *   f(x) = scaleVal * x   otherwise
 *
 * where the approximation is evaluated over the interval [lowerBound, upperBound].
 * The computation is parallelized across input channels.
 *
 * @param encryptedInputs  Vector of ciphertexts representing encrypted inputs,
 *                         sized [inputChannels].
 * @param scaleValues      Per-channel scaling factors applied before ReLU.
 * @param inputChannels    Number of input channels.
 * @param vectorSize       Number of valid slots per ciphertext.
 * @param polyDegree       Degree of the Chebyshev polynomial approximation.
 *
 * @param layerName Prefix of the cached inputs (layerName_channel_{i}), used by backward_pass_relu.
 * @return A vector of ciphertexts where each element contains the encrypted
 *         ReLU-activated output for the corresponding input channel.
 */

vector<Ctext> FHEONTrainForwardPassController::he_forward_relu(vector<Ctext>& encryptedInputs, vector<int> scaleValues,  int inputChannels, int vectorSize, int polyDegree, string layerName) {
    double lowerBound = -1;
    double upperBound = 1;
    
    auto encryptInn = encryptedInputs;
    // int scaleVal = 110;
    vector<Ctext> relu_results(inputChannels);
    int totalElements =  nextPowerOf2(vectorSize);

    run_parallel_for(inputChannels, [&](int start, int end) {
        for(int inCh = start; inCh < end; inCh++){
            string inputFileName = layerName+ "_channel_" + to_string(inCh);
            FHEONTrainSupportController::serialize_and_store(encryptedInputs[inCh], inputFileName);
        }
    });

    // Lambda for per-thread work

    auto start_time = startTime();
    auto worker = [&](int start, int end) {
        for(int i=start; i<end; i++){
            int scaleVal = scaleValues[i];
            if(scaleVal <= 1){
                encryptInn[i] = encryptedInputs[i];
                scaleVal = 1;
                relu_results[i] = context->EvalChebyshevFunction(
                                                [](double x) -> double { return (x < 0) ? 0 : x; },
                                                encryptInn[i], lowerBound, upperBound, polyDegree);
            }
            else{
                // Normalize input into [-1, 1] by multiplying by 1/(2*scaleVal)
                double normFactor = 1.0 / (2.0 * scaleVal);
                auto mask_data = context->MakeCKKSPackedPlaintext(generate_scale_mask(1, vectorSize), 1, encryptedInputs[i]->GetLevel(), nullptr, totalElements);
                encryptInn[i] = context->EvalMult(encryptedInputs[i], normFactor);
                relu_results[i] = context->EvalChebyshevFunction(
                                            [](double x) -> double { return (x < 0) ? 0 : x; },
                                            encryptInn[i], lowerBound, upperBound, polyDegree
                );
                relu_results[i] = context->EvalMult(relu_results[i], (double)(2 * scaleVal));
            }
        }
    };

    run_parallel_for(inputChannels, worker);
    printTiming("ReLU evaluated...", "relue", start_time);

    return relu_results;
}


/**
 * @brief Perform optimized secure average pooling on batched encrypted feature maps.
 *
 * This function computes average pooling across multiple input channels 
 * for batched data under homomorphic encryption. For each channel, it 
 * partitions the input feature map into local regions of size 
 * (kernelWidth × kernelWidth), sums the values in each region, 
 * and applies scaling to obtain the average. The stride length determines 
 * how pooling regions are traversed across the feature map. 
 *
 * The operation is applied independently to each input channel. Within each 
 * channel, ciphertext rotations and additions are used efficiently to align 
 * and sum pooling regions across batches, minimizing the number of 
 * homomorphic operations. The output is one ciphertext per channel, where 
 * each ciphertext encodes the pooled results for all batches.
 *
 * @param encryptedInputs   Vector of ciphertexts, one per input channel, 
 *                          each encoding batched feature maps of size 
 *                          (batchSize × inputWidth × inputWidth).
 * @param batchSize         Number of batches encoded in the ciphertexts.
 * @param inputWidth        Width (and height) of the input feature maps (assumed square).
 * @param kernelWidth       Width (and height) of the pooling kernel (assumed square).
 * @param stridingLen       Stride length used to slide the pooling window 
 *                          across the input feature maps.
 *
 * @param layerName Prefix of the cached inputs (layerName_channel_{i}).
 * @return vector<Ctext>    Vector of ciphertexts where each element corresponds 
 *                          to one pooled channel across all batches. Each output 
 *                          ciphertext encodes multiple pooled values per batch, 
 *                          arranged as:
 *                          [batch0 pooled values][batch1 pooled values]...
 *
 * @note 
 * - This implementation is optimized for FHE-based CNN/ANN architectures where 
 *   reducing the number of ciphertext rotations and multiplications is critical 
 *   for efficiency. 
 * - The function assumes ciphertext packing follows contiguous layout: 
 *   [batch0: channel_i feature map][batch1: channel_i feature map]...
 */
vector<Ctext> FHEONTrainForwardPassController::he_forward_avgPool_optimized(vector<Ctext>& encryptedInputs,  int batchSize, int inputWidth, int kernelWidth, int stridingLen, string layerName){

    int kernelSize = pow(kernelWidth, 2);
    int inputSize = pow(inputWidth, 2);
    int outputWidth = inputWidth / stridingLen;
    int outputSize = outputWidth * outputWidth;
    int inVecSize = batchSize * inputSize;
    int outVecSize = batchSize * outputSize; 
    int inputChannels = encryptedInputs.size();

    int encode_level = encryptedInputs[0]->GetLevel();
    Ptext reducingMask =  context->MakeCKKSPackedPlaintext(generate_scale_mask(kernelSize, inVecSize), 1, encode_level);
    Ptext cleaningInMask = context->MakeCKKSPackedPlaintext(generate_mixed_mask(inputSize, inVecSize), 1, encode_level);
    vector<double> cleaningoutputVec = generate_mixed_mask(outVecSize, inVecSize);

    /*** save alll inputs */
    run_parallel_for(inputChannels, [&](int start, int end) {
        for(int inCh = start; inCh < end; inCh++){
            string inputFileName = layerName+ "_channel_" + to_string(inCh);
            FHEONTrainSupportController::serialize_and_store(encryptedInputs[inCh], inputFileName);
        }
    });

    /*** STEP 1 - ROTATE THE CIPHERTEXT into by k^2-1 and create a k^2 rotated right positions ***/
    vector<Ctext> batched_ciphertexts(inputChannels);

    auto start_time = startTime();
    auto in_worker = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            vector<Ctext> rotated_ciphertexts;
            Ctext encryptedInput = encryptedInputs[i];
            auto digits = context->EvalFastRotationPrecompute(encryptedInput);
            rotated_ciphertexts.push_back(encryptedInput);
            rotated_ciphertexts.push_back(context->EvalFastRotation(encryptedInput, 1, context->GetCyclotomicOrder(), digits));
            rotated_ciphertexts.push_back(context->EvalFastRotation(encryptedInput, inputWidth, context->GetCyclotomicOrder(), digits));
            rotated_ciphertexts.push_back(context->EvalRotate(context->EvalFastRotation(encryptedInput, inputWidth, context->GetCyclotomicOrder(), digits), 1));
            Ctext sum_cipher = context->EvalAddMany(rotated_ciphertexts);

            /*** STEP 3: Multiply the scale value with the sum cipher */
            batched_ciphertexts[i] = context->EvalMult(sum_cipher, reducingMask);
        }
    };
    run_parallel_for(inputChannels, in_worker);

    /**** Caryout the average pooling if we have less than 4 elements per channel elements*/
    vector<Ctext> pooled_ciphertexts(inputChannels);
    if(inputWidth <= 2){
        auto worker = [&](int start, int end) {
            for (int i = start; i < end; i++) {
                Ctext inn_ciphertext = batched_ciphertexts[i];
                vector<Ctext> inn_ciphers;
                for(int i = 0; i<batchSize; i++){
                    inn_ciphers.push_back(context->EvalSum(context->EvalMult(inn_ciphertext, cleaningInMask), inputSize));
                    inn_ciphertext = context->EvalRotate(inn_ciphertext, inputSize);
                }
                pooled_ciphertexts[i] = context->EvalMerge(inn_ciphers);
            }
        };
        run_parallel_for(inputChannels, worker);

    }
    else{
        auto worker = [&](int start, int end) {
            for (int i = start; i < end; i++) {
                Ctext result = FHEONANNController::downsample_with_multiple_channels(batched_ciphertexts[i], inputWidth, stridingLen, batchSize);
                pooled_ciphertexts[i] = context->EvalMult(result, context->MakeCKKSPackedPlaintext(cleaningoutputVec, 1, result->GetLevel()));
            }
        };
        run_parallel_for(inputChannels, worker);
    }
    printTiming("Average pooling Optimized...", "pooling", start_time);

    return pooled_ciphertexts;
}


/**
 * @brief Fully connected layer of the training forward pass, one ciphertext per neuron.
 *
 * output[ou] = sum_in weight[ou][in] * input[in] + bias[ou], with one distinct encrypted weight per (output, input)
 * pair broadcast over the vectorSize slots (the images of the batch). The inputs are cached for the backward pass.
 *
 * Parameter files: layerName_out_neuron_{ou}_in_{in} and layerName_out_neuron_{ou}_bias. Cached inputs:
 * layerName_in_neuron_{in}.
 *
 * @param encryptedInputs   One ciphertext per input neuron (one value per image in the first vectorSize slots).
 * @param vectorSize        Number of slots of every neuron (the batch size after the global average pooling).
 * @param inputSize         Number of input neurons.
 * @param outputSize        Number of output neurons.
 * @param layerName         Prefix of the parameter and cache files.
 * @param pretrainedFromCsv If true, missing parameters are read from weights_folder + layerName + "_weight.csv" /
 *                          "_bias.csv" (flat [out][in], nn.Linear layout) as plaintexts, and stored as ciphertexts.
 *
 * @return One ciphertext per output neuron.
 */
vector<Ctext> FHEONTrainForwardPassController::he_forward_linear_multiple_outputs(vector<Ctext>& encryptedInputs,
                                                int vectorSize, int inputSize, int outputSize, string layerName, bool pretrainedFromCsv){

    for(int inCh =0; inCh<inputSize; inCh++){
        string inputFileName = layerName+ "_in_neuron_" + to_string(inCh);
        FHEONTrainSupportController::serialize_and_store(encryptedInputs[inCh], inputFileName);
    }

    const vector<double>* csvWeights = nullptr;
    const vector<double>* csvBias = nullptr;
    if (pretrainedFromCsv) {
        csvWeights = &load_pretrained_csv_cached(weights_folder + layerName + "_weight.csv");
        csvBias = &load_pretrained_csv_cached(weights_folder + layerName + "_bias.csv");
    }


    vector<vector<Ctext>> weightCiphertexts(outputSize, vector<Ctext>(inputSize));
    vector<Ctext> biasCiphertexts(outputSize);
    double fan_in = inputSize;

    auto worker = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            for (int inCh = 0; inCh < inputSize; inCh++) {
                string fileName = layerName + "_out_neuron_" + to_string(i) + "_in_" + to_string(inCh);
                if (pretrainedFromCsv && !fs::exists(training_data_dir + fileName + ".bin")) {
                    continue;
                }
                weightCiphertexts[i][inCh] = deserialize_and_load(fileName, fan_in, vectorSize);
                serialize_and_store(weightCiphertexts[i][inCh], fileName);
            }
            string biasFileName = layerName + "_out_neuron_" + to_string(i) + "_bias";
            if (pretrainedFromCsv && !fs::exists(training_data_dir + biasFileName + ".bin")) {
                continue;
            }
            biasCiphertexts[i] = deserialize_and_load(biasFileName, 1.0, vectorSize);
            serialize_and_store(biasCiphertexts[i], biasFileName);
        }
    };
    run_parallel_for(outputSize, worker);

    // take every input and multiply by its own weight
    vector<Ctext> results(outputSize);

    auto start_time = startTime();
    auto in_worker = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            vector<Ctext> temp_results(inputSize);
            for(int inCh = 0; inCh < inputSize; inCh++){
                string fileName = layerName + "_out_neuron_" + to_string(i) + "_in_" + to_string(inCh);
                if (pretrainedFromCsv && !fs::exists(training_data_dir + fileName + ".bin")) {
                    int idx = i * inputSize + inCh;
                    double val = csv_scalar_or_default(*csvWeights, idx, 0.0);
                    if (std::abs(val) < 1e-10) val = 0.0;
                    import_scalar_param(val, fileName, vectorSize);
                    vector<double> broadcastVec(vectorSize, val);
                    Ptext weightPtxt = context->MakeCKKSPackedPlaintext(broadcastVec, 1, encryptedInputs[inCh]->GetLevel());
                    temp_results[inCh] = context->EvalMult(encryptedInputs[inCh], weightPtxt);
                } else {
                    temp_results[inCh] = context->EvalMult(encryptedInputs[inCh], weightCiphertexts[i][inCh]);
                }
            }
            Ctext temp = context->EvalAddMany(temp_results);

            string biasFileName = layerName + "_out_neuron_" + to_string(i) + "_bias";
            if (pretrainedFromCsv && !fs::exists(training_data_dir + biasFileName + ".bin")) {
                double val = csv_scalar_or_default(*csvBias, i, 0.0);
                if (std::abs(val) < 1e-10) val = 0.0;
                import_scalar_param(val, biasFileName, vectorSize);
                vector<double> broadcastVec(vectorSize, val);
                Ptext biasPtxt = context->MakeCKKSPackedPlaintext(broadcastVec, 1, temp->GetLevel());
                temp = context->EvalAdd(temp, biasPtxt);
            } else {
                temp = context->EvalAdd(temp, biasCiphertexts[i]);
            }
            results[i] = temp;
        }
    };
    run_parallel_for(outputSize, in_worker);
    printTiming("Linear multiple outputs...", "linear", start_time);

    return results;
}

/**
 * @brief Computing approximate softmax using a Taylor-approximated exponential
 * and a Newton-Raphson-approximated reciprocal.
 *
 * We use the standard L1 softmax formulation: exp(x_i) / sum(exp(x_j)), with
 * exp(x) approximated by its degree-2 Taylor polynomial (1 + x + x^2/2).
 *
 * The sum of exponentials is strictly positive but only loosely bounded
 * (its true per-slot value can range anywhere up to the calibration-based
 * worst-case upper bound `sumVal`, derived from `scalingValues`), so its
 * reciprocal is computed via Newton-Raphson on 1/sqrt(x) (then squared to
 * get 1/x) rather than a Chebyshev polynomial fit of 1/x directly: a
 * Chebyshev fit needs a fixed, tight interval, and any interval wide enough
 * to safely contain 1/x's domain here sits at or straddles x=0 — 1/x's pole
 * — which blows up the fitted polynomial's coefficients and corrupts
 * decoding (this is what used to crash with "the approximation error is too
 * high"). Newton-Raphson only needs a same-order-of-magnitude starting
 * guess and converges quadratically, so a single scalar init derived from
 * sumVal/2 is enough for the whole batch. Same pattern already used for
 * BatchNorm's inverse-std (see newtonraphson_approximate_inverse_sqrt).
 *
 * @param encryptedInputs Vector of ciphertexts representing encrypted logits.
 * @param inputSize       Size of the input (not used directly, but keeps signature aligned).
 * @param appxPolyDegree  Unused now (kept for call-site compatibility) — the
 *                        reciprocal step uses a fixed, small-iteration-count
 *                        Newton-Raphson instead of a polynomial degree.
 * @param scalingValues Per-class bound on |logit| (e.g. read_batch_scaling_values): sets the worst-case sum of the exponentials, hence the start of the Newton reciprocal.
 * @return Vector of ciphertexts containing softmax probabilities.
 */
vector<Ctext> FHEONTrainForwardPassController::he_forward_softmax(vector<Ctext> &encryptedInputs, const vector<int>& scalingValues, int inputSize, int appxPolyDegree) {

    (void)appxPolyDegree;

    /** Compute the exponentials using a polynomial approximation of exp(x) = 1 + X + (X^2/2!) */

    int softmaxInputSize = encryptedInputs.size();
    vector<Ctext> expInputs(encryptedInputs.size());
    vector<double> scalevals(encryptedInputs.size());

    auto start_time = startTime();
    auto in_worker = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            expInputs[i] = context->EvalSquare(encryptedInputs[i]);
            expInputs[i] = context->EvalMult(expInputs[i], 0.5);
            expInputs[i] = context->EvalAdd(expInputs[i], encryptedInputs[i]);
            expInputs[i] = context->EvalAdd(expInputs[i], 1.0);
            scalevals[i] = ((pow(scalingValues[i], 2.0) / 2.0) + (double)scalingValues[i] + 1.0);
        }
    };
    run_parallel_for(softmaxInputSize, in_worker);
    printTiming("Softmax evaluated...", "Softmax", start_time);

    Ctext sumedExp = context->EvalAddMany(expInputs);
    double sumVal = 0.0;
    for (double val : scalevals) {
        sumVal += val;
    }
    cout << "Worst-case upper bound on sum of exponentials: " << sumVal << endl;

    sumedExp = context->EvalBootstrap(sumedExp, 2);
    double expectedSum = sumVal / 2.0;
    double initValue = 1.0 / sqrt(expectedSum);
    Ctext invSqrt = newtonraphson_approximate_inverse_sqrt(sumedExp, num_slots, initValue, 4, true);

    start_time = startTime();
    Ctext actualDivideResult = context->EvalSquare(invSqrt);  // 1/sqrt(x)^2 = 1/x
    vector<Ctext> softMaxCtxts(softmaxInputSize);
    auto worker = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            softMaxCtxts[i] = context->EvalMult(expInputs[i], actualDivideResult);
        }
    };
    run_parallel_for(softmaxInputSize, worker);
    printTiming("Softmax evaluated...", "Softmax", start_time);

    return softMaxCtxts;
}

vector<Ctext> FHEONTrainForwardPassController::he_forward_batch_norm(vector<Ctext> &encryptedInputs, int inputSize, double appxInitValue, int appxDegree, string layerName, bool isBootstrap, bool pretrainedFromCsv) {

    double scaler = (1.0 / inputSize);
    auto cleaningMaskVec = generate_scale_and_clean_mask(scaler, inputSize);
    size_t numChannels = encryptedInputs.size();
    vector<Ctext> normResults(numChannels);
    vector<Ctext> topVecs(numChannels);
    vector<Ctext> varVals(numChannels);

    const vector<double>* csvGamma = nullptr;
    const vector<double>* csvBeta = nullptr;
    if (pretrainedFromCsv) {
        csvGamma = &load_pretrained_csv_cached(weights_folder + layerName + "_gamma.csv");
        csvBeta = &load_pretrained_csv_cached(weights_folder + layerName + "_beta.csv");
    }

    // Phase 1: mean-centered value and variance (+epsilon) for every channel.
    auto start_time = startTime();
    auto in_worker = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            uint32_t currentLevel = encryptedInputs[i]->GetLevel();
            auto cleaningMaskPtxt = context->MakeCKKSPackedPlaintext(cleaningMaskVec, 1, currentLevel, nullptr, inputSize);

            /** Compute the mean */
            Ctext sumVal = context->EvalSum(encryptedInputs[i], inputSize);
            Ctext meanVal = context->EvalMult(sumVal, cleaningMaskPtxt);
            meanVal = repeat_encrypted_values(meanVal, 1, inputSize);

            /**** Compute the variance + epsilon for numerical stability */
            Ctext topVec = context->EvalSub(encryptedInputs[i], meanVal);
            Ctext topVecSquare = context->EvalSquare(topVec);
            Ctext varValSum = context->EvalSum(topVecSquare, inputSize);
            auto cleaningMaskPtxt2 = context->MakeCKKSPackedPlaintext(cleaningMaskVec, 1, varValSum->GetLevel(), nullptr, inputSize);
            Ctext varVal = context->EvalMult(varValSum, cleaningMaskPtxt2);
            varVal = repeat_encrypted_values(varVal, 1, inputSize);

            topVecs[i] = topVec;
            varVals[i] = varVal;
        }
    };
    run_parallel_for(numChannels, in_worker);
    measureTiming("Batch norm evaluating...", "batch_norm", start_time);

    
    auto worker = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            if (isBootstrap){
                varVals[i] = context->EvalBootstrap(varVals[i], 2);
            }

            double initVal = (appxInitValue > 0.0) ? appxInitValue : 0.005;
            Ctext approxStd = newtonraphson_approximate_inverse_sqrt(varVals[i], inputSize, initVal, appxDegree, isBootstrap);

            Ctext xHat = context->EvalMult(topVecs[i], approxStd);
            string xHatFileName = "batch_norm_xhat_" + layerName + "_channel_" + to_string(i);
            serialize_and_store(xHat, xHatFileName);

            string fileName = "batch_norm_" + layerName + "_channel_" + to_string(i);
            string biasFileName = "batch_norm_bias_" + layerName + "_channel_" + to_string(i);

            Ctext learnedandinvstd;

            auto phase_two_start_time = startTime();
            if (pretrainedFromCsv && !fs::exists(training_data_dir + fileName + ".bin")) {
                double val = csv_scalar_or_default(*csvGamma, i, 1.0);
                if (std::abs(val) < 1e-10) val = 0.0;
                import_scalar_param(val, fileName, inputSize);
                vector<double> gammaVec(inputSize, val);
                Ptext gammaPtxt = context->MakeCKKSPackedPlaintext(gammaVec, 1, approxStd->GetLevel());
                learnedandinvstd = context->EvalMult(approxStd, gammaPtxt);
            } else {
                Ctext learnedCtxt = deserialize_and_load(fileName, 1.0, inputSize);
                learnedandinvstd = context->EvalMult(learnedCtxt, approxStd);
            }
            Ctext scaledXhat = context->EvalMult(learnedandinvstd, topVecs[i]);
            measureTiming("Batch norm evaluating...", "batch_norm", phase_two_start_time);

            if (pretrainedFromCsv && !fs::exists(training_data_dir + biasFileName + ".bin")) {
                double val = csv_scalar_or_default(*csvBeta, i, 0.0);
                if (std::abs(val) < 1e-10) val = 0.0;
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
    };
    run_parallel_for(numChannels, worker);
    

    return normResults;
}

Ctext FHEONTrainForwardPassController::newtonraphson_approximate_inverse_sqrt(Ctext& encryptedInput, int inputSize, double initValue, int appxDegree, bool isBootstrap) {
    vector<double> initVec(inputSize, initValue);
    Ptext initPlain = context->MakeCKKSPackedPlaintext(initVec, 1, encryptedInput->GetLevel());
    Ctext y = context->EvalAdd(initPlain, encryptedZeros);

    const int itersPerBootstrap = 2;
    for (int n = 0; n < appxDegree; n++) {
        if ((n > 0 && n % itersPerBootstrap == 0) && isBootstrap) {
            // encryptedInput = context->EvalBootstrap(encryptedInput, 2);
            y = context->EvalBootstrap(y, 2);
        }
        auto y2 = context->EvalSquare(y);
        auto g  = context->EvalMult(encryptedInput, y2);
        auto t = context->EvalSub(1.5, context->EvalMult(0.5, g));
        y = context->EvalMult(y, t);
    }
    return y;
}

Ctext FHEONTrainForwardPassController::repeat_encrypted_values(Ctext& encryptedInput, int vecSize, int batchSize){
    batchSize = nextPowerOf2(batchSize);
    Ctext tempIn = context->EvalAdd(encryptedInput, context->EvalRotate(encryptedInput, -vecSize));

    for(int j=1; j< log2(batchSize); j++){
        int powVal = pow(2, j);
        tempIn = context->EvalAdd(tempIn, context->EvalRotate(tempIn, -powVal));
    }
    return tempIn;
}