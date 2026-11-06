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

#include <fstream>
#include <filesystem>
#include <iostream>
#include <cmath>
// #include <thread>
#include "FHEONTrainBackwardPassController.h"
#include "UtilsTraining.h"

namespace fs = std::filesystem;


static inline double csv_scalar_or_default_backward(const vector<double>& values, size_t idx, double fallback) {
    if (idx < values.size()) {
        return values[idx];
    }
    return fallback;
}

/** Displays a welcome message for the FHEON Training Controller.*/
void FHEONTrainBackwardPassController::welcomeToTrainingMessage(){
    cout << endl << "***********************************************************************************************************************" << endl;
    cout << endl <<  "****************************** WELCOME TO FHEON TRAINING CONTROLLER! ******************************" << endl;
    cout << "Welcome to FHEON Training Controller! This controller is designed to manage and orchestrate the training process "
            "of homomorphic encrypted neural networks. It provides functionalities for setting up the training environment, "
            "managing encrypted data, and coordinating the training workflow. "
            "Let's get started on your secure and private machine learning journey!" << endl;
    cout << endl << "***********************************************************************************************************************"  << endl << endl;
}

/***
 * Computing approximate softmax using Chebyshev polynomial approximation
 *  @param encryptedInput: The encrypted input vector for which the softmax is to be computed.
 *  @return: The approximate softmax of the encrypted input vector.
 */
vector<Ctext> FHEONTrainBackwardPassController::softmax(vector<Ctext> &encryptedInputs, int inputSize, int appxPolyDegree) {
    auto start_time = startTime();
    double lowerBound = -20.0;
    double upperBound = 20.0;

    vector<Ctext> expInputs(encryptedInputs.size());
        run_parallel_for(encryptedInputs.size(), [&](int start, int end) {
            for(int i=start; i<end; i++){
                expInputs[i] = context->EvalSquare(encryptedInputs[i]);
                expInputs[i] = context->EvalMult(expInputs[i], 0.5);
                expInputs[i] = context->EvalAdd(expInputs[i], encryptedInputs[i]);
                expInputs[i] = context->EvalAdd(expInputs[i], 1.0);
            }
        });
    Ctext sumedExp = context->EvalAddMany(expInputs);
    Ctext divideResult = context->EvalChebyshevFunction(
            [](double x) -> double { return 1.0 / x; },
            sumedExp, lowerBound, upperBound, appxPolyDegree);

    vector<Ctext> softMaxCtxts(encryptedInputs.size());
    run_parallel_for(encryptedInputs.size(), [&](int start, int end) {
        for(int i=start; i<end; i++){
            softMaxCtxts[i] = context->EvalMult(expInputs[i], divideResult);
        }
    });

    printTiming("Backward softmax evaluated...", "backward_softmax", start_time);
    return softMaxCtxts;
}

Ctext FHEONTrainBackwardPassController::cross_entropy_loss(vector<Ctext> &encryptedInputs, vector<double> &labels, int batchSize, int numClasses, int appxDegree) {
    
    int numInputs = (int)encryptedInputs.size();
    vector<Ctext> innLossCtxts(numInputs);
    vector<Ptext> labelsPxt(numInputs);

    auto start_time = startTime();
    run_parallel_for(numInputs, [&](int start, int end) {
        for(int i=start; i<end; i++){
            vector<double> labelVec(batchSize, 0.0);
            for(int j=0; j<batchSize; j++){
                if((int)labels[j] == i){
                    labelVec[j] = 1.0;
                }
            }
            labelsPxt[i] = context->MakeCKKSPackedPlaintext(labelVec, 1, 1, nullptr, batchSize);

            Ctext approxLog = taylors_approximation(encryptedInputs[i], appxDegree);
            innLossCtxts[i] = context->EvalMult(approxLog, labelsPxt[i]);
        }
    });

    /** sum all the losses and approximate their log */
    Ctext innLossCtxt = context->EvalAddMany(innLossCtxts);

    /** compute the negative log-likelihood and take the average */
    double scaler = (-1.0)/batchSize;
    //create a cleaning mask and scaling mask to get the average loss
    auto scaleAndCleanMaskVec = generate_scale_and_clean_mask(scaler, batchSize);
    auto scaleAndCleanMaskPtxt = context->MakeCKKSPackedPlaintext(scaleAndCleanMaskVec, 1, innLossCtxt->GetLevel(), nullptr, batchSize);
    Ctext avgLoss = context->EvalMult(context->EvalSum(innLossCtxt, batchSize), scaleAndCleanMaskPtxt);

    printTiming("Cross entropy and softmax backward evaluating...", "cross_entrophy", start_time);
    return avgLoss;
}


/***
 * Backpropagation through cross-entropy loss and softmax activation.
 * This function computes the gradient of the loss with respect to the inputs
 * by leveraging the property that the gradient of cross-entropy loss with softmax
 * simplifies to (predictions - labels) / batchSize.
 *
 * @param encryptedInputs  Vector of ciphertexts containing encrypted predictions (logits).
 * @param labels           Vector of plaintext integers representing the true class indices.
 * @param batchSize        Number of elements in the batch (N).
 * @param vectorSize       Number of valid slots in each channel ciphertext (batchSize * spatialSize).
 *                         Only the first `batchSize` of these represent a real per-image
 *                         prediction (see he_forward_linear_multiple_outputs / cross_entropy_loss's
 *                         own doc comments — the FC layers broadcast identically across every
 *                         spatial position, so the remaining vectorSize-batchSize slots per
 *                         channel are unused broadcast copies, never part of the loss). Every
 *                         other backward_pass_* function reduces (EvalSum) over the full
 *                         vectorSize width, so a gradient left nonzero out there would leak into
 *                         every downstream weight/input gradient instead of being ignored like
 *                         the forward loss (which masks down to the same first-batchSize window
 *                         via generate_scale_and_clean_mask+EvalSum) already does.
 * @param layerName        Name of the layer for file naming purposes.
 *
 * @return A vector of ciphertexts containing the gradients for each input class.
 */
vector<Ctext> FHEONTrainBackwardPassController::backward_pass_cross_entropy_and_softmax(vector<Ctext> &encryptedInputs, vector<double> &labels, int batchSize, int vectorSize, string layerName) {

    int numInputs = (int)encryptedInputs.size();
    vector<Ptext> labelsPxt(numInputs);
    run_parallel_for(numInputs, [&](int start, int end) {
        for(int i=start; i<end; i++){
            vector<double> labelVec(batchSize, 0.0);
            for(int j=0; j<batchSize; j++){
                if((int)labels[j] == i){
                    labelVec[j] = 1.0;
                }
            }
            labelsPxt[i] = context->MakeCKKSPackedPlaintext(labelVec, 1, encryptedInputs[i]->GetLevel(), nullptr, batchSize);
        }
    });

    double scale = (1.0 / batchSize);
    vector<double> cleanMaskVec(vectorSize, 0.0);
    for (int j = 0; j < batchSize && j < vectorSize; j++) {
        cleanMaskVec[j] = scale;
    }

    vector<Ctext> initGrad(numInputs);
    auto start_time = startTime();
    run_parallel_for(numInputs, [&](int start, int end) {
        for(int i=start; i<end; i++){
            auto cleanMaskPtxt = context->MakeCKKSPackedPlaintext(cleanMaskVec, 1, encryptedInputs[i]->GetLevel(), nullptr, vectorSize);
            initGrad[i] = context->EvalSub(encryptedInputs[i], labelsPxt[i]);
            initGrad[i] = context->EvalMult(initGrad[i], cleanMaskPtxt);

            string gradFileName = "grad_" + layerName + "_channel_" + to_string(i);
            serialize_and_store(initGrad[i], gradFileName);
        }
    });
    printTiming("Cross entropy and softmax backward evaluating...", "Softmax", start_time);

    return initGrad;
}

/***
 * Backpropagation through ReLU activation.
 * This function computes the gradient of the loss with respect to the inputs
 * by leveraging the property that the gradient of ReLU is 1 for positive values and 0 for negative values.
 *
 * @param encryptedInputs  Vector of ciphertexts containing encrypted inputs.
 * @param scaler           Vector of integers representing the scale factors.
 * @param batchSize        Number of elements in the batch (N).
 * @param appxPolyDegree   Degree of the Chebyshev polynomial approximation.
 *
 * @return A vector of ciphertexts containing the gradients for each input.
 */
vector<Ctext> FHEONTrainBackwardPassController::backward_pass_relu(vector<Ctext> &gradInputs, vector<int> &scalers, 
                                                int appxPolyDegree, string layerName) {

    double lowerBound = -1;
    double upperBound = 1;
    int numChannels = (int)gradInputs.size();
    vector<Ctext> reluGradients(numChannels);

    auto start_time = startTime();
    run_parallel_for(numChannels, [&](int start, int end) {
        for(int i=start; i<end; i++){
            string inputFileName =  layerName+ "_channel_" + to_string(i);
            Ctext inputCtxt = deserialize_and_load(inputFileName);
            if(scalers[i] > 1){
                double scaleFactor = 1.0/(2*scalers[i]);
                inputCtxt = context->EvalMult(inputCtxt, scaleFactor);
            }

            Ctext reluDerivative = context->EvalChebyshevFunction(
                                                [](double x) -> double {
                                                        return (x < 0) ? 0.0 : 1.0;},
                                                        inputCtxt,
                                                        lowerBound,
                                                        upperBound,
                                                        appxPolyDegree);

            reluGradients[i] = context->EvalMult(gradInputs[i], reluDerivative);

            string gradFileName = "grad_" + layerName + "_channel_" + to_string(i);
            serialize_and_store(reluGradients[i], gradFileName);
        }
    });
    printTiming("ReLU backward evaluating...", "relu", start_time);
    return reluGradients;
}

/**
 * Backpropagation through batch normalization.
 * This function computes the gradient of the loss with respect to the inputs
 * by performing the necessary computations for batch normalization backpropagation.
 *
 * @param gradInputs          Vector of ciphertexts representing the gradients.
 * @param inputSize           Size of the input vector.
 * @param learningRate        Learning rate for updating parameters.
 *
 * @param encryptedInputs Unused: the normalized inputs (xhat) and gamma / std are read from the caches of he_forward_batch_norm.
 * @param layerName Prefix of the batch norm parameter (gamma, beta) and cache files, as in he_forward_batch_norm.
 * @return A vector of ciphertexts containing the gradients for each input.
 */
vector<Ctext> FHEONTrainBackwardPassController::backward_pass_batch_norm(vector<Ctext> &encryptedInputs, vector<Ctext> &gradInputs, int inputSize, double learningRate, string layerName) {

    double scaler = (-1.0/inputSize);
    auto cleaningMaskVec = generate_scale_and_clean_mask(scaler, inputSize);
    vector<Ctext> batchNormGradients(gradInputs.size());
    
    int numChannels = (int)gradInputs.size();

    // Load ahead of time in parallel so the bootstrap step below only has to
    // block on levels, not disk I/O.
    vector<Ctext> xHatAll(numChannels);
    vector<Ctext> learnedandinvstdAll(numChannels);
    run_parallel_for(numChannels, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            string xHatFileName = "batch_norm_xhat_" + layerName + "_channel_" + to_string(i);
            xHatAll[i] = deserialize_and_load(xHatFileName);
            string invstdFileName = "batch_norm_invstd" + layerName + "_channel_" + to_string(i);
            learnedandinvstdAll[i] = deserialize_and_load(invstdFileName);
        }
    });
    // EvalBootstrap must run on this (the caller's) thread, one value at a time --
    // it is not safe to call from a worker thread at all, not just concurrently.
    for (int i = 0; i < numChannels; i++) {
        if (xHatAll[i]->GetLevel() > 10) xHatAll[i] = context->EvalBootstrap(xHatAll[i], 2);
        if (learnedandinvstdAll[i]->GetLevel() > 10) learnedandinvstdAll[i] = context->EvalBootstrap(learnedandinvstdAll[i], 2);
    }
    
    vector<Ctext> sumedLossValues(numChannels);
    vector<Ctext> sumedInnGammaValues(numChannels);
    auto gradients_start_time = startTime();
    run_parallel_for(numChannels, [&](int start, int end) {
        for(int i=start; i<end; i++){
            Ctext xHat = xHatAll[i];

            uint32_t currentLevel = gradInputs[i]->GetLevel();
            auto cleaningMaskPtxt = context->MakeCKKSPackedPlaintext(cleaningMaskVec, 1, currentLevel, nullptr, inputSize);

            /** Compute the mean */
            Ctext sumedLoss = context->EvalSum(gradInputs[i], inputSize);
            Ctext meanVal = context->EvalMult(sumedLoss, cleaningMaskPtxt);
            meanVal = repeat_encrypted_values(meanVal, 1, inputSize);

            Ctext innGamma = context->EvalMult(gradInputs[i], xHat);
            Ctext sumedInnGamma = context->EvalSum(innGamma, inputSize);

            auto cleaningMaskPtxt2 = context->MakeCKKSPackedPlaintext(cleaningMaskVec, 1, sumedInnGamma->GetLevel(), nullptr, inputSize);
            Ctext meanInnGamma = context->EvalMult(sumedInnGamma, cleaningMaskPtxt2);
            meanInnGamma = repeat_encrypted_values(meanInnGamma, 1, inputSize);

            Ctext varienceTerm = context->EvalMult(xHat, meanInnGamma);
            Ctext innTerm = context->EvalAdd(gradInputs[i], context->EvalAdd(varienceTerm, meanVal));

            Ctext learnedandinvstd = learnedandinvstdAll[i];
            batchNormGradients[i] = context->EvalMult(innTerm, learnedandinvstd);
            sumedLossValues[i] = sumedLoss;
            sumedInnGammaValues[i] = sumedInnGamma;
        }
    });
    measureTiming("batch norm gradients...", "decrypt_batch_norm", gradients_start_time);

    run_parallel_for(numChannels, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            /** Update the learnable parameters using the computed gradients */
            string fileName = "batch_norm_" + layerName + "_channel_" + to_string(i);
            Ctext learnableCtxts = deserialize_and_load(fileName);
            learnableCtxts = optimizer_update(learnableCtxts, sumedInnGammaValues[i], learningRate);
            serialize_and_store(learnableCtxts, fileName);

            string biasFileName = "batch_norm_bias_" + layerName + "_channel_" + to_string(i);
            Ctext learnablebiasCtxts = deserialize_and_load(biasFileName);
            learnablebiasCtxts = optimizer_update(learnablebiasCtxts, sumedLossValues[i], learningRate);
            serialize_and_store(learnablebiasCtxts, biasFileName);

            string backwardGradFileName = "batch_norm_grad_" + layerName + "_channel_" + to_string(i);
            serialize_and_store(batchNormGradients[i], backwardGradFileName);
        }
    });

    return batchNormGradients;
}

/***
 * Backpropagation through a fully connected layer.
 *
 * Matches he_forward_linear_multiple_outputs's actual forward semantics: a
 * distinct weight per (input channel, output neuron) pair, broadcast as a
 * scalar over the full vectorSize width — NOT a single weight shared across
 * all input channels. For output[ou] = sum_inCh(weight[ou][inCh]*input[inCh]) + bias[ou]:
 *
 *   dL/dInput[inCh]        = sum_ou( gradOutput[ou] * weight[ou][inCh] )
 *   dL/dWeight[ou][inCh]   = sum_over_vector( gradOutput[ou] * input[inCh] )
 *   dL/dBias[ou]           = sum_over_vector( gradOutput[ou] )
 *
 * @param gradInputs   Vector of ciphertexts representing the loss gradients from the next layer (one per output neuron).
 * @param vectorSize   Number of valid slots in each channel ciphertext (batchSize * spatialSize) — must match the
 *                      vectorSize used by the matching forward call, since it bounds the EvalSum reduction below.
 * @param inputSize    Number of input channels (neurons).
 * @param outputSize   Number of output channels (neurons).
 * @param learningRate Learning rate for updating weights and biases.
 * @param layerName    Name of the layer for file naming purposes.
 *
 * @return A vector of ciphertexts containing the gradients for each input channel.
 */
vector<Ctext> FHEONTrainBackwardPassController::backward_pass_fully_connected(vector<Ctext> &gradInputs, int vectorSize, int inputSize, int outputSize,
                                double learningRate, string layerName) {

    vector<vector<Ctext>> weightCiphertexts(outputSize, vector<Ctext>(inputSize));
    double fan_in = inputSize;
    run_parallel_for(outputSize, [&](int start, int end) {
        for (int ou = start; ou < end; ou++) {
            for (int inCh = 0; inCh < inputSize; inCh++) {
                string fileName = layerName + "_out_neuron_" + to_string(ou) + "_in_" + to_string(inCh);
                weightCiphertexts[ou][inCh] = deserialize_and_load(fileName, fan_in, vectorSize);
            }
        }
    });

    vector<Ctext> gradients(inputSize);
    vector<vector<Ctext>> weightGrad(outputSize, vector<Ctext>(inputSize));

    // dL/dInput, parallelized across input channels.
    auto start_time = startTime();
    run_parallel_for(inputSize, [&](int start, int end) {
        for(int inCh = start; inCh < end; inCh++){
            string inputFileName = layerName + "_in_neuron_" + to_string(inCh);
            Ctext inputCtxt = deserialize_and_load(inputFileName);

            vector<Ctext> inputGradTerms(outputSize);
            for(int ou = 0; ou < outputSize; ou++){
                string weightsFileName = layerName + "_out_neuron_" + to_string(ou) + "_in_" + to_string(inCh);
                Ctext weightCtxt = deserialize_and_load(weightsFileName);
                inputGradTerms[ou] = context->EvalMult(gradInputs[ou], weightCtxt);
                weightGrad[ou][inCh] = context->EvalMult(gradInputs[ou], inputCtxt);
            }
            gradients[inCh] = context->EvalAddMany(inputGradTerms);
        }
    });

    printTiming("Fully Connected backward evaluating...", "fc", start_time);

    // Weight and bias updates, parallelized across output neurons.
    run_parallel_for(outputSize, [&](int start, int end) {
        for(int ou = start; ou < end; ou++){
            for(int inCh = 0; inCh < inputSize; inCh++){
                string weightsFileName = layerName + "_out_neuron_" + to_string(ou) + "_in_" + to_string(inCh);
                Ctext sumedWeightGrad = context->EvalSum(weightGrad[ou][inCh], vectorSize);

                Ctext weightCtxt = optimizer_update(weightCiphertexts[ou][inCh], sumedWeightGrad, learningRate);
                serialize_and_store(weightCtxt, weightsFileName);
            }

            Ctext biasGradSum = context->EvalSum(gradInputs[ou], vectorSize);
            string biasFileName = layerName + "_out_neuron_" + to_string(ou) + "_bias";

            Ctext inputbiasCtxt = deserialize_and_load(biasFileName);
            inputbiasCtxt = optimizer_update(inputbiasCtxt, biasGradSum, learningRate);
            serialize_and_store(inputbiasCtxt, biasFileName);
        }
    });

    return gradients;
}

/***
 * Backpropagation through a convolutional layer.
 * This function computes the gradient of the loss with respect to the inputs
 * by performing convolution operations with the weight matrix and adding the bias.
 *
 * @param encryptedInputs  Vector of ciphertexts containing encrypted inputs.
 * @param gradInputs       Vector of ciphertexts representing the loss gradients from the next layer.
 * @param batchSize        Number of elements in the batch (N).
 * @param inputWidth       Width of the input feature map.
 * @param inputChannels    Number of channels in the input feature map.
 * @param outputChannels   Number of channels in the output feature map.
 * @param kernelWidth      Width of the convolutional kernel.
 * @param learningRate     Learning rate for updating weights and biases.
 * @param layerName        Name of the layer for file naming purposes.
 *
 * @return A vector of ciphertexts containing the gradients for each input channel.
 */
vector<Ctext> FHEONTrainBackwardPassController::backward_pass_convolution(vector<Ctext> &gradInputs, int batchSize, int inputWidth,
                    int inputChannels, int outputChannels, int kernelWidth, double learningRate, string layerName, int stride) {

    int kernelSize = kernelWidth * kernelWidth;
    int totalInputSize = inputWidth * inputWidth * batchSize;
    std::cout << "[DEBUG] kernelSize: " << kernelSize << ", totalInputSize: " << totalInputSize << std::endl;
    cout << "Starting backpropagation through convolutional layer: " << layerName << endl;
    cout << "Batch size: " << batchSize << ", Input width: " << inputWidth << ", Input channels: " << inputChannels
         << ", Output channels: " << outputChannels << ", Kernel width: " << kernelWidth << ", Stride: " << stride << endl;

    int outputWidth = inputWidth / stride;
    int totalOutputSize = outputWidth * outputWidth * batchSize;
    vector<Ctext> biasGradAll(outputChannels);
    run_parallel_for(outputChannels, [&](int start, int end) {
        for (int oc = start; oc < end; oc++) {
            biasGradAll[oc] = context->EvalSum(gradInputs[oc], totalOutputSize);
        }
    });

    if (stride > 1) {
        run_parallel_for(outputChannels, [&](int start, int end) {
            for (int oc = start; oc < end; oc++) {
                gradInputs[oc] = backward_dilate_logdepth(gradInputs[oc], batchSize, outputWidth, stride);
            }
        });
        for (int oc = 0; oc < outputChannels; oc++) {
            gradInputs[oc] = context->EvalBootstrap(gradInputs[oc], 2);
        }
    }

    auto start_time = startTime();
    vector<vector<Ctext>> rotatedGradOut(outputChannels, vector<Ctext>(kernelSize));
    run_parallel_for(outputChannels, [&](int start, int end) {
        for (int oc = start; oc < end; oc++) {
            Ctext replicatedGradOut = replicate_across_ring(context, gradInputs[oc], totalInputSize, num_slots);
            auto digits = context->EvalFastRotationPrecompute(replicatedGradOut);

            Ctext first_shot = context->EvalRotate(replicatedGradOut, -1);
            Ctext second_shot = context->EvalFastRotation(replicatedGradOut, 1, context->GetCyclotomicOrder(), digits);

            rotatedGradOut[oc][0] = context->EvalRotate(first_shot, -inputWidth);
            rotatedGradOut[oc][1] = context->EvalRotate(replicatedGradOut, -inputWidth);
            rotatedGradOut[oc][2] = context->EvalRotate(second_shot, -inputWidth);
            rotatedGradOut[oc][3] = first_shot;
            rotatedGradOut[oc][4] = replicatedGradOut;
            rotatedGradOut[oc][5] = second_shot;
            rotatedGradOut[oc][6] = context->EvalRotate(first_shot, inputWidth);
            rotatedGradOut[oc][7] = context->EvalFastRotation(replicatedGradOut, inputWidth, context->GetCyclotomicOrder(), digits);
            rotatedGradOut[oc][8] = context->EvalRotate(second_shot, inputWidth);
        }
    });

    vector<Ctext> gradiantsCtxts(inputChannels);
    run_parallel_for(inputChannels, [&](int start, int end) {
        for (int in = start; in < end; in++) {
            vector<Ctext> outputGradients(outputChannels);
            for(int oc = 0; oc < outputChannels; oc++){
                vector<Ctext> kernelCtxts(kernelSize);
                for (int ik = 0; ik < kernelSize; ik++) {
                    int idx = kernelSize - (ik + 1);
                    string weightsFileName = layerName + "_weights_" + to_string(oc) + "_" + to_string(in) + "_" + to_string(idx);
                    Ctext weightCtxt = deserialize_and_load(weightsFileName, inputChannels * kernelSize, totalInputSize);
                    kernelCtxts[ik] = context->EvalMult(rotatedGradOut[oc][ik], weightCtxt);
                }
                outputGradients[oc] = context->EvalAddMany(kernelCtxts);
            }
            gradiantsCtxts[in] = context->EvalAddMany(outputGradients);
        }
    });

    vector<vector<Ctext>> rotatedInputs(inputChannels, vector<Ctext>(kernelSize));
    run_parallel_for(inputChannels, [&](int start, int end) {
        for (int in = start; in < end; in++) {
            string inputFileName = layerName + "_channel_" + to_string(in);
            Ctext cInput = deserialize_and_load(inputFileName);
            cInput = replicate_across_ring(context, cInput, totalInputSize, num_slots);
            auto digits = context->EvalFastRotationPrecompute(cInput);

            Ctext first_shot = context->EvalRotate(cInput, -1);   // input[idx-1]
            Ctext second_shot = context->EvalFastRotation(cInput, 1, context->GetCyclotomicOrder(), digits);   // input[idx+1]

            rotatedInputs[in][0] = context->EvalRotate(first_shot, -inputWidth);                                          // -(W+1)
            rotatedInputs[in][1] = context->EvalRotate(cInput, -inputWidth);                                              // -W
            rotatedInputs[in][2] = context->EvalRotate(second_shot, -inputWidth);                                         // -(W-1)
            rotatedInputs[in][3] = first_shot;                                                                            // -1
            rotatedInputs[in][4] = cInput;                                                                                // 0
            rotatedInputs[in][5] = second_shot;                                                                           // +1
            rotatedInputs[in][6] = context->EvalRotate(first_shot, inputWidth);                                           // (W-1)
            rotatedInputs[in][7] = context->EvalFastRotation(cInput, inputWidth, context->GetCyclotomicOrder(), digits);  // W
            rotatedInputs[in][8] = context->EvalRotate(second_shot, inputWidth);                                          // (W+1)
        }
    });
    printTiming("Convolution backward evaluating...", "convolution", start_time);

    run_parallel_for(outputChannels, [&](int start, int end) {
        for(int oc = start; oc < end; oc++){
            for(int in = 0; in < inputChannels; in++){
                for(int ik = 0; ik < kernelSize; ik++){
                    Ctext weigthGrad = context->EvalMult(gradInputs[oc], rotatedInputs[in][ik]);
                    Ctext sumedWeightGrad = context->EvalSum(weigthGrad, totalInputSize);
                    string weightsFileName = layerName + "_weights_" + to_string(oc) + "_" + to_string(in) + "_" + to_string(ik);
                    Ctext weightCtxt = deserialize_and_load(weightsFileName, inputChannels * kernelSize, num_slots);
                    weightCtxt = optimizer_update(weightCtxt, sumedWeightGrad, learningRate);
                    serialize_and_store(weightCtxt, weightsFileName);
                }
            }

            // bias update (biasGradAll computed pre-dilation above, at output resolution)
            Ctext biasGrad = biasGradAll[oc];
            string biasFileName = layerName + "_bias_" + to_string(oc);
            Ctext biasCtxt = deserialize_and_load(biasFileName, 1.0, num_slots);
            biasCtxt = optimizer_update(biasCtxt, biasGrad, learningRate);
            serialize_and_store(biasCtxt, biasFileName);
        }
    });

    return gradiantsCtxts;
}

/***
 * Backpropagation through the 1x1 stride-capable "shortcut" convolution (see
 * FHEONTrainForwardPassController::he_forward_convolution_shortcut_optimized).
 * No spatial tap: structurally identical to backward_pass_fully_connected, with
 * gradInputs first upsampled from output to input resolution via backward_dilate
 * (see backward_pass_convolution's stride>1 doc comment for why this dilation,
 * not an average, is the correct adjoint of forward's downsample-by-selection).
 *
 *   dL/dInput[in]      = sum_oc( dilatedGrad[oc] * weight[oc][in] )   (elementwise, input resolution)
 *   dL/dWeight[oc][in] = sum_over_input_res( dilatedGrad[oc] * input[in] )
 *   dL/dBias[oc]       = sum_over_output_res( gradInputs[oc] )   (pre-dilation, output resolution)
 *
 * @param gradInputs      Loss gradients from the next layer, one per output channel, at output resolution.
 * @param batchSize       Number of elements in the batch (N).
 * @param inputWidth      Width of this layer's INPUT feature map (pre-downsample).
 * @param inputChannels   Number of input channels.
 * @param outputChannels  Number of output channels.
 * @param stride          Stride used by the matching forward call (1 or 2 in this codebase).
 * @param learningRate    Learning rate for updating weights and bias.
 * @param layerName       Name of the layer for file naming purposes (must match the forward call's).
 *
 * @return A vector of ciphertexts containing the gradients for each input channel, at input resolution.
 */
vector<Ctext> FHEONTrainBackwardPassController::backward_pass_convolution_shortcut(vector<Ctext> &gradInputs, int batchSize, int inputWidth,
                    int inputChannels, int outputChannels, int stride, double learningRate, string layerName) {
   
    int outputWidth = inputWidth / stride;
    int inVecSize = inputWidth * inputWidth * batchSize;
    int outVecSize = outputWidth * outputWidth * batchSize;

    // Bias gradient at OUTPUT resolution, pre-dilation (bias is added after
    // downsampling in the forward pass, same reasoning as backward_pass_convolution).
    vector<Ctext> biasGradAll(outputChannels);
    run_parallel_for(outputChannels, [&](int start, int end) {
        for (int oc = start; oc < end; oc++) {
            biasGradAll[oc] = context->EvalSum(gradInputs[oc], outVecSize);
        }
    });

    auto start_time = startTime();

    // Upsample gradInputs to input resolution (no-op copy when stride==1). The
    // dilation runs in parallel; the refresh bootstrap runs afterward, one channel
    // at a time on this (the caller's) thread -- EvalBootstrap is not safe to call
    // from a worker thread at all, not just concurrently.
    vector<Ctext> dilatedGrad(outputChannels);
    run_parallel_for(outputChannels, [&](int start, int end) {
        for (int oc = start; oc < end; oc++) {
            dilatedGrad[oc] = (stride > 1) ? backward_dilate_logdepth(gradInputs[oc], batchSize, outputWidth, stride) : gradInputs[oc];
        }
    });

    if (stride > 1) {
        for (int oc = 0; oc < outputChannels; oc++) {
            dilatedGrad[oc] = context->EvalBootstrap(dilatedGrad[oc], 2);
        }
    }

    vector<Ctext> cachedInputs(inputChannels);
    run_parallel_for(inputChannels, [&](int start, int end) {
        for (int inCh = start; inCh < end; inCh++) {
            string inputFileName = layerName + "_in_neuron_" + to_string(inCh);
            cachedInputs[inCh] = deserialize_and_load(inputFileName);
        }
    });

    vector<vector<Ctext>> weightCiphertexts(outputChannels, vector<Ctext>(inputChannels));
    double fan_in = inputChannels;
    run_parallel_for(outputChannels, [&](int start, int end) {
        for (int oc = start; oc < end; oc++) {
            for (int inCh = 0; inCh < inputChannels; inCh++) {
                string fileName = layerName + "_out_neuron_" + to_string(oc) + "_in_" + to_string(inCh);
                weightCiphertexts[oc][inCh] = deserialize_and_load(fileName, fan_in, inVecSize);
            }
        }
    });

    // dL/dInput, parallelized across input channels.
    vector<Ctext> gradientsCtxts(inputChannels);
    run_parallel_for(inputChannels, [&](int start, int end) {
        for (int inCh = start; inCh < end; inCh++) {
            vector<Ctext> terms(outputChannels);
            for (int oc = 0; oc < outputChannels; oc++) {
                terms[oc] = context->EvalMult(dilatedGrad[oc], weightCiphertexts[oc][inCh]);
            }
            gradientsCtxts[inCh] = context->EvalAddMany(terms);
        }
    });
    printTiming("Convolution Shortcut backward evaluating...", "batch_norm", start_time);

    run_parallel_for(outputChannels, [&](int start, int end) {
        for (int oc = start; oc < end; oc++) {
            for (int inCh = 0; inCh < inputChannels; inCh++) {
                Ctext weightGrad = context->EvalMult(dilatedGrad[oc], cachedInputs[inCh]);
                Ctext sumedWeightGrad = context->EvalSum(weightGrad, inVecSize);

                string fileName = layerName + "_out_neuron_" + to_string(oc) + "_in_" + to_string(inCh);
                Ctext weightCtxt = optimizer_update(weightCiphertexts[oc][inCh], sumedWeightGrad, learningRate);
                serialize_and_store(weightCtxt, fileName);
            }

            string biasFileName = layerName + "_out_neuron_" + to_string(oc) + "_bias";
            Ctext biasCtxt = deserialize_and_load(biasFileName, 1.0, num_slots);
            biasCtxt = optimizer_update(biasCtxt, biasGradAll[oc], learningRate);
            serialize_and_store(biasCtxt, biasFileName);
        }
    });

    cout << "convolution shortcut layer " << layerName << " backward pass completed." << endl;
    // printTiming("Backward convolution shortcut evaluated...", "backward_convolution_shortcut", start_time);
    return gradientsCtxts;
}

/***
 * Backpropagation through global average pooling (see
 * FHEONTrainForwardPassController::he_forward_global_avgpool).
 *
 * Forward compacts each channel's (batchSize x dataWidth x dataWidth) map down
 * to one scalar per image (contiguous, one slot per image). Backward is the
 * adjoint: broadcast dOutput[b]/N back to all N spatial positions of image b's
 * block, for every image -- built entirely from two already-validated
 * primitives, no new low-level HE math:
 *   1. backward_dilate(gradOutput[oc], batchSize, 1, dataWidth) scatters each of
 *      the batchSize compact scalars to the leading slot of its own N-sized
 *      block (degenerate case of the same top-left-corner placement
 *      backward_pass_convolution's stride>1 path relies on).
 *   2. repeat_encrypted_values(dilated, 1, N) smears that leading slot across
 *      the rest of its own N-sized block via the standard log2-doubling
 *      rotate-add (the smear's rotation offsets {1,2,4,...,N/2} never reach or
 *      exceed the N-wide block period, so this never bleeds into a
 *      neighboring image's block).
 *   3. Scale by 1/N (average, not sum).
 *
 * @param gradInputs   Loss gradients from the next layer, one per channel, compact (one value per image).
 * @param batchSize    Number of elements in the batch (N).
 * @param dataWidth    Width of the pre-pool feature map (the forward call's dataWidth).
 *
 * @return A vector of ciphertexts containing the full-resolution gradients, one per channel.
 */
vector<Ctext> FHEONTrainBackwardPassController::backward_pass_global_avgpool(vector<Ctext> &gradInputs, int batchSize, int dataWidth) {

    int spatialSize = dataWidth * dataWidth;
    double scaler = 1.0 / spatialSize;
    vector<Ctext> result(gradInputs.size());
    
    auto start_time = startTime();
    run_parallel_for(gradInputs.size(), [&](int start, int end) {
        for (int oc = start; oc < end; oc++) {
            Ctext dilated = backward_dilate(gradInputs[oc], batchSize, 1, dataWidth);
            Ctext smeared = repeat_encrypted_values(dilated, 1, spatialSize);
            result[oc] = context->EvalMult(smeared, scaler);
        }
    });
    printTiming("Backward global average pooling evaluated...", "backward_global_avgpool", start_time);

    return result;
}


/***
 * Computing approximate square root using Newton–Raphson Iteration (double-precision)
 *  @param encryptedInput: The encrypted input value for which the square root is to be computed.
 *  @return: The approximate square root of the encrypted input value.
 */
Ctext FHEONTrainBackwardPassController::newtonraphson_approximate_inverse_sqrt(Ctext& encryptedInput, int inputSize, double initValue, int appxDegree) {
    vector<double> initVec(inputSize, initValue);
    Ptext initPlain = context->MakeCKKSPackedPlaintext(initVec, 1, encryptedInput->GetLevel());
    Ctext y = context->EvalAdd(initPlain, encryptedZeros);

    for (int n = 0; n < appxDegree; n++) {
        auto y2 = context->EvalSquare(y);
        auto g  = context->EvalMult(encryptedInput, y2);
        auto t = context->EvalSub(1.5, context->EvalMult(0.5, g));
        y = context->EvalMult(y, t);
    }
    return y;
}



/***
 * Helper function to repeat encrypted values across the batch for convolution operations.
 * This function takes an encrypted input and creates a new ciphertext where the input value is repeated
 * across all slots corresponding to the batch size. This is useful for convolution operations where the same input value needs to be applied across multiple positions in the feature map.
 *
 * @param encryptedInput  The encrypted input value to be repeated.
 * @param vecSize         The size of the vector (number of slots) in the ciphertext.
 * @param batchSize       The number of elements in the batch (N).
 *
 * @return A new ciphertext with the input value repeated across the batch.
 */
Ctext FHEONTrainBackwardPassController::repeat_encrypted_values(Ctext& encryptedInput, int vecSize, int batchSize){
    batchSize = nextPowerOf2(batchSize);
    Ctext tempIn = context->EvalAdd(encryptedInput, context->EvalRotate(encryptedInput, -vecSize));

    for(int j=1; j< log2(batchSize); j++){
        int powVal = pow(2, j);
        tempIn = context->EvalAdd(tempIn, context->EvalRotate(tempIn, -powVal));
    }
    return tempIn;
}

/***
 * Optimizer update function.
 * This function updates the encrypted parameters using the encrypted gradients and learning rate.
 *
 * @param encryptedParam   The encrypted parameter to be updated.
 * @param encryptedGrad    The encrypted gradient.
 * @param learningRate     The learning rate for the optimizer.
 *
 * @return A new ciphertext with the updated parameter.
 */
Ctext FHEONTrainBackwardPassController::optimizer_update(Ctext& encryptedParam, Ctext& encryptedGrad, double learningRate){
    Ctext grad = encryptedGrad;
    // if (grad->GetLevel() > 20) {
    //     cerr << "[optimizer_update] Bootstrapping high level gradient (level=" << grad->GetLevel() << ")" << endl;
    //     grad = context->EvalBootstrap(grad, 2);
    // }
    return context->EvalSub(encryptedParam, context->EvalMult(grad, learningRate));
}

/***
 * Same as the Ctext overload above, but for a parameter that has never been
 * trained yet -- still just a pretrained plaintext constant (see
 * he_forward_convolution_optimized's pretrainedFromCsv doc comment), not a
 * ciphertext. EvalSub(Plaintext, Ciphertext) is a valid mixed CKKS op and
 * produces the first-ever trained Ctext for that param; every later call for
 * the same param goes through the Ctext overload above instead, since it will
 * have a cached ciphertext on disk from this point on.
 */
Ctext FHEONTrainBackwardPassController::optimizer_update(Ptext& constParam, Ctext& encryptedGrad, double learningRate){
    Ctext grad = encryptedGrad;
    // if (grad->GetLevel() > 20) {
    //     cerr << "[optimizer_update] Bootstrapping high level gradient (level=" << grad->GetLevel() << ")" << endl;
    //     grad = context->EvalBootstrap(grad, 2);
    // }
    return context->EvalSub(constParam, context->EvalMult(grad, learningRate));
}

/***
 * DEPRECATED — kept only as a backup of the original design. Do not call.
 *
 * This has two bugs: (1) backward_dilate places each pooled gradient at only
 * the top-left corner of its 2x2 window, leaving the other 3 positions at
 * zero — real avg-pool backward must give all 4 positions an equal share;
 * (2) it then multiplies by dilateFactor^2 (=4) instead of dividing, so the
 * one populated position ends up 16x too large relative to a correct equal
 * split, and the total gradient mass per window (4x forward's pooled
 * gradient) isn't conserved. See backward_pass_avgpool for the fix, verified
 * against a standalone OpenFHE test (single- and multi-batch) before being
 * wired in here.
 */
vector<Ctext> FHEONTrainBackwardPassController::backward_pass_avgpool_legacy(vector<Ctext> &encryptedInputs, int batchSize, int inputWidth, int dilateFactor) {

  
    vector<Ctext> pooledGradients(encryptedInputs.size());
    double scaler = (dilateFactor * dilateFactor);

    auto start_time = startTime();
    run_parallel_for(encryptedInputs.size(), [&](int start, int end) {
        for(int i=start; i<end; i++){
            Ctext dilatedCtxt = backward_dilate(encryptedInputs[i], batchSize, inputWidth, dilateFactor);
            pooledGradients[i] = context->EvalMult(dilatedCtxt, scaler);
        }
    });
    printTiming("Backward average pooling (legacy) evaluated...", "backward_avgpool", start_time);
    return pooledGradients;
}

/***
 * Backward average pooling function (2x2, stride 2 — the only configuration
 * used in this codebase).
 *
 * backward_dilate places each pooled gradient value at the top-left corner
 * of its 2x2 window (zero elsewhere) — this part is correct and unchanged.
 * What's added here is the adjoint of forward avg-pool's own STEP1 (which
 * sums the 4 window members via shifts {0,+1,+width,+width+1} then divides
 * by 4): summing the dilated ciphertext with itself shifted by
 * {0,-1,-width,-(width+1)} moves that corner value into all 4 of its window
 * positions (exactly one of the 4 shifted copies is nonzero at any given
 * target slot, since the sparse pattern only has values on the even-row,
 * even-col corners), then dividing by dilateFactor^2 gives every position
 * its correct equal share of the pooled gradient. Validated (including the
 * multi-batch block-boundary case, where dilateFactor*inputWidth being even
 * guarantees the wraparound rotations only ever land on non-corner — hence
 * zero — slots of the adjacent block) against a standalone OpenFHE test
 * before being wired in here.
 *
 * @param encryptedInputs The pooled-resolution gradients, one ciphertext per channel.
 * @param batchSize       The number of elements in the batch (N).
 * @param inputWidth      The (pooled/source) width of the feature map being backward-dilated.
 * @param dilateFactor    The pooling kernel/stride width (always 2 here).
 *
 * @return The full-resolution gradients, one ciphertext per channel.
 */
vector<Ctext> FHEONTrainBackwardPassController::backward_pass_avgpool(vector<Ctext> &encryptedInputs, int batchSize, int inputWidth, int dilateFactor) {

    if (dilateFactor != 2) {
        throw std::runtime_error("backward_pass_avgpool: the 4-term adjoint sum below is hardcoded for a 2x2 pooling window (dilateFactor=2); generalize the shift set {0,-1,-w,-(w+1)} before using a different dilateFactor.");
    }

    auto start_time = startTime();
    vector<Ctext> pooledGradients(encryptedInputs.size());
    int dilatedWidth = dilateFactor * inputWidth;
    double scaler = 1.0 / (dilateFactor * dilateFactor);

    run_parallel_for(encryptedInputs.size(), [&](int start, int end) {
        for(int i=start; i<end; i++){
            Ctext dilatedCtxt = backward_dilate(encryptedInputs[i], batchSize, inputWidth, dilateFactor);

            vector<Ctext> shifted;
            shifted.push_back(dilatedCtxt);
            shifted.push_back(context->EvalRotate(dilatedCtxt, -1));
            shifted.push_back(context->EvalRotate(dilatedCtxt, -dilatedWidth));
            shifted.push_back(context->EvalRotate(dilatedCtxt, -(dilatedWidth + 1)));
            Ctext smeared = context->EvalAddMany(shifted);

            pooledGradients[i] = context->EvalMult(smeared, scaler);
        }
    });
    printTiming("Backward average pooling evaluated...", "backward_avgpool", start_time);
    return pooledGradients;
}

/***
 * Backward dilation function.
 * This function performs backward dilation on an encrypted input.
 * 
 * For each input ciphertext, the function creates a dilated version by inserting
 * zeros between the values according to the specified dilation factor between the rows.
 * It also introduce zeros between the rows of the input feature map.
 * 
 * @param encryptedInput  The encrypted input to be dilated.
 * @param batchSize       The number of elements in the batch (N).
 * @param inputWidth      The width of the input feature map.
 * @param dilateFactor    The dilation factor for the backward dilation operation.
 *
 * @return The dilated encrypted ciphertext.
 */
Ctext FHEONTrainBackwardPassController::backward_dilate(Ctext& encryptedInput, int batchSize, int inputWidth, int dilateFactor){
    
    Ctext rowCtxt;
    Ctext dilatedCtxt;
    int rowSize = dilateFactor * inputWidth;
    int channelSize = inputWidth * inputWidth;
    int dilatedChannelSize = rowSize * rowSize;
    vector<Ctext> rowDilatedCtxts(rowSize);

    // cout << "Starting backward dilation for input width: " << inputWidth << ", batch size: " << batchSize << ", dilation factor: " << dilateFactor << endl;
    
    Ctext dilatedInput = encryptedInput->Clone();
    for(int bt = 0; bt < batchSize; bt++){
        if(bt > 0){
            dilatedInput = context->EvalRotate(dilatedInput, channelSize);
        }
        Ctext channelShifted = dilatedInput->Clone();

        // cout << "Processing batch " << bt << " for backward dilation." << endl;
        for(int vt=0; vt < inputWidth; vt++){
            if(vt > 0){
                channelShifted = context->EvalRotate(channelShifted, inputWidth);    
            }

            Ctext rowShifted = channelShifted->Clone();
            for(int ht =0; ht < inputWidth; ht++){
                for(int dt = 0; dt < dilateFactor; dt++){
                    int idx = ht * dilateFactor + dt;
                    if(dt == 0){
                        rowDilatedCtxts[idx] = rowShifted;
                        rowShifted = context->EvalRotate(rowShifted, 1);
                    }
                    else{
                        rowDilatedCtxts[idx] = encryptedZeros->Clone();
                    }
                }
            }

            Ctext mergedRow = context->EvalMerge(rowDilatedCtxts); 
            int rowsShift = (vt * dilateFactor) * rowSize;
            if(rowsShift == 0){
                rowCtxt = mergedRow;
            }
            else{
                rowCtxt = context->EvalAdd(rowCtxt, context->EvalRotate(mergedRow, -rowsShift));
            }
        }

        // calculate the shift amount for the current batch
        int batchShift = bt * dilatedChannelSize;
        if(batchShift == 0){
            dilatedCtxt = rowCtxt;
        }
        else{
            dilatedCtxt = context->EvalAdd(dilatedCtxt, context->EvalRotate(rowCtxt, -batchShift));
        }
    }
    return dilatedCtxt;
}


vector<int> FHEONTrainBackwardPassController::backward_dilate_indices(int batchSize, int inputWidth, int dilateFactor){
    vector<int> keys_positions;
    int rowSize = dilateFactor * inputWidth;
    int channelSize = inputWidth * inputWidth;
    int dilatedChannelSize = rowSize * rowSize;
    keys_positions.push_back(inputWidth);
    keys_positions.push_back(channelSize);

    for(int i=1; i< rowSize; i++){
        keys_positions.push_back(i);
        // keys_positions.push_back(-i);

        if(i<= inputWidth){
            int vt = i * dilateFactor * rowSize;
            keys_positions.push_back(-vt);
        }
    }

    for(int i=1; i< batchSize; i++){
        int batchShift = i * dilatedChannelSize;
        keys_positions.push_back(-batchShift);
    }

    std::sort(keys_positions.begin(), keys_positions.end());
    auto new_end = std::remove(keys_positions.begin(), keys_positions.end(), 0);
    new_end = std::unique(keys_positions.begin(), keys_positions.end());
    unique(keys_positions.begin(), keys_positions.end());
    keys_positions.erase(new_end, keys_positions.end());
    std::sort(keys_positions.begin(), keys_positions.end());
    return keys_positions;
}

Ptext FHEONTrainBackwardPassController::gen_slot0_mask(uint32_t level){
    size_t slots = context->GetRingDimension() / 2;
    vector<double> maskVec(slots, 0.0);
    maskVec[0] = 1.0;
    return context->MakeCKKSPackedPlaintext(maskVec, 1, level);
}

/***
 * Optimized backward dilation -- mathematically identical output to
 * backward_dilate (see its doc comment for what dilation means here), but
 * far cheaper.
 *
 * backward_dilate builds each dilated row by handing EvalMerge a
 * rowSize=dilateFactor*inputWidth-length vector where only every
 * dilateFactor-th entry is real data and the rest are encryptedZeros
 * clones -- EvalMerge still pays a full EvalMult(mask) + EvalAtIndex +
 * EvalAdd for EVERY one of those rowSize entries internally, including the
 * (dilateFactor-1)*inputWidth entries that are provably no-ops (0*mask=0,
 * x+0=x). This version isolates and places only the inputWidth REAL values
 * directly -- mask out everything but the value currently rotated into slot
 * 0, rotate that isolated value to its target column, accumulate -- cutting
 * the per-row cost from O(dilateFactor*inputWidth) down to O(inputWidth),
 * independent of dilateFactor (a `dilateFactor`x reduction on that term:
 * e.g. ~2x fewer ops for the stage2/stage3 downsample-block dilate calls
 * (dilateFactor=2), ~8x fewer for global-avgpool's (dilateFactor=dataWidth)).
 * The row/batch shift-and-accumulate structure (already O(inputWidth) /
 * O(batchSize), not the bottleneck) is unchanged from backward_dilate.
 *
 * Needs its own rotation-key set: see backward_dilate_optimized_indices.
 * The per-column "-ht*dilateFactor" placement rotations here are NOT the
 * same indices backward_dilate's EvalMerge needs internally, so a caller
 * switching a real pipeline call from backward_dilate to this must also add
 * backward_dilate_optimized_indices(...)'s output to that stage's rotation
 * key set (e.g. via add_and_persist_rotation_keys) before running -- this
 * function has not been wired into tuneResNet20.cpp's real per-stage
 * keysets, only exercised against TrainingModules.cpp's validation keyset
 * (see validate_avgpool_optimized/validate_global_avgpool_optimized there).
 */
Ctext FHEONTrainBackwardPassController::backward_dilate_optimized(Ctext& encryptedInput, int batchSize, int inputWidth, int dilateFactor){

    Ctext rowCtxt;
    Ctext dilatedCtxt;
    int rowSize = dilateFactor * inputWidth;
    int channelSize = inputWidth * inputWidth;
    int dilatedChannelSize = rowSize * rowSize;

    Ctext dilatedInput = encryptedInput->Clone();
    for(int bt = 0; bt < batchSize; bt++){
        if(bt > 0){
            dilatedInput = context->EvalRotate(dilatedInput, channelSize);
        }
        Ctext channelShifted = dilatedInput->Clone();

        for(int vt=0; vt < inputWidth; vt++){
            if(vt > 0){
                channelShifted = context->EvalRotate(channelShifted, inputWidth);
            }

            Ctext rowShifted = channelShifted->Clone();
            Ctext mergedRow;
            for(int ht = 0; ht < inputWidth; ht++){
                if(ht > 0){
                    rowShifted = context->EvalRotate(rowShifted, 1);
                }
                Ptext slot0Mask = gen_slot0_mask(rowShifted->GetLevel());
                Ctext isolated = context->EvalMult(rowShifted, slot0Mask);

                int idx = ht * dilateFactor;
                if(idx == 0){
                    mergedRow = isolated;
                } else {
                    mergedRow = context->EvalAdd(mergedRow, context->EvalRotate(isolated, -idx));
                }
            }

            int rowsShift = (vt * dilateFactor) * rowSize;
            if(rowsShift == 0){
                rowCtxt = mergedRow;
            } else {
                rowCtxt = context->EvalAdd(rowCtxt, context->EvalRotate(mergedRow, -rowsShift));
            }
        }

        int batchShift = bt * dilatedChannelSize;
        if(batchShift == 0){
            dilatedCtxt = rowCtxt;
        } else {
            dilatedCtxt = context->EvalAdd(dilatedCtxt, context->EvalRotate(rowCtxt, -batchShift));
        }
    }
    return dilatedCtxt;
}

/***
 * Rotation-index set for backward_dilate_optimized. Structurally the same
 * batch-shift/row-shift/row-step indices backward_dilate_indices computes
 * (inputWidth, channelSize, the -vt*dilateFactor*rowSize row shifts, the
 * -bt*dilatedChannelSize batch shifts), PLUS the "1" single-step column
 * rotation and the "-ht*dilateFactor" per-column placement shifts this
 * algorithm actually performs (backward_dilate never needs those directly --
 * EvalMerge handles that internally with its own key requirements).
 */
vector<int> FHEONTrainBackwardPassController::backward_dilate_optimized_indices(int batchSize, int inputWidth, int dilateFactor){
    vector<int> keys_positions;
    int rowSize = dilateFactor * inputWidth;
    int channelSize = inputWidth * inputWidth;
    int dilatedChannelSize = rowSize * rowSize;

    keys_positions.push_back(inputWidth);
    keys_positions.push_back(channelSize);
    keys_positions.push_back(1);

    for(int ht = 1; ht < inputWidth; ht++){
        keys_positions.push_back(-(ht * dilateFactor));
    }
    for(int vt = 1; vt < inputWidth; vt++){
        keys_positions.push_back(-(vt * dilateFactor * rowSize));
    }
    for(int bt = 1; bt < batchSize; bt++){
        keys_positions.push_back(-(bt * dilatedChannelSize));
    }

    std::sort(keys_positions.begin(), keys_positions.end());
    auto new_end = std::remove(keys_positions.begin(), keys_positions.end(), 0);
    new_end = std::unique(keys_positions.begin(), new_end);
    keys_positions.erase(new_end, keys_positions.end());
    return keys_positions;
}

/***
 * Same as backward_pass_avgpool, but built on backward_dilate_optimized
 * instead of backward_dilate -- see that function's doc comment for why
 * it's cheaper. Mathematically identical output; kept as a separate
 * function (not a swap-in replacement) so the original stays available as
 * the known-good reference -- see validate_avgpool_optimized in
 * HETrainingModels/TrainingModules.cpp for a direct numeric comparison
 * against backward_pass_avgpool on identical inputs.
 */
vector<Ctext> FHEONTrainBackwardPassController::backward_pass_avgpool_optimized(vector<Ctext> &encryptedInputs, int batchSize, int inputWidth, int dilateFactor) {

    if (dilateFactor != 2) {
        throw std::runtime_error("backward_pass_avgpool_optimized: the 4-term adjoint sum below is hardcoded for a 2x2 pooling window (dilateFactor=2); generalize the shift set {0,-1,-w,-(w+1)} before using a different dilateFactor.");
    }

    auto start_time = startTime();
    vector<Ctext> pooledGradients(encryptedInputs.size());
    int dilatedWidth = dilateFactor * inputWidth;
    double scaler = 1.0 / (dilateFactor * dilateFactor);

        run_parallel_for(encryptedInputs.size(), [&](int start, int end) {
            for(int i=start; i<end; i++){
                Ctext dilatedCtxt = backward_dilate_optimized(encryptedInputs[i], batchSize, inputWidth, dilateFactor);

                vector<Ctext> shifted;
                shifted.push_back(dilatedCtxt);
                shifted.push_back(context->EvalRotate(dilatedCtxt, -1));
                shifted.push_back(context->EvalRotate(dilatedCtxt, -dilatedWidth));
                shifted.push_back(context->EvalRotate(dilatedCtxt, -(dilatedWidth + 1)));
                Ctext smeared = context->EvalAddMany(shifted);

                pooledGradients[i] = context->EvalMult(smeared, scaler);
            }
        });
        printTiming("Backward average pooling (optimized) evaluated...", "backward_avgpool", start_time);
    return pooledGradients;
}

/***
 * Same as backward_pass_global_avgpool, but built on
 * backward_dilate_optimized instead of backward_dilate. Mathematically
 * identical output; see backward_pass_avgpool_optimized's doc comment for
 * why it's kept as a separate function, and
 * validate_global_avgpool_optimized in HETrainingModels/TrainingModules.cpp
 * for the numeric comparison against backward_pass_global_avgpool.
 */
vector<Ctext> FHEONTrainBackwardPassController::backward_pass_global_avgpool_optimized(vector<Ctext> &gradInputs, int batchSize, int dataWidth) {

    int spatialSize = dataWidth * dataWidth;
    double scaler = 1.0 / spatialSize;
    vector<Ctext> result(gradInputs.size());
    
    auto start_time = startTime();
    run_parallel_for(gradInputs.size(), [&](int start, int end) {
        for (int oc = start; oc < end; oc++) {
            Ctext dilated = backward_dilate_optimized(gradInputs[oc], batchSize, 1, dataWidth);
            Ctext smeared = repeat_encrypted_values(dilated, 1, spatialSize);
            result[oc] = context->EvalMult(smeared, scaler);
        }
    });
    printTiming("Backward global average pooling (optimized) evaluated...", "backward_global_avgpool", start_time);
    return result;
}

Ptext FHEONTrainBackwardPassController::gen_prefix_mask(int prefixLen, uint32_t level){
    size_t slots = context->GetRingDimension() / 2;
    vector<double> maskVec(slots, 0.0);
    for (int i = 0; i < prefixLen && i < (int)slots; i++) {
        maskVec[i] = 1.0;
    }
    return context->MakeCKKSPackedPlaintext(maskVec, 1, level);
}

Ptext FHEONTrainBackwardPassController::gen_binary_run_mask(int pattern, uint32_t level){
    size_t slots = context->GetRingDimension() / 2;
    vector<double> maskVec(slots, 0.0);
    int period = 2 * pattern;
    for (size_t i = 0; i < slots; i++) {
        if ((int)(i % period) < pattern) {
            maskVec[i] = 1.0;
        }
    }
    return context->MakeCKKSPackedPlaintext(maskVec, 1, level);
}

Ptext FHEONTrainBackwardPassController::gen_stride_mask(int period, uint32_t level){
    size_t slots = context->GetRingDimension() / 2;
    vector<double> maskVec(slots, 0.0);
    for (size_t i = 0; i < slots; i += period) {
        maskVec[i] = 1.0;
    }
    return context->MakeCKKSPackedPlaintext(maskVec, 1, level);
}

/***
 * Log-depth column scatter -- the core piece backward_dilate_logdepth builds
 * on. Takes a row whose inputWidth REAL values sit contiguously at slots
 * [0, inputWidth) (everything else on the ciphertext is irrelevant/garbage,
 * masked away by the first step below) and produces a row of the same width
 * with those values moved to slots {0, dilateFactor, 2*dilateFactor, ...},
 * zero elsewhere -- exactly what a single row of backward_dilate/
 * backward_dilate_optimized computes, just via O(log2(inputWidth)) chained
 * mask/rotate/add steps instead of O(inputWidth) independent ones.
 *
 * Derivation: this is the ADJOINT of FHEONANNController::
 * downsample_with_multiple_channels's own column-COMPACTION network (the
 * masked-doubling loop that gathers every stride-th element of a row into a
 * contiguous prefix -- see that function's implementation). That forward
 * network is a composition of simple linear steps:
 *   x0 = firstmask_d * raw          (d = stride, a diagonal projection)
 *   x_s = mask(2^s) * (x_{s-1} + rotate(x_{s-1}, 2^(s-1)))   for s=1..L-1
 *   x_L = x_{L-1} + rotate(x_{L-1}, 2^(L-1))                  (final merge)
 * (L = log2(outputWidth)). Transposing a composition reverses step order and
 * adjoints each step; a diagonal mask is self-adjoint, and (I + Rotate_k)'s
 * adjoint is (I + Rotate_{-k}) (rotation matrices are orthogonal, so
 * Rotate_k^T = Rotate_{-k}). Working through the algebra for one step:
 *   x_s = mask(2^s) * (I + Rotate_{2^(s-1)}) x_{s-1}
 *   grad_{x_{s-1}} = (I + Rotate_{-2^(s-1)}) (mask(2^s) * grad_{x_s})
 *                  = mask(2^s)*grad_{x_s} + rotate(mask(2^s)*grad_{x_s}, -2^(s-1))
 * i.e. mask FIRST, then add the negatively-rotated copy of the masked
 * result -- mirroring the forward step's shape with the rotation sign
 * flipped and the mask moved to before the add. Chaining these (L steps,
 * reversed order) and re-using this exact machinery with "stride" set to
 * dilateFactor (this function's own d, matching backward_dilate's own
 * dilateFactor parameter, standing in for downsample's stride in this
 * transposed reading) gives dilate directly -- no separate "downsample
 * problem" needs to actually run.
 *
 * Verified by hand-tracing inputWidth=4, dilateFactor=2 before implementing
 * (produces v0,0,v1,0,v2,0,v3,0 as expected) and by
 * validate_avgpool_logdepth/validate_global_avgpool_logdepth's numeric
 * comparison against the known-good backward_pass_avgpool/
 * backward_pass_global_avgpool at real network dimensions.
 *
 * Real tradeoff vs backward_dilate_optimized: fewer total operations, but
 * MORE consumed levels -- backward_dilate_optimized's per-column mask-mults
 * each branch off a fresh copy of the row (net +1 level regardless of
 * inputWidth), while this chains log2(inputWidth)+1 mask-mults sequentially
 * (net +(log2(inputWidth)+1) levels). See the validators' printed levels
 * for the actual number at real network widths before deciding which to use
 * where -- this may need an extra bootstrap nearby that
 * backward_dilate_optimized didn't.
 *
 * inputWidth must be a power of two (only shapes this codebase actually
 * uses -- see backward_dilate_logdepth's own doc comment).
 */
Ctext FHEONTrainBackwardPassController::column_scatter_log_depth(Ctext& rowInput, int inputWidth, int dilateFactor){
    if (inputWidth <= 0 || (inputWidth & (inputWidth - 1)) != 0) {
        throw std::runtime_error("column_scatter_log_depth: inputWidth must be a power of two");
    }

    Ctext g = context->EvalMult(rowInput, gen_prefix_mask(inputWidth, rowInput->GetLevel()));

    if (inputWidth > 1) {
        int L = 0;
        while ((1 << L) < inputWidth) L++;   // L = log2(inputWidth)

        g = context->EvalAdd(g, context->EvalRotate(g, -(1 << (L - 1))));
        for (int s = L - 1; s >= 1; s--) {
            g = context->EvalMult(g, gen_binary_run_mask(1 << s, g->GetLevel()));
            g = context->EvalAdd(g, context->EvalRotate(g, -(1 << (s - 1))));
        }
    }

    g = context->EvalMult(g, gen_stride_mask(dilateFactor, g->GetLevel()));
    return g;
}

/***
 * Log-depth variant of backward_dilate/backward_dilate_optimized -- same
 * row/batch shift-and-accumulate structure as both (already O(inputWidth)/
 * O(batchSize), see backward_dilate_optimized's doc comment for why that
 * part isn't the bottleneck), but the within-row column placement now goes
 * through column_scatter_log_depth instead of either EvalMerge
 * (backward_dilate) or an O(inputWidth) direct accumulate
 * (backward_dilate_optimized). See column_scatter_log_depth's doc comment
 * for the derivation and the level-cost tradeoff.
 */
Ctext FHEONTrainBackwardPassController::backward_dilate_logdepth(Ctext& encryptedInput, int batchSize, int inputWidth, int dilateFactor){

    Ctext rowCtxt;
    Ctext dilatedCtxt;
    int rowSize = dilateFactor * inputWidth;
    int channelSize = inputWidth * inputWidth;
    int dilatedChannelSize = rowSize * rowSize;

    Ctext dilatedInput = encryptedInput->Clone();
    for(int bt = 0; bt < batchSize; bt++){
        if(bt > 0){
            dilatedInput = context->EvalRotate(dilatedInput, channelSize);
        }
        Ctext channelShifted = dilatedInput->Clone();

        for(int vt=0; vt < inputWidth; vt++){
            if(vt > 0){
                channelShifted = context->EvalRotate(channelShifted, inputWidth);
            }

            Ctext mergedRow = column_scatter_log_depth(channelShifted, inputWidth, dilateFactor);

            int rowsShift = (vt * dilateFactor) * rowSize;
            if(rowsShift == 0){
                rowCtxt = mergedRow;
            } else {
                rowCtxt = context->EvalAdd(rowCtxt, context->EvalRotate(mergedRow, -rowsShift));
            }
        }

        int batchShift = bt * dilatedChannelSize;
        if(batchShift == 0){
            dilatedCtxt = rowCtxt;
        } else {
            dilatedCtxt = context->EvalAdd(dilatedCtxt, context->EvalRotate(rowCtxt, -batchShift));
        }
    }
    return dilatedCtxt;
}

/***
 * Rotation-index set for backward_dilate_logdepth. Same row-shift/batch-shift
 * indices backward_dilate_optimized_indices computes, plus
 * column_scatter_log_depth's own doubling rotations: -1, -2, -4, ...,
 * -(inputWidth/2) (used once each regardless of how many rows/batches reuse
 * the key -- see column_scatter_log_depth's doc comment for exactly where
 * each is used).
 */
vector<int> FHEONTrainBackwardPassController::backward_dilate_logdepth_indices(int batchSize, int inputWidth, int dilateFactor){
    vector<int> keys_positions;
    int rowSize = dilateFactor * inputWidth;
    int channelSize = inputWidth * inputWidth;
    int dilatedChannelSize = rowSize * rowSize;

    keys_positions.push_back(inputWidth);
    keys_positions.push_back(channelSize);

    for (int p = 1; p < inputWidth; p *= 2) {
        keys_positions.push_back(-p);
    }

    for (int vt = 1; vt < inputWidth; vt++) {
        keys_positions.push_back(-(vt * dilateFactor * rowSize));
    }
    for (int bt = 1; bt < batchSize; bt++) {
        keys_positions.push_back(-(bt * dilatedChannelSize));
    }

    std::sort(keys_positions.begin(), keys_positions.end());
    auto new_end = std::remove(keys_positions.begin(), keys_positions.end(), 0);
    new_end = std::unique(keys_positions.begin(), new_end);
    keys_positions.erase(new_end, keys_positions.end());
    return keys_positions;
}

/***
 * Same as backward_pass_avgpool/backward_pass_avgpool_optimized, built on
 * backward_dilate_logdepth. See that function's doc comment for the
 * level-cost tradeoff before using this over backward_pass_avgpool_optimized
 * in a level-constrained context.
 */
vector<Ctext> FHEONTrainBackwardPassController::backward_pass_avgpool_logdepth(vector<Ctext> &encryptedInputs, int batchSize, int inputWidth, int dilateFactor) {

    if (dilateFactor != 2) {
        throw std::runtime_error("backward_pass_avgpool_logdepth: the 4-term adjoint sum below is hardcoded for a 2x2 pooling window (dilateFactor=2); generalize the shift set {0,-1,-w,-(w+1)} before using a different dilateFactor.");
    }

    auto start_time = startTime();
    vector<Ctext> pooledGradients(encryptedInputs.size());
    int dilatedWidth = dilateFactor * inputWidth;
    double scaler = 1.0 / (dilateFactor * dilateFactor);

        run_parallel_for(encryptedInputs.size(), [&](int start, int end) {
            for(int i=start; i<end; i++){
                Ctext dilatedCtxt = backward_dilate_logdepth(encryptedInputs[i], batchSize, inputWidth, dilateFactor);

                vector<Ctext> shifted;
                shifted.push_back(dilatedCtxt);
                shifted.push_back(context->EvalRotate(dilatedCtxt, -1));
                shifted.push_back(context->EvalRotate(dilatedCtxt, -dilatedWidth));
                shifted.push_back(context->EvalRotate(dilatedCtxt, -(dilatedWidth + 1)));
                Ctext smeared = context->EvalAddMany(shifted);

                pooledGradients[i] = context->EvalMult(smeared, scaler);
            }
        });
        printTiming("Backward average pooling (log-depth) evaluated...", "backward_avgpool", start_time);
    return pooledGradients;
}

/***
 * Same as backward_pass_global_avgpool/backward_pass_global_avgpool_optimized,
 * built on backward_dilate_logdepth. Note inputWidth=1 in this call (see
 * backward_pass_global_avgpool's doc comment for why), so
 * column_scatter_log_depth skips its doubling loop entirely (L=0) -- just
 * the prefix mask then the stride mask, 2 chained levels, vs
 * backward_pass_global_avgpool_optimized's 1 (a single isolate-mult).
 */
vector<Ctext> FHEONTrainBackwardPassController::backward_pass_global_avgpool_logdepth(vector<Ctext> &gradInputs, int batchSize, int dataWidth) {

    auto start_time = startTime();
    int spatialSize = dataWidth * dataWidth;
    double scaler = 1.0 / spatialSize;

    vector<Ctext> result(gradInputs.size());
        run_parallel_for(gradInputs.size(), [&](int start, int end) {
            for (int oc = start; oc < end; oc++) {
        Ctext dilated = backward_dilate_logdepth(gradInputs[oc], batchSize, 1, dataWidth);
        Ctext smeared = repeat_encrypted_values(dilated, 1, spatialSize);
        result[oc] = context->EvalMult(smeared, scaler);
            }
        });
        printTiming("Backward global average pooling (log-depth) evaluated...", "backward_global_avgpool", start_time);
    return result;
}

Ctext FHEONTrainBackwardPassController::taylors_approximation(Ctext& encryptedInput, int appxDegree) {

    if (appxDegree <= 0) {
        throw std::runtime_error("Degree must be > 0");
    }

    // powers[i] stores x^i
    vector<Ctext> powers(appxDegree);
    powers[0] = context->EvalSub(encryptedInput, 1.0);

    // Generate powers iteratively using reuse computed powers
    for (int i = 1; i < appxDegree; i++) {
        int degree = i + 1;
        int a = degree / 2;
        int b = degree - a;
        powers[i] = context->EvalMult(powers[a-1], powers[b-1]);
    }

    // result = x
    Ctext result = powers[0]->Clone();
    // Sum Taylor terms
    for (int i = 1; i < appxDegree; i++) {
        double scaler = 1.0 / (i + 1);
        if(i % 2 == 1){
            scaler = -scaler;
        }
        result = context->EvalAdd(result, context->EvalMult(powers[i], scaler));
    }
    return result;
}
