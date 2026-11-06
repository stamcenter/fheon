
/************************************************************************************************
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
**********************************************************************************************/

/** Encrypted-domain fine-tuning driver for the CIFAR ResNet20 model. */

#include <iostream>
#include <sys/stat.h>
#include <filesystem>

#include "FHEONHEController.h"
#include "FHEONANNController.h"
#include "FHEONTrainSupportController.h"
#include "FHEONTrainStandardForwardPassController.h"
#include "FHEONTrainStandardBackwardPassController.h"

using namespace std;

#ifndef DEFAULT_TEST_SIZE
#define DEFAULT_TEST_SIZE 16
#endif

#ifndef INDEX_VALUE
#define INDEX_VALUE 0
#endif

CryptoContext<DCRTPoly> context;
FHEONHEController fheonHEController(context);
FHEONANNController fheonANNController(context);
FHEONTrainSupportController fheonTrainSupportController(context);
/** the standard training controllers (same layers, files and keys as FHEONTrainForwardPassController /
 *  FHEONTrainBackwardPassController, multithreaded) with the precision fixes: 2-iteration bootstrapping with its
 *  precision argument, batch norm started per channel from the running variances, softmax by repeated squaring */
FHEONTrainStandardForwardPassController fheonTrainForwardPassController(context);
FHEONTrainStandardBackwardPassController fheonTrainBackwardPassController(context);

int batchSize = 16;

// channels[0..4] = {stem_in, stage1_out, stage2_out, stage3_out, num_classes}
vector<int> channels = {3, 16, 32, 64, 10};
// dataWidths[0..3] = spatial width at stem/stage1, stage2, stage3, post-globalpool
vector<int> dataWidths = {32, 16, 8, 1};
vector<int> dataSizes = {1024, 256, 64}; // dataWidths[0..2] squared

int kernelWidth = 3;
int paddingLen = 1;
int polyDegree = 59;
int newtonApproxDegree = 3;
int taylorApproxDegree = 3;
int bootstrap_level = 2;
int numClasses = 10;
double learningRate = 0.01;

// Placeholder until real per-layer activation variance is measured (see file
// header) -- TrainModel.cpp's own BN initVal constants were only tuned after
// inspecting real data through the same debugging process this session used.
/** fallback start of Newton's 1/sqrt(var) when a layer has no <layer>_running_var.csv (a constant start only converges
 *  near var = 1 / start^2: the batch variances of the pretrained network span 1e-13 ... 2) */
double bnInitVal = 0.5;
/** batch norm epsilon (PyTorch's default) */
double bnEpsilon = 1e-5;
/** bound on |logit| for the softmax by repeated squaring (the pretrained logits reach ~16) */
double logitBound = 32.0;
/** precision (bits) of the first of the 2 bootstrapping iterations: 0 (OpenFHE's default) leaves ~8e-3 error on deep
 *  ciphertexts at these parameters, 15 gives ~3e-7 */
int bootstrapPrecision = 15;
vector<int> printingCtxt = {1, 16};

static inline void printStep(const string& label) {
    cout << "\n[" << label << "]" << endl;
}

static inline void printBlock(const string& label) {
    cout << "\n========== " << label << " ==========" << endl;
}

static inline void bind_training_contexts(CryptoContext<DCRTPoly>& activeContext) {
    fheonANNController.setContext(activeContext);
    fheonTrainSupportController.setContext(activeContext);
    fheonTrainForwardPassController.setContext(activeContext);
    fheonTrainBackwardPassController.setContext(activeContext);
}

/**
 * Batch norm with Newton's 1/sqrt(var + eps) of every channel started from the pretrained running variance
 * (weights_folder + <layer>_running_var.csv, see python/training/export_tuneresnet20_weights.py). Falls back to the
 * constant bnInitVal (which does not converge over the range of the batch variances) when the file is missing.
 */
vector<Ctext> batch_norm_forward(vector<Ctext>& x, int bnSize, const string& layerName, bool isBootstrap, bool pretrainedFromCsv) {
    string path = weights_folder + layerName + "_running_var.csv";
    vector<double> runningVar;
    if (std::filesystem::exists(path)) runningVar = utilsbatchdata::flatten_csv_data(utilsbatchdata::loadCSV(path));
    if (runningVar.size() < x.size()) {
        cout << "  [warning] " << path << " missing: batch norm Newton starts from the constant " << bnInitVal << endl;
        return fheonTrainForwardPassController.he_forward_batch_norm(x, bnSize, bnInitVal, newtonApproxDegree, layerName, isBootstrap, pretrainedFromCsv);
    }
    vector<double> inverseStdInit(x.size());
    for (size_t c = 0; c < x.size(); c++) inverseStdInit[c] = 1.0 / std::sqrt(std::max(0.0, runningVar[c]) + bnEpsilon);
    return fheonTrainForwardPassController.he_forward_batch_norm(x, bnSize, inverseStdInit, newtonApproxDegree, layerName, isBootstrap,
                                                                 pretrainedFromCsv, bnEpsilon);
}

/** Forward pass for one ResNet basic block, including its cached activations. */
vector<Ctext> resnet_block_forward(vector<Ctext>& input, string blockName, int batchSz, int inputWidth,
                                     int inputChannels, int outputChannels, bool downsample) {
                                        
    int stride = downsample ? 2 : 1;
    int outputWidth = inputWidth / stride;
    int bnSize = batchSz * outputWidth * outputWidth;

    auto freshInput = fheonHEController.batch_bootstrap_function(input, inputChannels, bootstrap_level);

    cout << "[resnet_block_forward] " << blockName << " (inCh=" << inputChannels
         << ", outCh=" << outputChannels << ", stride=" << stride
         << ", inW=" << inputWidth << ", outW=" << outputWidth
         << ", downsample=" << downsample << ")" << endl;

    auto x = fheonTrainForwardPassController.he_forward_convolution_optimized(freshInput, batchSz, inputWidth, inputChannels,
                                                                    outputChannels, stride, blockName + "_conv1", true);
    x = fheonHEController.batch_bootstrap_function(x, outputChannels, bootstrap_level);
    x = batch_norm_forward(x, bnSize, blockName + "_bn1", true, false);
    x = fheonHEController.batch_bootstrap_function(x, outputChannels, bootstrap_level);
    auto scales = fheonHEController.read_batch_scaling_values(x, outputChannels, bnSize);
    x = fheonTrainForwardPassController.he_forward_relu(x, scales, outputChannels, bnSize, polyDegree, blockName + "_relu1");

    x = fheonHEController.batch_bootstrap_function(x, outputChannels, bootstrap_level);
    x = fheonTrainForwardPassController.he_forward_convolution_optimized(x, batchSz, outputWidth, outputChannels, 
                                                        outputChannels, 1, blockName + "_conv2", true);
    x = fheonHEController.batch_bootstrap_function(x, outputChannels, bootstrap_level);
    x = batch_norm_forward(x, bnSize, blockName + "_bn2", true, true);

    cout << "[resnet_block_forward] " << blockName << " first set finished" << endl;

    vector<Ctext> shortcut;
    if (downsample) {
        shortcut = fheonTrainForwardPassController.he_forward_convolution_shortcut_optimized(freshInput, batchSz, inputWidth,
                                                                inputChannels, outputChannels, stride, blockName + "_shortcut", true);
        shortcut = fheonHEController.batch_bootstrap_function(shortcut, outputChannels, bootstrap_level);
        shortcut = batch_norm_forward(shortcut, bnSize, blockName + "_bn_shortcut", true, true);
    } else {
        shortcut = freshInput; // identity shortcut: inputChannels == outputChannels, inputWidth == outputWidth
    }

    vector<Ctext> sum(outputChannels);
    for (int oc = 0; oc < outputChannels; oc++) {
        sum[oc] = context->EvalAdd(x[oc], shortcut[oc]);
    }
    sum = fheonHEController.batch_bootstrap_function(sum, outputChannels, bootstrap_level);
    scales = fheonHEController.read_batch_scaling_values(sum, outputChannels, bnSize);
    // he_forward_relu caches its own input (the residual sum) under blockName+"_relu2" --
    // no dedicated "residual sum" primitive/cache is needed for the backward split below.
    auto out = fheonTrainForwardPassController.he_forward_relu(sum, scales, outputChannels, bnSize, polyDegree, blockName + "_relu2");

    return out;
}


/** Backward pass for one ResNet basic block, including both residual branches. */
vector<Ctext> resnet_block_backward(vector<Ctext>& gradOutput, string blockName, int batchSz, int inputWidth,
                                      int inputChannels, int outputChannels, bool downsample) {
    int stride = downsample ? 2 : 1;
    int outputWidth = inputWidth / stride;
    int bnSize = batchSz * outputWidth * outputWidth;

    auto freshGradOutput = fheonHEController.batch_bootstrap_function(gradOutput, outputChannels, bootstrap_level);

    cout << "[resnet_block_backward] " << blockName << " (inCh=" << inputChannels
         << ", outCh=" << outputChannels << ", stride=" << stride
         << ", inW=" << inputWidth << ", outW=" << outputWidth
         << ", downsample=" << downsample << ")" << endl;

    vector<Ctext> relu2ForwardAct(outputChannels);
    for (int oc = 0; oc < outputChannels; oc++) {
        string fname = blockName + "_relu2_channel_" + to_string(oc);
        relu2ForwardAct[oc] = fheonTrainBackwardPassController.deserialize_and_load(fname);
    }
    auto scales = fheonHEController.read_batch_scaling_values(relu2ForwardAct, outputChannels, bnSize);
    auto dSum = fheonTrainBackwardPassController.backward_pass_relu(freshGradOutput, scales, polyDegree, blockName + "_relu2");
    dSum = fheonHEController.batch_bootstrap_function(dSum, outputChannels, bootstrap_level);

    // The residual sum sends the same gradient to the main and shortcut branches.
    auto dBn2 = fheonTrainBackwardPassController.backward_pass_batch_norm(dSum, dSum, bnSize, learningRate, blockName + "_bn2");
    dBn2 = fheonHEController.batch_bootstrap_function(dBn2, outputChannels, bootstrap_level);
    auto dConv2 = fheonTrainBackwardPassController.backward_pass_convolution(
        dBn2, batchSz, outputWidth, outputChannels, outputChannels, kernelWidth, learningRate, blockName + "_conv2", 1);
    dConv2 = fheonHEController.batch_bootstrap_function(dConv2, outputChannels, bootstrap_level);

    vector<Ctext> relu1ForwardAct(outputChannels);
    for (int oc = 0; oc < outputChannels; oc++) {
        string fname = blockName + "_relu1_channel_" + to_string(oc);
        relu1ForwardAct[oc] = fheonTrainBackwardPassController.deserialize_and_load(fname);
    }
    scales = fheonHEController.read_batch_scaling_values(relu1ForwardAct, outputChannels, bnSize);
    auto dRelu1 = fheonTrainBackwardPassController.backward_pass_relu(dConv2, scales, polyDegree, blockName + "_relu1");
    dRelu1 = fheonHEController.batch_bootstrap_function(dRelu1, outputChannels, bootstrap_level);
    auto dBn1 = fheonTrainBackwardPassController.backward_pass_batch_norm(dRelu1, dRelu1, bnSize, learningRate, blockName + "_bn1");
    dBn1 = fheonHEController.batch_bootstrap_function(dBn1, outputChannels, bootstrap_level);
    auto dConv1 = fheonTrainBackwardPassController.backward_pass_convolution(dBn1, batchSz, inputWidth, inputChannels, outputChannels, kernelWidth, 
                                                        learningRate, blockName + "_conv1", stride);

    // Shortcut branch
    vector<Ctext> dShortcutInput;
    if (downsample) {
        auto dBnShortcut = fheonTrainBackwardPassController.backward_pass_batch_norm(dSum, dSum, bnSize, learningRate, blockName + "_bn_shortcut");
        dBnShortcut = fheonHEController.batch_bootstrap_function(dBnShortcut, outputChannels, bootstrap_level);
        dShortcutInput = fheonTrainBackwardPassController.backward_pass_convolution_shortcut(
            dBnShortcut, batchSz, inputWidth, inputChannels, outputChannels, stride, learningRate, blockName + "_shortcut");
    } else {
        dShortcutInput = dSum; // identity shortcut: gradient flows straight through unchanged
    }

    vector<Ctext> gradInput(inputChannels);
    for (int ic = 0; ic < inputChannels; ic++) {
        gradInput[ic] = context->EvalAdd(dConv1[ic], dShortcutInput[ic]);
    }
    return gradInput;
}




int main(int argc, char *argv[]) {

    auto runtime_config = parse_runtime_args(argc, argv, DEFAULT_TEST_SIZE, INDEX_VALUE);
    int index_value = runtime_config.index_value;


    auto begin_time = startTime();
    printWelcomeMessage();
    cout << "Initializing FHEON ResNet20 Fine-Tuning Controller..." << endl;

    FHEONHEController::HEConfig heinitConfig;
    heinitConfig.ringDim = 15;
    heinitConfig.numSlots = 14;
    heinitConfig.mlevelBootstrap = 12;
    heinitConfig.dcrtBits = 46;
    heinitConfig.firstMod = 50;
    heinitConfig.numDigits = 4;
    heinitConfig.levelBudget = {3, 3};
    heinitConfig.serialize = true;
    heinitConfig.keysFolder = "../AllHEKeys/";
    if (!runtime_config.keys_folder.empty()) {
        heinitConfig.keysFolder = runtime_config.keys_folder;
    }

    fheonHEController.load_context(heinitConfig, runtime_config.load_context);
    printDuration(begin_time, runtime_config.load_context ? "Context Loaded" : "Context Generated and Keys Serialization", false);

    context = fheonHEController.getContext();
    bind_training_contexts(context);

    /** precise 2-iteration bootstrapping everywhere: the model's batch bootstrapping and the controllers' own */
    fheonHEController.set_bootstrap_precision(bootstrapPrecision);
    fheonTrainForwardPassController.setBootstrapPrecision(bootstrapPrecision);
    fheonTrainBackwardPassController.setBootstrapPrecision(bootstrapPrecision);
    auto groupBootstrap = [](vector<Ctext>& cts) { return fheonHEController.batch_bootstrap_function(cts, cts.size(), bootstrap_level); };
    fheonTrainForwardPassController.setBootstrapFunction(groupBootstrap);
    fheonTrainBackwardPassController.setBootstrapFunction(groupBootstrap);

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
    fheonTrainBackwardPassController.setHEController(&fheonHEController);

    /* Keep all rotation families in one keyset and CryptoContext. */
    printBlock("Rotation Key Generation");
    auto keyGenStart = startTime();

    // Rotation keys for each convolution width and stride.
    auto width32_stride1_keys = fheonTrainSupportController.generate_convolution_batch_rotation_positions(batchSize, dataWidths[0], kernelWidth, paddingLen, 1);
    auto width32_stride2_keys = fheonTrainSupportController.generate_convolution_batch_rotation_positions(batchSize, dataWidths[0], kernelWidth, paddingLen, 2);
    auto width16_stride1_keys = fheonTrainSupportController.generate_convolution_batch_rotation_positions(batchSize, dataWidths[1], kernelWidth, paddingLen, 1);
    auto width16_stride2_keys = fheonTrainSupportController.generate_convolution_batch_rotation_positions(batchSize, dataWidths[1], kernelWidth, paddingLen, 2);
    auto width8_stride1_keys = fheonTrainSupportController.generate_convolution_batch_rotation_positions(batchSize, dataWidths[2], kernelWidth, paddingLen, 1);

    // Rotation keys for downsampling and global-pool backward dilation.
    auto dilate_stage2_keys = fheonTrainBackwardPassController.backward_dilate_indices(batchSize, dataWidths[1], 2);
    auto dilate_stage3_keys = fheonTrainBackwardPassController.backward_dilate_indices(batchSize, dataWidths[2], 2);
    auto dilate_globalpool_keys = fheonTrainBackwardPassController.backward_dilate_indices(batchSize, 1, dataWidths[2]);

    // Log-depth dilation keys used by strided convolution backward passes.
    auto dilate_logdepth_stage2_keys = fheonTrainBackwardPassController.backward_dilate_logdepth_indices(batchSize, dataWidths[1], 2);
    auto dilate_logdepth_stage3_keys = fheonTrainBackwardPassController.backward_dilate_logdepth_indices(batchSize, dataWidths[2], 2);

    // Broadcast and global-pool smear keys.
    auto norm_stage1_keys = fheonTrainSupportController.generate_normalization_rotation_positions(batchSize * dataSizes[0]);
    auto norm_stage2_keys = fheonTrainSupportController.generate_normalization_rotation_positions(batchSize * dataSizes[1]);
    auto norm_stage3_keys = fheonTrainSupportController.generate_normalization_rotation_positions(batchSize * dataSizes[2]);
    auto norm_globalpool_keys = fheonTrainSupportController.generate_normalization_rotation_positions(dataSizes[2]);

    vector<vector<int>> allKeyFamilies = {
        width32_stride1_keys, width32_stride2_keys, width16_stride1_keys,
        width16_stride2_keys, width8_stride1_keys,
        dilate_stage2_keys, dilate_stage3_keys, dilate_globalpool_keys,
        dilate_logdepth_stage2_keys, dilate_logdepth_stage3_keys,
        norm_stage1_keys, norm_stage2_keys, norm_stage3_keys, norm_globalpool_keys
    };
    auto allRotationKeys = serialize_rotation_keys(allKeyFamilies);
    cout << "  combined rotation keys: " << allRotationKeys.size() << endl;
    fheonHEController.generate_bootstrapping_and_rotation_keys(
        allRotationKeys, heinitConfig.numSlots, "resnet20_all.bin", true, true);
    printDuration(keyGenStart, "Rotation and Bootstrapping Keys Generation", false);

    fheonTrainSupportController.welcomeToTrainingMessage();

    string dataPath = "./../exported_samples";
    ofstream predictedOutputsFile;
    predictedOutputsFile.open("./../results/Training/resnet20_predicted_outputs.txt", ios_base::app);

    printBlock("Batch Data Load + Encryption");
    auto batchedImages = readBatchCIFAR10Images(dataPath, index_value, batchSize, true);
    auto batchedLabels = readBatchedCIFAR10Labels(dataPath, index_value, batchSize);
    auto inputDatas = convert_inputData(batchedImages, batchSize, channels[0], dataSizes[0]);
    cout << "   Images: " << batchedImages.size() << "   Channels: " << inputDatas.size()
         << "   Labels: " << batchedLabels << endl;
    auto encryptedInputs = fheonHEController.encrypt_multiple_inputs(inputDatas);

    /** forward pass **/
    printBlock("Stem: Conv0 -> BN0 -> ReLU0");
    auto forwardData = fheonTrainForwardPassController.he_forward_convolution_optimized(encryptedInputs, batchSize, dataWidths[0], channels[0], 
                                                            channels[1], 1, "layer0_conv1", true);
    int stemBnSize = batchSize * dataSizes[0];
    forwardData = batch_norm_forward(forwardData, stemBnSize, "layer0_bn1", false, true);
    auto scales = fheonHEController.read_batch_scaling_values(forwardData, channels[1], stemBnSize);
    forwardData = fheonTrainForwardPassController.he_forward_relu(forwardData, scales, channels[1], stemBnSize, polyDegree, "layer0_relu1");
    fheonHEController.decrypt_batch_data(forwardData, printingCtxt[0], printingCtxt[1]);

    printBlock("Stage 1: 3 BasicBlocks (16->16, stride 1)");
    forwardData = resnet_block_forward(forwardData, "layer1_block1", batchSize, dataWidths[0], channels[1], channels[1], false);
    forwardData = resnet_block_forward(forwardData, "layer1_block2", batchSize, dataWidths[0], channels[1], channels[1], false);
    forwardData = resnet_block_forward(forwardData, "layer1_block3", batchSize, dataWidths[0], channels[1], channels[1], false);
    fheonHEController.decrypt_batch_data(forwardData, printingCtxt[0], printingCtxt[1]);

    printBlock("Stage 2: 3 BasicBlocks (16->32, first stride 2)");
    forwardData = resnet_block_forward(forwardData, "layer2_block1", batchSize, dataWidths[0], channels[1], channels[2], true);
    forwardData = resnet_block_forward(forwardData, "layer2_block2", batchSize, dataWidths[1], channels[2], channels[2], false);
    forwardData = resnet_block_forward(forwardData, "layer2_block3", batchSize, dataWidths[1], channels[2], channels[2], false);
    fheonHEController.decrypt_batch_data(forwardData, printingCtxt[0], printingCtxt[1]);

    printBlock("Stage 3: 3 BasicBlocks (32->64, first stride 2)");
    forwardData = resnet_block_forward(forwardData, "layer3_block1", batchSize, dataWidths[1], channels[2], channels[3], true);
    forwardData = resnet_block_forward(forwardData, "layer3_block2", batchSize, dataWidths[2], channels[3], channels[3], false);
    forwardData = resnet_block_forward(forwardData, "layer3_block3", batchSize, dataWidths[2], channels[3], channels[3], false);
    fheonHEController.decrypt_batch_data(forwardData, printingCtxt[0], printingCtxt[1]);

    printBlock("Global Average Pool + FC + Softmax + Loss");
    forwardData = fheonHEController.batch_bootstrap_function(forwardData, channels[3], bootstrap_level);
    forwardData = fheonTrainForwardPassController.he_forward_global_avgpool(forwardData, batchSize, dataWidths[2], channels[3], "global_avgpool");

    int fcVectorSize = batchSize; // post-globalpool, one value per image, no spatial width left
    forwardData = fheonTrainForwardPassController.he_forward_linear_multiple_outputs(forwardData, fcVectorSize, channels[3], channels[4], "layer_fc", true);
    fheonHEController.decrypt_batch_data(forwardData, printingCtxt[0], min(printingCtxt[1], batchSize));

    /** softmax by repeated squaring: 1 + x + x^2/2 is not monotonic for the negative logits (down to ~-8) */
    forwardData = fheonTrainForwardPassController.he_forward_softmax_normalized(forwardData, batchSize, logitBound);
    fheonHEController.decrypt_batch_data(forwardData, printingCtxt[0], min(printingCtxt[1], batchSize));

    printStep("Predicted Labels");
    fheonHEController.read_forward_inferenced_label_multiple_outputs(forwardData, batchSize, channels[4], predictedOutputsFile, 0);

    printStep("Loss Calculation");
    forwardData = fheonHEController.batch_bootstrap_function(forwardData, channels[4], bootstrap_level);
    auto loss = fheonTrainBackwardPassController.cross_entropy_loss(forwardData, batchedLabels, batchSize, numClasses, taylorApproxDegree);
    cout << "  Cross-entropy loss: ";
    fheonHEController.decrypt_and_print(loss, batchSize);

    printTotalDurations();

    /** backward pass **/
    printBlock("Backward Pass and Weight Updates");
    // bind_training_contexts(context);

    printStep("Backward - Cross-Entropy/Softmax gradient");
    auto gradInputs = fheonTrainBackwardPassController.backward_pass_cross_entropy_and_softmax(forwardData, batchedLabels, batchSize, fcVectorSize, "layer_fc");
    fheonHEController.decrypt_batch_data(gradInputs, printingCtxt[0], min(printingCtxt[1], batchSize));

    printStep("Backward - FC gradients");
    gradInputs = fheonTrainBackwardPassController.backward_pass_fully_connected(gradInputs, fcVectorSize, channels[3], channels[4], learningRate, "layer_fc");
    fheonHEController.decrypt_batch_data(gradInputs, printingCtxt[0], min(printingCtxt[1], batchSize));

    printStep("Backward - Global AvgPool gradients");
    gradInputs = fheonHEController.batch_bootstrap_function(gradInputs, channels[3], bootstrap_level);
    gradInputs = fheonTrainBackwardPassController.backward_pass_global_avgpool(gradInputs, batchSize, dataWidths[2]);
    fheonHEController.decrypt_batch_data(gradInputs, printingCtxt[0], printingCtxt[1]);

    




    printBlock("Stage 3 Backward (reverse order)");
    // resnet_block_backward bootstraps its own input.
    gradInputs = resnet_block_backward(gradInputs, "layer3_block3", batchSize, dataWidths[2], channels[3], channels[3], false);
    gradInputs = resnet_block_backward(gradInputs, "layer3_block2", batchSize, dataWidths[2], channels[3], channels[3], false);
    gradInputs = resnet_block_backward(gradInputs, "layer3_block1", batchSize, dataWidths[1], channels[2], channels[3], true);
    fheonHEController.decrypt_batch_data(gradInputs, printingCtxt[0], printingCtxt[1]);

    printBlock("Stage 2 Backward (reverse order)");
    gradInputs = resnet_block_backward(gradInputs, "layer2_block3", batchSize, dataWidths[1], channels[2], channels[2], false);
    gradInputs = resnet_block_backward(gradInputs, "layer2_block2", batchSize, dataWidths[1], channels[2], channels[2], false);
    gradInputs = resnet_block_backward(gradInputs, "layer2_block1", batchSize, dataWidths[0], channels[1], channels[2], true);
    fheonHEController.decrypt_batch_data(gradInputs, printingCtxt[0], printingCtxt[1]);

    printBlock("Stage 1 Backward (reverse order)");
    gradInputs = resnet_block_backward(gradInputs, "layer1_block3", batchSize, dataWidths[0], channels[1], channels[1], false);
    
    cout << endl << endl << endl;
    printTotalDurations();
    cout << endl << endl << endl;
    cout << "  Stage1 block3 finished" << endl;
    gradInputs = fheonHEController.batch_bootstrap_function(gradInputs, channels[1], bootstrap_level);
    fheonHEController.decrypt_batch_data(gradInputs, printingCtxt[0], printingCtxt[1]);


    gradInputs = resnet_block_backward(gradInputs, "layer1_block2", batchSize, dataWidths[0], channels[1], channels[1], false);
    gradInputs = resnet_block_backward(gradInputs, "layer1_block1", batchSize, dataWidths[0], channels[1], channels[1], false);
    fheonHEController.decrypt_batch_data(gradInputs, printingCtxt[0], printingCtxt[1]);


    cout << endl << endl << endl;
    printTotalDurations();
    cout << endl << endl << endl;


    printBlock("Stem Backward");
    vector<Ctext> relu0ForwardAct(channels[1]);
    for (int c = 0; c < channels[1]; c++) {
        string fname = "layer0_relu1_channel_" + to_string(c);
        relu0ForwardAct[c] = fheonTrainBackwardPassController.deserialize_and_load(fname);
    }
    scales = fheonHEController.read_batch_scaling_values(relu0ForwardAct, channels[1], stemBnSize);
    
    
    // Refresh the stem gradient before the ReLU backward pass.
    gradInputs = fheonHEController.batch_bootstrap_function(gradInputs, channels[1], bootstrap_level);
    gradInputs = fheonTrainBackwardPassController.backward_pass_relu(gradInputs, scales, polyDegree, "layer0_relu1");
    gradInputs = fheonHEController.batch_bootstrap_function(gradInputs, channels[1], bootstrap_level);
    gradInputs = fheonTrainBackwardPassController.backward_pass_batch_norm(gradInputs, gradInputs, stemBnSize, learningRate, "layer0_bn1");
    gradInputs = fheonHEController.batch_bootstrap_function(gradInputs, channels[1], bootstrap_level);
    gradInputs = fheonTrainBackwardPassController.backward_pass_convolution(
        gradInputs, batchSize, dataWidths[0], channels[0], channels[1], kernelWidth, learningRate, "layer0_conv1", 1);
    fheonHEController.decrypt_batch_data(gradInputs, printingCtxt[0], printingCtxt[1]);

    printDuration(begin_time, "Total Time (fetch + forward + backward)", false);
    cout << "\nBatch fine-tuning step completed." << endl;

    return 0;
}
