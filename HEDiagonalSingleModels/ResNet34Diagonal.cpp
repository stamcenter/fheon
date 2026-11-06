
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

#include <iostream>
#include <sys/stat.h>

#include "FHEONHEController.h"
#include "FHEONANNDiagonalController.h"

#ifndef DEFAULT_TEST_SIZE
#define DEFAULT_TEST_SIZE 100
#endif

#ifndef INDEX_VALUE
#define INDEX_VALUE 0
#endif

using namespace std;
CryptoContext<DCRTPoly> context;
FHEONHEController fheonHEController(context);
/** the ANN controller: every call below uses the shared interface of the ANN controllers */
using ANNController = ANNDiagonalController;

/** a single bootstrapping iteration: the second one doubles the cost without improving the precision when no precision is supplied */
const int bootstrapIterations = 1;

/** degree of the ReLU approximation (as in ResNet20Diagonal). The Chebyshev evaluation uses 6 levels at degree 59, 5 at
 *  degree 27 and 4 at degree 13, plus one for the scaling of the input; the depth below follows from this value */
const int reluDegree = 59;
const int reluLevels = ((reluDegree > 27) ? 6 : (reluDegree > 13) ? 5 : 4) + 1;

/** a bootstrapping is followed either by a ReLU and a convolution (reluLevels + 1 levels) or by a stride-2 block.
 *  When the two channel stages of the stride-2 downsampling do not fit in those levels, a single stage is used */
FHEONLayout stride2Layout(int inputWidth, int inputChannels){
    FHEONLayout layout;
    int twoStageLevels = static_cast<int>(log2(inputWidth/2)) + 1 + 2 + 1;
    layout.channelGroup = (twoStageLevels > reluLevels + 1) ? inputChannels : 0;
    return layout;
}

Ctext convolution_block(FHEONHEController &fheonHEController, ANNController &fheonANNController, string layer, Ctext &encryptedInput,
                        int dataWidth, int inputChannels, int outputChannels);
vector<Ctext> double_shortcut_convolution_block(FHEONHEController &fheonHEController, ANNController &fheonANNController, string layer,
                        Ctext &encryptedInput, int dataWidth, int inputChannels, int outputChannels);
Ctext resnet_block(FHEONHEController &fheonHEController, ANNController &fheonANNController, string layer, Ctext encryptedInput, int &dataWidth,
                   int inputChannels, int outputChannels, bool bootstrapState, const string &nextKeyFile);
Ctext avgpool_fc_layer_block(FHEONHEController &fheonHEController, ANNController &fheonANNController, string layer, Ctext &encryptedInput,
                   int dataWidth, int inputChannels, int outputChannels);

int main(int argc, char *argv[]) {
    weights_folder = "./../weights/resnet34/";
    predictions_file = "./../results/resnet34/fhepredictions.txt";
    auto runtime_config = parse_runtime_args(argc, argv, DEFAULT_TEST_SIZE, INDEX_VALUE);
    int test_size = runtime_config.test_size;
    int index_value = runtime_config.index_value;

    auto begin_time = startTime();
    printWelcomeMessage();
    bool loadContext = runtime_config.load_context;
    FHEONHEController::HEConfig heinitConfig;
    heinitConfig.ringDim = 15;
    heinitConfig.numSlots = 14;
    heinitConfig.mlevelBootstrap = reluLevels + 2;
    heinitConfig.dcrtBits = 48;
    heinitConfig.firstMod = 52;
    heinitConfig.numDigits = 4;
    heinitConfig.levelBudget = {4, 4};
    heinitConfig.serialize = true;
    if (!runtime_config.keys_folder.empty()) {
        heinitConfig.keysFolder = runtime_config.keys_folder;
    }
    fheonHEController.load_context(heinitConfig, loadContext);
    context = fheonHEController.getContext();
    ANNController fheonANNController(context);
    printDuration(begin_time, loadContext ? "Context Loaded" : "Context Generated and Keys Serialization", false);
    /** several slot counts are bootstrapped (one per stage): they must share the same correction factor */
    fheonHEController.set_bootstrap_correction_factor(9);
    cout << "---------------RESNET34-------------"<< to_string(test_size) << "--------------------" << endl; 

    /*************************************************** Network ***************************************************************************
     * 4 stages of basic blocks; the first block of stages 2-4 is a stride-2 block (3x3 convolution + 1x1 shortcut) that halves the
     * width and returns ciphertexts with half the slots. Every convolution is evaluated over channel diagonals.
     **************************************************************************************************************************************/
    int img_cols = 32;
    int img_depth = 3;
    vector<int> stageChannels = {16, 32, 64, 128};
    vector<int> stageBlocks = {3, 4, 6, 3};
    int numClasses = 100;
    int numStages = stageChannels.size();
    vector<int> stageWidth, stageSlots;
    for (int stage = 0; stage < numStages; stage++) {
        stageWidth.push_back(img_cols >> stage);
        stageSlots.push_back(stageChannels[stage]*stageWidth[stage]*stageWidth[stage]);
    }
    /** the first layer has 3 input channels: its input is replicated every 4 blocks, so it needs 4 diagonals instead of 16 */

    /*********************************************** Key Generation ******************************************************************************
     * One key file per stage: the bootstrapping keys for the slot count of the stage, the rotations of its convolutions and of the
     * stride-2 block that ends it (the last stage holds the pooling and the linear layer). Only the current stage is in memory.
     **********************************************************************************************************************************************/
    auto begin_rotkeygenerate_time = startTime();
    fheonHEController.clear_context(1 << heinitConfig.numSlots);
    vector<string> stageKeyFiles(numStages);
    for (int stage = 0; stage < numStages; stage++) {
        vector<vector<int>> stageKeys;
        int width = stageWidth[stage];
        stageKeys.push_back(fheonANNController.generate_convolution_rotation_positions(width, stageChannels[stage], stageChannels[stage], 3, 1, 1));
        if (stage == 0) {
            /** first layer (3 input channels, replicated input) */
            stageKeys.push_back(fheonANNController.generate_convolution_rotation_positions(width, img_depth, stageChannels[stage], 3, 1, 1));
        }
        if (stage + 1 < numStages) {
            stageKeys.push_back(fheonANNController.generate_convolution_rotation_positions(width, stageChannels[stage], stageChannels[stage+1], 3, 1, 2, stride2Layout(width, stageChannels[stage])));
        } else {
            FHEONLayout fcLayout;
            fcLayout.inputSlots = stageSlots[stage];
            stageKeys.push_back(fheonANNController.generate_linear_rotation_positions(numClasses, 0, fcLayout));
        }
        vector<int> serializedKeys = serialize_rotation_keys(stageKeys);
        stageKeyFiles[stage] = "layer" + to_string(stage+1) + ".bin";
        cout << "Stage " << stage+1 << " (" << stageSlots[stage] << " slots) keys (" << serializedKeys.size() << ") " << serializedKeys << endl;
        fheonHEController.generate_bootstrapping_and_rotation_keys(serializedKeys, stageSlots[stage], stageKeyFiles[stage], true);
        fheonHEController.clear_context(stageSlots[stage]);
    }
    printDuration(begin_rotkeygenerate_time, "Rotation KeyGen Time", false);
    /********************************************************************************************************************************************/
    int numImages = test_size + index_value;
    int dataSize = img_depth*pow(img_cols, 2);
    /** trained on CIFAR-100 (fine labels), with the CIFAR-10 normalization that read_images applies */
    vector<vector<double>> imagesData = read_images(cifar100tPath, numImages, dataSize);
    /** the circuit before the first bootstrapping is 3 convolutions and 2 ReLUs, and has to end with 3 towers */
    int numTowers = context->GetCryptoParameters()->GetElementParams()->GetParams().size();
    int inputLevel = max(1, numTowers - (3 + 2*reluLevels + 3));

    for (int imageIndex = index_value; imageIndex < numImages; imageIndex++) {
        int dataWidth = img_cols;
        auto image = imagesData[imageIndex];
        Ctext convData = fheonHEController.encrypt_input_at_level(image, inputLevel);
        cout << endl << imageIndex+1 << " - image Read, Normalized and Encrypted with " << image.size() << " Elements" << endl;
        auto inference_time = startTime();

        fheonHEController.clear_context(stageSlots[numStages-1]);
        context = fheonHEController.read_evaluation_keys(context, stageKeyFiles[0]);
        fheonHEController.prepare_bootstrapping(stageSlots[0]);
        convData = convolution_block(fheonHEController, fheonANNController, "layer0_conv1", convData, dataWidth, img_depth, stageChannels[0]);
        int reluScale = fheonHEController.read_scaling_value(convData, stageSlots[0]);
        convData = fheonANNController.he_relu(convData, reluScale, convData->GetSlots(), reluDegree);

        for (int stage = 0; stage < numStages; stage++) {
            for (int block = 1; block <= stageBlocks[stage]; block++) {
                string layer = "layer" + to_string(stage+1) + "_block" + to_string(block);
                bool downsample = (stage > 0 && block == 1);
                int inputChannels = downsample ? stageChannels[stage-1] : stageChannels[stage];
                /** the first block runs on the fresh ciphertext; a stride-2 block loads the keys of its stage once it has downsampled */
                convData = resnet_block(fheonHEController, fheonANNController, layer, convData, dataWidth, inputChannels, stageChannels[stage],
                                        !(stage == 0 && block == 1), downsample ? stageKeyFiles[stage] : "");
            }
        }

        /** the pooling and the linear layer need one level: bootstrap only when it is not available */
        if (convData->GetElements()[0].GetNumOfElements() - (convData->GetNoiseScaleDeg() - 1) < 3) {
            convData = fheonHEController.bootstrap_function(convData, bootstrapIterations);
        }
        convData = avgpool_fc_layer_block(fheonHEController, fheonANNController, "layer_fc", convData, dataWidth, stageChannels[numStages-1], numClasses);

        string infereMessage =  to_string(imageIndex + 1)+"  --  "; 
        printDuration(inference_time, infereMessage, false);
        fheonHEController.read_inferenced_label(convData, numClasses, predictions_file);
    }
    
    cout << "All predicted results printed to File." << endl;
    clear_images(imagesData, numImages);
    return 0;
}

Ctext convolution_block(FHEONHEController &fheonHEController, ANNController &fheonANNController, string layer, Ctext &encryptedInput,
                        int dataWidth, int inputChannels, int outputChannels){
    if (layer.find("block") == std::string::npos && layer.find("_conv") == std::string::npos) { printModelLayer(layer); }
    string dataPath = weights_folder+layer;
    int kernelWidth = 3;
    auto biasVector = load_bias(dataPath+"_bias.csv");
    auto rawKernelData = load_weights(dataPath+"_weight.csv", outputChannels, inputChannels, kernelWidth, kernelWidth);
    int numSlots = encryptedInput->GetSlots();
    int encode_level = fheonANNController.convolution_kernel_level(encryptedInput, dataWidth);
    auto kernelData = fheonANNController.encode_convolution_kernel(fheonHEController, rawKernelData, dataWidth, numSlots, encode_level);
    auto biasEncoded = fheonANNController.encode_convolution_bias(fheonHEController, biasVector, dataWidth, numSlots, encode_level);
    return fheonANNController.he_convolution(encryptedInput, kernelData, biasEncoded, dataWidth, inputChannels, outputChannels, 3, 1);
}

vector<Ctext> double_shortcut_convolution_block(FHEONHEController &fheonHEController, ANNController &fheonANNController, string layer,
                        Ctext &encryptedInput, int dataWidth, int inputChannels, int outputChannels){
    string dataPath = weights_folder+layer;
    int kernelWidth = 3;
    int widthOut = dataWidth/2;
    int inputSlots = encryptedInput->GetSlots();
    FHEONLayout layout = stride2Layout(dataWidth, inputChannels);
    int encode_level = fheonANNController.convolution_kernel_level(encryptedInput, dataWidth, 2, layout);

    auto biasVector = load_bias(dataPath+"_conv1_bias.csv");
    auto rawKernelData = load_weights(dataPath+"_conv1_weight.csv", outputChannels, inputChannels, kernelWidth, kernelWidth);
    auto kernelData = fheonANNController.encode_convolution_kernel(fheonHEController, rawKernelData, dataWidth, inputSlots, encode_level, 2, 1, layout);
    auto biasEncoded = fheonANNController.encode_convolution_bias(fheonHEController, biasVector, widthOut, inputSlots/2, encode_level);

    auto shortcutbiasVector = load_bias(dataPath+"_shortcut_bias.csv");
    auto shortcutrawKernelData = load_fc_weights(dataPath+"_shortcut_weight.csv",  outputChannels, inputChannels);
    auto shortcutKernelData = fheonANNController.encode_shortcut_kernel(fheonHEController, shortcutrawKernelData, dataWidth, inputSlots, encode_level);
    auto shortcutBiasEncoded = fheonANNController.encode_convolution_bias(fheonHEController, shortcutbiasVector, widthOut, inputSlots/2, encode_level);

    return fheonANNController.he_convolution_and_shortcut_optimized(encryptedInput, kernelData, shortcutKernelData, biasEncoded, shortcutBiasEncoded,
                                                                   dataWidth, inputChannels, outputChannels, layout);
}

Ctext resnet_block(FHEONHEController &fheonHEController, ANNController &fheonANNController, string layer, Ctext encryptedInput, int &dataWidth,
                   int inputChannels, int outputChannels, bool bootstrapState, const string &nextKeyFile){
    printModelLayer(layer);
    Ctext shortcut_convData = encryptedInput;
    Ctext convData;
    if (!nextKeyFile.empty()) {
        encryptedInput = fheonHEController.bootstrap_function(encryptedInput, bootstrapIterations);
        /** the outputs come back with half the slots: from here on the keys of the next stage are needed */
        auto doubleResults = double_shortcut_convolution_block(fheonHEController, fheonANNController, layer, encryptedInput, dataWidth, inputChannels, outputChannels);
        fheonHEController.clear_context(encryptedInput->GetSlots());
        context = fheonHEController.read_evaluation_keys(context, nextKeyFile);
        fheonHEController.prepare_bootstrapping(doubleResults[0]->GetSlots());
        dataWidth = dataWidth/2;
        convData = doubleResults[0];
        shortcut_convData = doubleResults[1];
    }
    else {
        convData = convolution_block(fheonHEController, fheonANNController, layer+"_conv1", encryptedInput, dataWidth, inputChannels, outputChannels);
    }
    int dataSize = convData->GetSlots();
    if (bootstrapState) {
        convData = fheonHEController.bootstrap_function(convData, bootstrapIterations);
    }
    int reluScale = fheonHEController.read_scaling_value(convData, dataSize);
    convData = fheonANNController.he_relu(convData, reluScale, convData->GetSlots(), reluDegree);

    auto second_convData = convolution_block(fheonHEController, fheonANNController, layer+"_conv2", convData, dataWidth, outputChannels, outputChannels);
    Ctext sum_convData = fheonANNController.he_sum_two_ciphertexts(second_convData, shortcut_convData);
    sum_convData = fheonHEController.bootstrap_function(sum_convData, bootstrapIterations);

    reluScale = fheonHEController.read_scaling_value(sum_convData, dataSize);
    return fheonANNController.he_relu(sum_convData, reluScale, sum_convData->GetSlots(), reluDegree);
}

Ctext avgpool_fc_layer_block(FHEONHEController &fheonHEController, ANNController &fheonANNController, string layer, Ctext &encryptedInput,
                   int dataWidth, int inputChannels, int outputChannels){
    printModelLayer(layer);
    string dataPath = weights_folder+layer;
    auto fc_biasVector = load_bias(dataPath+"_bias.csv");
    auto fc_rawKernelData = load_fc_weights(dataPath+"_weight.csv", outputChannels, inputChannels);
    /** the global average pooling is part of the linear layer: every input is the average of the dataWidth^2 values of its channel */
    int channelSize = dataWidth*dataWidth;
    FHEONLayout layout;
    layout.weightScale = 1.0/channelSize;
    layout.inputPositions.resize(inputChannels);
    for (int ch = 0; ch < inputChannels; ch++) {
        for (int i = 0; i < channelSize; i++) { layout.inputPositions[ch].push_back(ch*channelSize + i); }
    }
    int numSlots = encryptedInput->GetSlots();
    int encode_level = fheonANNController.convolution_kernel_level(encryptedInput, 1);
    auto fc_kernelData = fheonANNController.encode_linear_weights(fheonHEController, fc_rawKernelData, numSlots, encode_level, layout);
    Ptext fc_biasEncoded = fheonANNController.encode_linear_bias(fheonHEController, fc_biasVector, numSlots, encode_level);
    return fheonANNController.he_linear(encryptedInput, fc_kernelData, fc_biasEncoded, inputChannels, outputChannels, 0, layout);
}
