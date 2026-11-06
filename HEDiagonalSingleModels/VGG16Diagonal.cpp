
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
const int reluDegree = 27;
const int reluLevels = ((reluDegree > 27) ? 6 : (reluDegree > 13) ? 5 : 4) + 1;

/** a bootstrapping is followed by a ReLU and a convolution or linear layer (reluLevels + 1 levels), or by a pooling and a
 *  convolution. When the two channel stages of the pooling downsampling do not fit in those levels, a single stage is used */
int poolChannelGroup(int inputWidth, int inputChannels){
    int twoStageLevels = static_cast<int>(log2(inputWidth/2)) + 1 + 2 + 1;
    return (twoStageLevels > reluLevels + 1) ? inputChannels : 0;
}

/** layout of the pooling that ends a stage: output on the slots of the next stage */
FHEONLayout poolLayout(int inputWidth, int inputChannels, int outputSlots){
    FHEONLayout layout;
    layout.outputSlots = outputSlots;
    layout.channelGroup = poolChannelGroup(inputWidth, inputChannels);
    return layout;
}

/** the activations are kept normalized: every ReLU divides its input by the largest value it holds (read like the ReLU
 *  scales of the other models) and returns ReLU(x / s) within [0, 1], so the approximation uses its whole range and the
 *  bootstrappings see small values. The ciphertexts then hold the true activations divided by activationScale, the product
 *  of those maxima: each layer divides its bias by it, and the last layer multiplies its weights by it to give true logits */
double activationScale = 1.0;

/** a convolution of the network, with the 2x2 average pooling that follows it (if any) */
struct ConvLayer {
    string name;
    int inputChannels;
    int outputChannels;
    bool poolAfter;
};

Ctext convolution_block(FHEONHEController &fheonHEController, ANNController &fheonANNController, const ConvLayer &layer, Ctext &encryptedInput,
                        int dataWidth, double weightScale);
Ctext relu_block(FHEONHEController &fheonHEController, ANNController &fheonANNController, Ctext &encryptedInput);
Ctext linear_block(FHEONHEController &fheonHEController, ANNController &fheonANNController, string layer, Ctext &encryptedInput, int inputChannels,
                   int outputChannels, const vector<vector<int>> &inputPositions, double weightScale, double biasScale);

int main(int argc, char *argv[]) {
    weights_folder = "./../weights/vgg16/";
    predictions_file = "./../results/vgg16/fhepredictions.txt";
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
    cout << "---------------VGG16-------------"<< to_string(test_size) << "--------------------" << endl; 

    /*************************************************** Network ***************************************************************************
     * Every convolution is a 3x3 "same" convolution evaluated over channel diagonals; its channels sit in consecutive blocks
     * and the ciphertext has exactly outputChannels * width^2 slots. A pooling halves the width and repacks its output for
     * the next stage, whose ciphertexts (and bootstrappings) have fewer slots. The global average pooling is part of fc1.
     **************************************************************************************************************************************/
    int img_depth = 3;
    int img_cols = 32;
    int numClasses = 10;
    int fcWidth = 1024;
    int minimumSlots = 1024;
    vector<ConvLayer> layers = {
        {"conv1", img_depth, 16, false},
        {"conv2", 16, 16, true},
        {"conv3", 16, 32, false},
        {"conv4", 32, 32, true},
        {"conv5", 32, 64, false},
        {"conv6", 64, 64, false},
        {"conv7", 64, 64, true},
        {"conv8", 64, 128, false},
        {"conv9", 128, 128, false},
        {"conv10", 128, 128, true},
        {"conv11", 128, 128, false},
        {"conv12", 128, 128, false},
        {"conv13", 128, 128, false},
    };

    /** stage of every layer, and width and slot count of every stage */
    vector<int> layerStage, stageWidth = {img_cols}, stageSlots;
    for (const auto &layer : layers) {
        layerStage.push_back(stageWidth.size() - 1);
        if (static_cast<int>(stageSlots.size()) < static_cast<int>(stageWidth.size())) {
            /** bootstrapping below 1024 slots is not accurate with this level budget: smaller stages are zero-padded to 1024 slots */
            stageSlots.push_back(max(minimumSlots, layer.outputChannels * stageWidth.back() * stageWidth.back()));
        }
        if (layer.poolAfter) {
            stageWidth.push_back(stageWidth.back() / 2);
        }
    }
    int numStages = stageWidth.size();
    int lastWidth = stageWidth.back();
    int lastChannels = layers.back().outputChannels;
    int fcSlots = fcWidth;

    /** inputs of fc1: the 4 values of every channel of the last convolution (global average pooling) */
    vector<vector<int>> fc1Positions(lastChannels), fcPositions(fcWidth);
    for (int ch = 0; ch < lastChannels; ch++) {
        for (int i = 0; i < lastWidth*lastWidth; i++) {
            fc1Positions[ch].push_back(ch*lastWidth*lastWidth + i);
        }
    }
    for (int i = 0; i < fcWidth; i++) { fcPositions[i] = {i}; }

    /*********************************************** Key Generation ******************************************************************************
     * One key file per stage, holding the bootstrapping keys for the slot count of the stage and the rotations of its
     * convolutions and of the pooling that ends it. Only the keys of the current stage are in memory during the inference.
     **********************************************************************************************************************************************/
    auto begin_rotkeygenerate_time = startTime();
    fheonHEController.clear_context(1 << heinitConfig.numSlots);
    vector<string> stageKeyFiles(numStages);
    vector<vector<int>> fcKeys;
    FHEONLayout fcLayout;
    fcLayout.inputSlots = fcSlots;
    fcKeys.push_back(fheonANNController.generate_linear_rotation_positions(fcWidth, 0, fcLayout));
    fcKeys.push_back(fheonANNController.generate_linear_rotation_positions(numClasses, 0, fcLayout));
    vector<int> serializedFcKeys = serialize_rotation_keys(fcKeys);
    cout << "FC (" << fcSlots << " slots) keys (" << serializedFcKeys.size() << ") " << serializedFcKeys << endl;
    fheonHEController.generate_bootstrapping_and_rotation_keys(serializedFcKeys, fcSlots, "fc_layer.bin", true);
    fheonHEController.clear_context(fcSlots);
    for (int stage = 0; stage < numStages; stage++) {
        vector<vector<int>> stageKeys;
        int width = stageWidth[stage];
        for (size_t i = 0; i < layers.size(); i++) {
            if (layerStage[i] != stage) continue;
            FHEONLayout convLayout;
            convLayout.inputSlots = stageSlots[stage];
            stageKeys.push_back(fheonANNController.generate_convolution_rotation_positions(width, layers[i].inputChannels, layers[i].outputChannels, 3, 1, 1, convLayout));
            if (layers[i].poolAfter) {
                stageKeys.push_back(fheonANNController.generate_avgpool_rotation_positions(width, 2, 2, layers[i].outputChannels, poolLayout(width, layers[i].outputChannels, stageSlots[stage+1])));
            }
        }
        vector<int> serializedKeys = serialize_rotation_keys(stageKeys);
        stageKeyFiles[stage] = "stage" + to_string(stage+1) + ".bin";
        cout << "Stage " << stage+1 << " (" << stageSlots[stage] << " slots) keys (" << serializedKeys.size() << ") " << serializedKeys << endl;
        fheonHEController.generate_bootstrapping_and_rotation_keys(serializedKeys, stageSlots[stage], stageKeyFiles[stage], true);
        fheonHEController.clear_context(stageSlots[stage]);
    }
    printDuration(begin_rotkeygenerate_time, "Rotation KeyGen Time", false);

    /********************************************************************************************************************************************/
    int numImages = test_size + index_value;
    int dataSize = img_depth*pow(img_cols, 2);
    vector<vector<double>> imagesData = read_images(cifar10tPath, numImages, dataSize);
    /** the first convolution, its ReLU and the next convolution run on the fresh ciphertext, which has to end with 3 towers */
    int numTowers = context->GetCryptoParameters()->GetElementParams()->GetParams().size();
    int inputLevel = max(1, numTowers - (1 + reluLevels + 1 + 3));

    for (int imageIndex = index_value; imageIndex < numImages; imageIndex++) {
        auto image = imagesData[imageIndex];
        Ctext convData = fheonHEController.encrypt_input_at_level(image, inputLevel);
        cout << endl << imageIndex+1 << " - image Read, Normalized and Encrypted with " << image.size() << " Elements" << endl;
        auto inference_time = startTime();

        int stage = 0;
        fheonHEController.clear_context(stageSlots[numStages-1]);
        context = fheonHEController.read_evaluation_keys(context, stageKeyFiles[stage]);
        fheonHEController.prepare_bootstrapping(stageSlots[stage]);
        double weightScale = 1.0;
        activationScale = 1.0;
        for (size_t i = 0; i < layers.size(); i++) {
            int width = stageWidth[stage];
            convData = convolution_block(fheonHEController, fheonANNController, layers[i], convData, width, weightScale);
            weightScale = 1.0;
            if (i > 0) {
                convData = fheonHEController.bootstrap_function(convData, bootstrapIterations);
            }
            convData = relu_block(fheonHEController, fheonANNController, convData);

            if (layers[i].poolAfter) {
                convData = fheonHEController.bootstrap_function(convData, bootstrapIterations);
                convData = fheonANNController.he_avgpool(convData, width, layers[i].outputChannels, 2, 2,
                                                                  poolLayout(width, layers[i].outputChannels, stageSlots[stage+1]));
                /** the factor of the pooling (1/4 for the diagonal sums) goes into the weights of the next convolution */
                weightScale *= fheonANNController.avgpool_weight_scale(2);
                fheonHEController.clear_context(stageSlots[stage]);
                stage++;
                context = fheonHEController.read_evaluation_keys(context, stageKeyFiles[stage]);
                fheonHEController.prepare_bootstrapping(stageSlots[stage]);
            }
        }

        /*** global average pooling + fully connected layers, on ciphertexts of fcSlots slots */
        fheonHEController.clear_context(stageSlots[stage]);
        context = fheonHEController.read_evaluation_keys(context, "fc_layer.bin");
        fheonHEController.prepare_bootstrapping(fcSlots);
        convData->SetSlots(fcSlots);
        convData = linear_block(fheonHEController, fheonANNController, "fc1", convData, lastChannels, fcWidth, fc1Positions,
                                1.0/(lastWidth*lastWidth), 1.0/activationScale);
        convData = fheonHEController.bootstrap_function(convData, bootstrapIterations);
        convData = relu_block(fheonHEController, fheonANNController, convData);
        convData = linear_block(fheonHEController, fheonANNController, "fc2", convData, fcWidth, fcWidth, fcPositions, 1.0, 1.0/activationScale);
        convData = fheonHEController.bootstrap_function(convData, bootstrapIterations);
        convData = relu_block(fheonHEController, fheonANNController, convData);
        convData = linear_block(fheonHEController, fheonANNController, "fc3", convData, fcWidth, numClasses, fcPositions, activationScale, 1.0);

        string infereMessage =  to_string(imageIndex + 1)+"  --  ";          
        printDuration(inference_time, infereMessage, false);
        fheonHEController.read_inferenced_label(convData, numClasses, predictions_file);
    }
    
    cout << "All predicted results printed to File." << endl;
    clear_images(imagesData, numImages);
    return 0;
}

Ctext convolution_block(FHEONHEController &fheonHEController, ANNController &fheonANNController, const ConvLayer &layer, Ctext &encryptedInput,
                        int dataWidth, double weightScale){
    printModelLayer(layer.name);
    string dataPath = weights_folder+layer.name;
    int kernelWidth = 3;
    auto biasVector = load_bias(dataPath+"_bias.csv");
    auto rawKernelData = load_weights(dataPath+"_weight.csv", layer.outputChannels, layer.inputChannels, kernelWidth, kernelWidth);
    for (auto &bias : biasVector) { bias /= activationScale; }
    FHEONLayout layout;
    layout.weightScale = weightScale;
    int numSlots = encryptedInput->GetSlots();
    int encode_level = fheonANNController.convolution_kernel_level(encryptedInput, dataWidth);
    auto kernelData = fheonANNController.encode_convolution_kernel(fheonHEController, rawKernelData, dataWidth, numSlots, encode_level, 1, 1, layout);
    auto biasEncoded = fheonANNController.encode_convolution_bias(fheonHEController, biasVector, dataWidth, numSlots, encode_level);
    return fheonANNController.he_convolution(encryptedInput, kernelData, biasEncoded, dataWidth, layer.inputChannels, layer.outputChannels, 3, 1);
}

Ctext relu_block(FHEONHEController &fheonHEController, ANNController &fheonANNController, Ctext &encryptedInput){
    /** largest input value (with a small margin for the noise): the output is ReLU(x / s) and the activations get divided by s */
    double reluScale = 1.01*max(fheonHEController.read_max_abs(encryptedInput, encryptedInput->GetSlots()), 1e-6);
    activationScale *= reluScale;
    return fheonANNController.he_relu(encryptedInput, reluScale, encryptedInput->GetSlots(), reluDegree, true);
}

Ctext linear_block(FHEONHEController &fheonHEController, ANNController &fheonANNController, string layer, Ctext &encryptedInput, int inputChannels,
                   int outputChannels, const vector<vector<int>> &inputPositions, double weightScale, double biasScale){
    printModelLayer(layer);
    string dataPath = weights_folder+layer;
    auto biasVector = load_bias(dataPath+"_bias.csv");
    auto rawKernelData = load_fc_weights(dataPath+"_weight.csv", outputChannels, inputChannels);
    for (auto &bias : biasVector) { bias *= biasScale; }
    FHEONLayout layout;
    layout.inputPositions = inputPositions;
    layout.weightScale = weightScale;
    int numSlots = encryptedInput->GetSlots();
    int encode_level = fheonANNController.convolution_kernel_level(encryptedInput, 1);
    auto kernelData = fheonANNController.encode_linear_weights(fheonHEController, rawKernelData, numSlots, encode_level, layout);
    Ptext biasEncoded = fheonANNController.encode_linear_bias(fheonHEController, biasVector, numSlots, encode_level);
    return fheonANNController.he_linear(encryptedInput, kernelData, biasEncoded, inputChannels, outputChannels, 0, layout);
}
