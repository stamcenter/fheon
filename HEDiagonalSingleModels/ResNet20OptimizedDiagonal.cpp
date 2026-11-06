
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
#define DEFAULT_TEST_SIZE 250
#endif

#ifndef INDEX_VALUE
#define INDEX_VALUE 0
#endif

using namespace std;
CryptoContext<DCRTPoly> context;
FHEONHEController fheonHEController(context);
/** the ANN controller: every call below uses the shared interface, so FHEONANNStandardController can be dropped in */
using ANNController = ANNDiagonalController;

/** a single bootstrapping iteration: the second one doubles the cost without improving the precision when no precision is supplied */
const int bootstrapIterations = 1;

/** degree of the ReLU approximation. The Chebyshev evaluation uses 6 levels at degree 59, 5 at degree 27 and 4 at
 *  degree 13, plus one for the scaling of the input. Degree 27 shortens the chain by one level and is about 8% faster
 *  than 59; the first 10 test images are still all classified correctly, with logits about 15% smaller. Set 59 to get
 *  the more precise activations back: the depth and the levels below follow from this value */
const int reluDegree = 27;
const int reluLevels = ((reluDegree > 27) ? 6 : (reluDegree > 13) ? 5 : 4) + 1;

/** a bootstrapping is followed either by a ReLU and a convolution (reluLevels + 1 levels) or by a stride-2 block.
 *  When the two channel stages of the stride-2 downsampling do not fit in those levels, a single stage is used */
FHEONLayout stride2Layout(int inputWidth, int inputChannels){
    FHEONLayout layout;
    int twoStageLevels = static_cast<int>(log2(inputWidth/2)) + 1 + 2 + 1;
    layout.channelGroup = (twoStageLevels > reluLevels + 1) ? inputChannels : 0;
    return layout;
}

/** slot count of every stage (one stage per width). All the keys are generated at the start and stay in memory
 *  for the whole inference (no key file is loaded between the stages: more memory, no reloading) */
vector<int> stageSlots;

vector<int> stageRotationKeys(ANNController &fheonANNController, int stage, int img_cols, int img_depth, const vector<int> &channelValues);
Ctext convolution_block(ANNController &fheonANNController, string layer, Ctext &encryptedInput, int dataWidth, int inputChannels, int outputChannels);
vector<Ctext> double_shortcut_convolution_block(ANNController &fheonANNController, string layer, Ctext &encryptedInput, int dataWidth,
                                                int inputChannels, int outputChannels);
Ctext resnet_block(ANNController &fheonANNController, string layer, Ctext encryptedInput, int &dataWidth, int stage,
                   int inputChannels, int outputChannels, bool bootstrapState, bool shortcutConv);
Ctext avgpool_fc_layer_block(ANNController &fheonANNController, string layer, Ctext &encryptedInput, int dataWidth, int inputChannels, int outputChannels);

int main(int argc, char *argv[]) {
    weights_folder = "./../weights/resnet20/";
    predictions_file = "./../results/resnet20/fhepredictions.txt";
    auto runtime_config = parse_runtime_args(argc, argv, DEFAULT_TEST_SIZE, INDEX_VALUE);
    int test_size = runtime_config.test_size;
    int index_value = runtime_config.index_value;

    auto begin_time = startTime();
    printWelcomeMessage();
    bool loadContext = runtime_config.load_context;
    
    // Initialize FHE context and controllers
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
    cout << "---------------------------------RESNET20-------------"<< to_string(test_size) << "--------------------------" << endl; 
    
    int img_cols = 32;
    int img_depth = 3;
    vector<int> channelValues = {16, 32, 64, 10};
    int numStages = 3;
    for (int stage = 0; stage < numStages; stage++) {
        stageSlots.push_back(channelValues[stage]*(img_cols >> stage)*(img_cols >> stage));
    }

    /*********************************************** Key Generation ******************************************************************************
     * All the keys at once: the bootstrapping keys for the slot count of every stage, and the rotations of all the layers.
     **********************************************************************************************************************************************/
    auto begin_rotkeygenerate_time = startTime();
    fheonHEController.clear_context(1 << heinitConfig.numSlots);
    vector<vector<int>> rotkeys;
    for (int stage = 0; stage < numStages; stage++) {
        rotkeys.push_back(stageRotationKeys(fheonANNController, stage, img_cols, img_depth, channelValues));
        fheonHEController.generate_bootstrapping_keys(stageSlots[stage], "", false);
    }
    vector<int> rotation_positions = serialize_rotation_keys(rotkeys);
    cout << "This is the rotation positions (" << rotation_positions.size() << "): "  << rotation_positions << endl;
    fheonHEController.generate_rotation_keys(rotation_positions, "rotation_keys.bin", true);
    printDuration(begin_rotkeygenerate_time, "Rotation KeyGen Time", false);
    /********************************************************************************************************************************************/
    int numImages = test_size+index_value;
    int dataSize = img_depth*pow(img_cols, 2);
    vector<vector<double>> imagesData = read_images(cifar10tPath, numImages, dataSize);
    /** the circuit before the first bootstrapping is 3 convolutions and 2 ReLUs, and has to end with 3 towers */
    int numTowers = context->GetCryptoParameters()->GetElementParams()->GetParams().size();
    int inputLevel = max(1, numTowers - (3 + 2*reluLevels + 3));

    for (int imageIndex = index_value; imageIndex < numImages; imageIndex++) {
        int dataWidth = img_cols;
        auto image = imagesData[imageIndex];
        Ctext convData = fheonHEController.encrypt_input_at_level(image, inputLevel);
        cout << endl << imageIndex+1 << " - image Read, Normalized and Encrypted with " << image.size() << " Elements" << endl;
        /************************************************************************************************ */
        auto inference_time = startTime();
        fheonHEController.prepare_bootstrapping(stageSlots[0]);
        convData = convolution_block(fheonANNController, "layer0_conv1", convData, dataWidth, img_depth, channelValues[0]);
        int reluScale = fheonHEController.read_scaling_value(convData, stageSlots[0]);
        convData = fheonANNController.he_relu(convData, reluScale, stageSlots[0], reluDegree);

        for (int stage = 0; stage < numStages; stage++) {
            for (int block = 1; block <= 3; block++) {
                string layer = "layer" + to_string(stage+1) + "_block" + to_string(block);
                bool downsample = (stage > 0 && block == 1);
                int inputChannels = downsample ? channelValues[stage-1] : channelValues[stage];
                /** the first block runs on the fresh ciphertext */
                convData = resnet_block(fheonANNController, layer, convData, dataWidth, stage, inputChannels, channelValues[stage],
                                        !(stage == 0 && block == 1), downsample);
            }
        }
        /** the pooling and the linear layer need one level: bootstrap only when it is not available */
        if (convData->GetElements()[0].GetNumOfElements() - (convData->GetNoiseScaleDeg() - 1) < 3) {
            convData = fheonHEController.bootstrap_function(convData, bootstrapIterations);
        }
        convData = avgpool_fc_layer_block(fheonANNController, "layer_fc", convData, dataWidth, channelValues[2], channelValues[3]);
        
        string infereMessage =  to_string(imageIndex + 1)+"  --  "; 
        printDuration(inference_time, infereMessage, false);
        fheonHEController.read_inferenced_label(convData, channelValues[3], predictions_file);
    }
    
    cout << "All predicted results printed to File." << endl;
    clear_images(imagesData, numImages);
    return 0;
}

/** rotations of a stage: its convolutions, the first layer (stage 1), and the stride-2 block that ends it or the pooling + linear layer */
vector<int> stageRotationKeys(ANNController &fheonANNController, int stage, int img_cols, int img_depth, const vector<int> &channelValues){
    int width = img_cols >> stage;
    int channels = channelValues[stage];
    vector<vector<int>> keys;
    keys.push_back(fheonANNController.generate_convolution_rotation_positions(width, channels, channels, 3, 1, 1));
    if (stage == 0) {
        keys.push_back(fheonANNController.generate_convolution_rotation_positions(width, img_depth, channels, 3, 1, 1));
    }
    if (stage + 1 < 3) {
        keys.push_back(fheonANNController.generate_convolution_rotation_positions(width, channels, channelValues[stage+1], 3, 1, 2, stride2Layout(width, channels)));
    } else {
        FHEONLayout layout;
        layout.inputSlots = channels * width * width;
        keys.push_back(fheonANNController.generate_linear_rotation_positions(channelValues[3], 0, layout));
    }
    return serialize_rotation_keys(keys);
}

Ctext convolution_block(ANNController &fheonANNController, string layer, Ctext &encryptedInput, int dataWidth, int inputChannels, int outputChannels){
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

vector<Ctext> double_shortcut_convolution_block(ANNController &fheonANNController, string layer, Ctext &encryptedInput, int dataWidth,
                                                int inputChannels, int outputChannels){
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

Ctext resnet_block(ANNController &fheonANNController, string layer, Ctext encryptedInput, int &dataWidth, int stage,
                   int inputChannels, int outputChannels, bool bootstrapState, bool shortcutConv){
    printModelLayer(layer);
    Ctext shortcut_convData = encryptedInput;
    Ctext convData;
    if (shortcutConv) {
        encryptedInput = fheonHEController.bootstrap_function(encryptedInput, bootstrapIterations);
        auto doubleResults = double_shortcut_convolution_block(fheonANNController, layer, encryptedInput, dataWidth, inputChannels, outputChannels);
        dataWidth = dataWidth/2;
        convData = doubleResults[0];
        shortcut_convData = doubleResults[1];
        /** the outputs have the slot count of the next stage (its keys are already loaded) */
        fheonHEController.prepare_bootstrapping(stageSlots[stage]);
    }
    else {
        convData = convolution_block(fheonANNController, layer+"_conv1", encryptedInput, dataWidth, inputChannels, outputChannels);
    }
    int dataSize = convData->GetSlots();
    if (bootstrapState) {
        convData = fheonHEController.bootstrap_function(convData, bootstrapIterations);
    }
    int reluScale = fheonHEController.read_scaling_value(convData, dataSize);
    convData = fheonANNController.he_relu(convData, reluScale, dataSize, reluDegree);

    auto second_convData = convolution_block(fheonANNController, layer+"_conv2", convData, dataWidth, outputChannels, outputChannels);
    Ctext sum_convData = fheonANNController.he_sum_two_ciphertexts(second_convData, shortcut_convData);
    sum_convData = fheonHEController.bootstrap_function(sum_convData, bootstrapIterations);

    reluScale = fheonHEController.read_scaling_value(sum_convData, dataSize);
    return fheonANNController.he_relu(sum_convData, reluScale, dataSize, reluDegree);
}

Ctext avgpool_fc_layer_block(ANNController &fheonANNController, string layer, Ctext &encryptedInput, int dataWidth, int inputChannels, int outputChannels){
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
