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

using namespace std;
/** the ANN controller: every call below uses the shared interface of the ANN controllers */
using ANNController = ANNDiagonalController;

CryptoContext<DCRTPoly> context;
FHEONHEController fheonHEController(context);

#ifndef DEFAULT_TEST_SIZE
#define DEFAULT_TEST_SIZE 250
#endif

#ifndef INDEX_VALUE
#define INDEX_VALUE 0
#endif

int main(int argc, char *argv[]) {
    weights_folder = "./../weights/basicmlp/";
    predictions_file = "./../results/basicmlp/fhepredictions.txt";
    auto runtime_config = parse_runtime_args(argc, argv, DEFAULT_TEST_SIZE, INDEX_VALUE);
    int test_size = runtime_config.test_size;
    int index_value = runtime_config.index_value;

    auto begin_time = startTime();
    printWelcomeMessage();
    bool loadContext = runtime_config.load_context;
    
    /** degree of the ReLU approximation: 27 gives the plaintext accuracy on the first 1000 test images (98.6%), like 59,
     *  with one level less per ReLU. A ReLU uses 7 levels at degree 59, 6 at degree 27 and 5 at degree 13 (scaling included) */
    int polyDegree = 27;
    int reluLevels = (polyDegree > 27) ? 7 : (polyDegree > 13) ? 6 : 5;
    FHEONHEController::HEConfig heinitConfig;
    heinitConfig.ringDim = 11;
    heinitConfig.numSlots = 10;
    /** leveled context, no bootstrapping: this is the whole depth of the chain.
     *  3 layers + 2 ReLUs, plus one tower so the logits end on 3 towers */
    heinitConfig.mlevelBootstrap = 3 + 2*reluLevels + 1;
    heinitConfig.dcrtBits = 32;
    heinitConfig.firstMod = 36;
    heinitConfig.numDigits = 1;
    heinitConfig.levelBudget = {3, 3};
    heinitConfig.serialize = true;
    if (!runtime_config.keys_folder.empty()) {
        heinitConfig.keysFolder = runtime_config.keys_folder;
    }
    
    if (loadContext) {
        fheonHEController.load_context(heinitConfig, true);
    } else {
        fheonHEController.generate_leveled_context(heinitConfig);
    }
    context = fheonHEController.getContext();
    ANNController fheonANNController(context);
    printDuration(begin_time, "Context Generated and Keys Serialization", false);

    int numSlots = 1 << heinitConfig.numSlots;
    vector<int> channels = {784, 128, 64, 10};

    /*************************************************** Layout of the network ************************************************************
     * Every layer is evaluated over its diagonals (baby-step giant-step rotations, one level), and returns its outputs
     * in the first slots of a ciphertext that only has as many slots as the layer has diagonals:
     *  fc1 : 784 -> 128, on the 1024-slot input,  128 diagonals
     *  fc2 : 128 -> 64,  on a 128-slot ciphertext, 64 diagonals
     *  fc3 : 64 -> 10,   on a 64-slot ciphertext,  16 diagonals
     **************************************************************************************************************************************/
    vector<int> fcSlots = {numSlots, 128, 64};

    //** generate rotation keys*/
    vector<vector<int>> rotation_keys;
    for (int layer = 0; layer < 3; layer++) {
        FHEONLayout keysLayout;
        keysLayout.inputSlots = fcSlots[layer];
        rotation_keys.push_back(fheonANNController.generate_linear_rotation_positions(channels[layer+1], 0, keysLayout));
    }
    vector<int> rotation_positions = serialize_rotation_keys(rotation_keys);
    
    /*** Generate the rotation positions, generate rotation keys, and load rotation keys */
    auto begin_rotkeygen_time = startTime();
    cout << "This is the rotation positions (" << rotation_positions.size() <<"): " << rotation_positions << endl;
    fheonHEController.generate_rotation_keys(rotation_positions, "rotation_keys.bin",  true, true);
    printDuration(begin_rotkeygen_time, "Rotation KeyGen (position, gen, and load) Time", false);

    /*************************************************** Prepare Weights for the network **************************************************/
    /** the levels are fixed by the circuit: 1 level per layer and reluLevels per ReLU, with no bootstrapping.
     *  The input is encrypted as low in the modulus chain as the circuit allows, so every operation works on few towers */
    auto wloading_time = startTime();
    string dataPath = weights_folder;
    int numTowers = context->GetCryptoParameters()->GetElementParams()->GetParams().size();
    int inputLevel = numTowers - (3 + 2*reluLevels + 2);
    if (inputLevel < 0) {
        cerr << "The modulus chain is too short for the circuit." << endl;
        return 1;
    }
    vector<int> fcLevels = {inputLevel, inputLevel + 1 + reluLevels, inputLevel + 2*(1 + reluLevels)};

    /*** first fully layer connected kernel and bias */
    auto fc1_biasVector = load_bias(dataPath+"fc1_bias.csv");
    auto fc1_rawKernel = load_fc_weights(dataPath+"fc1_weight.csv", channels[1], channels[0]);
    auto fc1_kernelData = fheonANNController.encode_linear_weights(fheonHEController, fc1_rawKernel, fcSlots[0], fcLevels[0]);
    Ptext fc1baisVector = fheonANNController.encode_linear_bias(fheonHEController, fc1_biasVector, fcSlots[0], fcLevels[0]);
    
     /*** second fully layer connected weights and bias */
    auto fc2_biasVector = load_bias(dataPath+"fc2_bias.csv");
    auto fc2_rawKernel = load_fc_weights(dataPath+"fc2_weight.csv", channels[2], channels[1]);
    auto fc2_kernelData = fheonANNController.encode_linear_weights(fheonHEController, fc2_rawKernel, fcSlots[1], fcLevels[1]);
    Ptext fc2baisVector = fheonANNController.encode_linear_bias(fheonHEController, fc2_biasVector, fcSlots[1], fcLevels[1]);

    auto fc3_biasVector = load_bias(dataPath+"fc3_bias.csv");
    auto fc3_rawKernel = load_fc_weights(dataPath+"fc3_weight.csv", channels[3], channels[2]);
    auto fc3_kernelData = fheonANNController.encode_linear_weights(fheonHEController, fc3_rawKernel, fcSlots[2], fcLevels[2]);
    Ptext fc3baisVector = fheonANNController.encode_linear_bias(fheonHEController, fc3_biasVector, fcSlots[2], fcLevels[2]);

    printDuration(wloading_time, "Weights Loading Time", false);

    /************************************************************************************************ */
    /************************************************************************************************ */
     /**** Read the MNIST Images and inference them */
    string mnistPath = utils::mnistPath;
    int numImages = 1;
    int imgWidth = 28;
    int imageSize = (imgWidth* imgWidth);
    int numtoShow = test_size + index_value;
    int reluScale = 25;
    
    /*** Read MNIST images ***/
    unsigned char** mnistData = read_mnist_images(mnistPath, numImages, imageSize);
    vector<double> readImage;

    for (int imageIndex = index_value; imageIndex < numtoShow; imageIndex++) {
        unsigned char* image = mnistData[imageIndex];
        readImage = read_single_mnist_image(image, imageSize);
        Ctext encryptedInput = fheonHEController.encrypt_input_at_level(readImage, inputLevel);
        cout << endl << imageIndex+1 << " - image Read, Normalized and Encrypt"<< endl;

        auto inference_time = startTime();

        printModelLayer("Layer", 1);
        auto mlpData = fheonANNController.he_linear(encryptedInput, fc1_kernelData, fc1baisVector, channels[0], channels[1], 0);
       
        reluScale = fheonHEController.read_scaling_value(mlpData, channels[1]); 
        cout << "ReLU Scale after FC1: " << reluScale << endl;
        mlpData = fheonANNController.he_relu(mlpData, reluScale, mlpData->GetSlots(), polyDegree);

        printModelLayer("Layer", 2);
        mlpData = fheonANNController.he_linear(mlpData, fc2_kernelData, fc2baisVector, channels[1], channels[2], 0);
        
        reluScale = fheonHEController.read_scaling_value(mlpData, channels[2]);
        cout << "ReLU Scale after FC2: " << reluScale << endl;
        mlpData = fheonANNController.he_relu(mlpData, reluScale, mlpData->GetSlots(), polyDegree);

        printModelLayer("Layer", 3);
        mlpData = fheonANNController.he_linear(mlpData, fc3_kernelData, fc3baisVector, channels[2], channels[3], 0);
        
        string infereMessage = "Total Run Time for Image " + to_string(imageIndex + 1);  
        printDuration(inference_time, infereMessage, false);
        fheonHEController.read_inferenced_label(mlpData, channels[3], predictions_file);
    }
    cout << "All predicted results printed to File." << endl;
    clear_mnist_images(mnistData, numImages);
    return 0;
}
