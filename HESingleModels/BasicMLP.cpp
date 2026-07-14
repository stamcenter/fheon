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
#include "FHEONANNController.h"

using namespace std;

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
    
    FHEONHEController::HEConfig heinitConfig;
    heinitConfig.ringDim = 11;
    heinitConfig.numSlots = 10;
    heinitConfig.mlevelBootstrap = 8;
    heinitConfig.dcrtBits = 32;
    heinitConfig.firstMod = 36;
    heinitConfig.numDigits = 3;
    heinitConfig.levelBudget = {3, 3};
    heinitConfig.serialize = true;
    if (!runtime_config.keys_folder.empty()) {
        heinitConfig.keysFolder = runtime_config.keys_folder;
    }
    
    fheonHEController.load_context(heinitConfig, loadContext);
    context = fheonHEController.getContext();
    FHEONANNController fheonANNController(context);
    printDuration(begin_time, "Context Generated and Keys Serialization", false);

    vector<vector<int>> rotation_keys;
    int rotPositions = 16;
    vector<int> channels = {784, 128, 64, 10};
   
    //** generate rotation keys*/
    auto rotation_positions = fheonANNController.generate_linear_rotation_positions(channels[0], rotPositions);
    
    /*** Generate the rotation positions, generate rotation keys, and load rotation keys */
    auto begin_rotkeygen_time = startTime();
    cout << "This is the rotation positions (" << rotation_positions.size() <<"): " << rotation_positions << endl;
    fheonHEController.generate_rotation_keys(rotation_positions, "rotation_keys.bin",  true, true);
    printDuration(begin_rotkeygen_time, "Rotation KeyGen (position, gen, and load) Time", false);

    /*************************************************** Prepare Weights for the network **************************************************/
    auto wloading_time = startTime();
    string dataPath = weights_folder;

    /*** first fully layer connected kernel and bias */
    auto fc1_biasVector = load_bias(dataPath+"fc1_bias.csv");
    auto fc1_rawKernel = load_fc_weights(dataPath+"fc1_weight.csv", channels[1], channels[0]);
    vector<Ptext> fc1_kernelData;
    for(int i=0; i < channels[1]; i++){
        auto encodeWeights = fheonHEController.encode_input(fc1_rawKernel[i]);
        fc1_kernelData.push_back(encodeWeights);
    }
    Ptext fc1baisVector = context->MakeCKKSPackedPlaintext(fc1_biasVector, 1);
    
     /*** second fully layer connected weights and bias */
    auto fc2_biasVector = load_bias(dataPath+"fc2_bias.csv");
    auto fc2_rawKernel = load_fc_weights(dataPath+"fc2_weight.csv", channels[2], channels[1]);
    vector<Ptext> fc2_kernelData;
    for(int i=0; i<channels[2]; i++){
        auto encodeWeights = fheonHEController.encode_input(fc2_rawKernel[i]);
        fc2_kernelData.push_back(encodeWeights);
    }
    Ptext fc2baisVector = context->MakeCKKSPackedPlaintext(fc2_biasVector, 1);

    auto fc3_biasVector = load_bias(dataPath+"fc3_bias.csv");
    auto fc3_rawKernel = load_fc_weights(dataPath+"fc3_weight.csv", channels[3], channels[2]);
    vector<Ptext> fc3_kernelData;
    for(int i=0; i<channels[3]; i++){
        auto encodeWeights = fheonHEController.encode_input(fc3_rawKernel[i]);
        fc3_kernelData.push_back(encodeWeights);
    }
    Ptext fc3baisVector = context->MakeCKKSPackedPlaintext(fc3_biasVector, 1);

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
        Ctext encryptedInput = fheonHEController.encrypt_input(readImage);
        cout << endl << imageIndex+1 << " - image Read, Normalized and Encrypt"<< endl;

        auto inference_time = startTime();

        printModelLayer("Layer", 1);
        auto mlpData = fheonANNController.he_linear(encryptedInput, fc1_kernelData, fc1baisVector, channels[0], channels[1], rotPositions);
       
        reluScale = fheonHEController.read_scaling_value(mlpData, channels[1]); 
        cout << "ReLU Scale after FC1: " << reluScale << endl;
        mlpData = fheonANNController.he_relu(mlpData, reluScale, channels[1]);

        printModelLayer("Layer", 2);
        mlpData = fheonANNController.he_linear(mlpData, fc2_kernelData, fc2baisVector, channels[1], channels[2], rotPositions);
        
        reluScale = fheonHEController.read_scaling_value(mlpData, channels[2]);
        cout << "ReLU Scale after FC2: " << reluScale << endl;
        mlpData = fheonANNController.he_relu(mlpData, reluScale, channels[2]);

        printModelLayer("Layer", 3);
        mlpData = fheonANNController.he_linear(mlpData, fc3_kernelData, fc3baisVector, channels[2], channels[3], rotPositions);
        
        string infereMessage = "Total Run Time for Image " + to_string(imageIndex + 1);  
        printDuration(inference_time, infereMessage, false);
        fheonHEController.read_inferenced_label(mlpData, channels[3], predictions_file);
    }
    cout << "All predicted results printed to File." << endl;
    clear_mnist_images(mnistData, numImages);
    return 0;
}
