
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

#include "../src/FHEONHEController.h"
#include "../src/FHEONANNController.h"

using namespace std;

CryptoContext<DCRTPoly> context;
FHEONHEController fheonHEController(context);

#ifndef INDEX_VALUE
#define INDEX_VALUE 0
#endif

#ifndef DEFAULT_ARG
#define DEFAULT_ARG 250
#endif

vector<int> measuringTime;
auto startIn = get_current_time();

int main(int argc, char *argv[]) {

    auto begin_time = startTime();
    printWelcomeMessage();
    int ringDegree = 13;
    int numSlots = 12;
    int circuitDepth = 11;
    int dcrtBits = 46;
    int firstMod = 50;
    int digitSize = 4;
    vector<uint32_t> levelBudget = {3, 3};
    int serialize = true;
    fheonHEController.generate_context(ringDegree, numSlots, circuitDepth, dcrtBits, firstMod, digitSize, levelBudget, serialize);
    context = fheonHEController.getContext();
    FHEONANNController fheonANNController(context);
    printDuration(begin_time, "Context Generated and Keys Serialization", false);

    vector<vector<int>> rotation_keys;
    int kernelWidth = 5;
    int poolSize = 2;
    int Stride = 1;
    int paddingLen = 0;
    int rotPositions = 8;
    vector<int> imgWidth = {28, 24, 12, 8, 4};
    vector<int> channels = {1, 6, 16, 256, 120, 84, 10};
   
    //** generate rotation keys*/
    auto conv1_keys = fheonANNController.generate_convolution_rotation_positions(imgWidth[0], channels[0], channels[1],  kernelWidth, paddingLen, Stride);
    auto avg1_keys = fheonANNController.generate_avgpool_optimized_rotation_positions(imgWidth[1], channels[1],  poolSize, poolSize, false, "single_channel");
    auto conv2_keys = fheonANNController.generate_convolution_rotation_positions(imgWidth[2], channels[1], channels[2], kernelWidth, paddingLen, Stride);
    auto avg2_keys = fheonANNController.generate_avgpool_optimized_rotation_positions(imgWidth[3],channels[2], poolSize, poolSize, false, "single_channel");
    auto fc_keys = fheonANNController.generate_linear_rotation_positions(channels[4], rotPositions);
    
    rotation_keys.push_back(conv1_keys);
    rotation_keys.push_back(avg1_keys);
    rotation_keys.push_back(conv2_keys);
    rotation_keys.push_back(avg2_keys);
    rotation_keys.push_back(fc_keys);

    /*** join all keys and generate unique values only */
    vector<int> rotation_positions;
    for (const auto& vec : rotation_keys) {
        rotation_positions.insert(rotation_positions.end(), vec.begin(), vec.end());
    }

    std::sort(rotation_positions.begin(), rotation_positions.end());
    auto new_end = std::remove(rotation_positions.begin(), rotation_positions.end(), 0);
    new_end = std::unique(rotation_positions.begin(), rotation_positions.end());
    unique(rotation_positions.begin(), rotation_positions.end());
    rotation_positions.erase(new_end, rotation_positions.end());
    std::sort(rotation_positions.begin(), rotation_positions.end());
    
    /*** Generate the rotation positions, generate rotation keys, and load rotation keys */
    auto begin_rotkeygenerate_time = startTime();
    cout << "This is the rotation positions (" << rotation_positions.size() <<"): " << rotation_positions << endl;
    fheonHEController.generate_rotation_keys(rotation_positions, "rotation_keys.bin", true);
    printDuration(begin_rotkeygenerate_time, "Rotation KeyGen (position, gen, and load) Time", false);

    /*************************************************** Prepare Weights for the network **************************************************/
    /*** 1st Convolution */
    auto wloading_time = startTime();
    string dataPath = "./../weights/lenet5/";

    auto conv1_biasVector = load_bias(dataPath+"Conv1_bias.csv");
    auto conv1_rawKernel = load_weights(dataPath+"Conv1_weight.csv", channels[1], channels[0], kernelWidth, kernelWidth);
    int conv1WidthSq = pow(imgWidth[0], 2);
    vector<vector<Ptext>> conv1_kernelData;
    for(int i=0; i<channels[1]; i++){
        auto encodeKernel = fheonHEController.encode_kernel(conv1_rawKernel[i], conv1WidthSq);
        conv1_kernelData.push_back(encodeKernel);
    }
    auto conv1biasEncoded = fheonHEController.encode_bais_input(conv1_biasVector, (imgWidth[1] * imgWidth[1]));
    
    auto conv2_rawKernel = load_weights(dataPath+"Conv2_weight.csv", channels[2], channels[1], kernelWidth, kernelWidth);
    auto conv2_biasVector = load_bias(dataPath+"Conv2_bias.csv");
    int conv2WidthSq = pow(imgWidth[2], 2);
    vector<vector<Ptext>> conv2_kernelData;
    for(int i=0; i<channels[2]; i++){
        auto encodeKernel = fheonHEController.encode_kernel(conv2_rawKernel[i], conv2WidthSq);
        conv2_kernelData.push_back(encodeKernel);
    }
    auto conv2biasEncoded = fheonHEController.encode_bais_input(conv2_biasVector, (imgWidth[3]* imgWidth[3]));

     /*** first fully layer connected kernel and bias */
    auto fc1_biasVector = load_bias(dataPath+"FC1_bias.csv");
    auto fc1_rawKernel = load_fc_weights(dataPath+"FC1_weight.csv", channels[4], channels[3]);
    vector<Ptext> fc1_kernelData;
    for(int i=0; i < channels[4]; i++){
        auto encodeWeights = fheonHEController.encode_input(fc1_rawKernel[i]);
        fc1_kernelData.push_back(encodeWeights);
    }
    Ptext fc1baisVector = context->MakeCKKSPackedPlaintext(fc1_biasVector, 1);
    
     /*** second fully layer connected weights and bias */
    auto fc2_biasVector = load_bias(dataPath+"FC2_bias.csv");
    auto fc2_rawKernel = load_fc_weights(dataPath+"FC2_weight.csv", channels[5], channels[4]);
    vector<Ptext> fc2_kernelData;
    for(int i=0; i<channels[5]; i++){
        auto encodeWeights = fheonHEController.encode_input(fc2_rawKernel[i]);
        fc2_kernelData.push_back(encodeWeights);
    }
    Ptext fc2baisVector = context->MakeCKKSPackedPlaintext(fc2_biasVector, 1);

     /*** third fully layer connected weights and bias */
    auto fc3_biasVector = load_bias(dataPath+"FC3_bias.csv");
    auto fc3_rawKernel = load_fc_weights(dataPath+"FC3_weight.csv", channels[6], channels[5]);
    vector<Ptext> fc3_kernelData;
    for(int i=0; i<channels[6]; i++){
        auto encodeWeights = fheonHEController.encode_input(fc3_rawKernel[i]);
        fc3_kernelData.push_back(encodeWeights);
    }
    Ptext fc3baisVector = context->MakeCKKSPackedPlaintext(fc3_biasVector, 1);

    printDuration(wloading_time, "Weights Loading Time", false);

    /************************************************************************************************ */
     /**** Read the MNIST Images and inference them */
    string mnistPath = "./../images/mnist_images/raw/t10k-images-idx3-ubyte";
    int numImages = 1;
    int imageSize = channels[0]*(imgWidth[0]* imgWidth[0]);
    int numtoShow = DEFAULT_ARG+INDEX_VALUE;
    int reluScale = 10;
    vector<int> dataSizeVector;
    dataSizeVector.push_back((channels[1]*pow(imgWidth[1], 2)));
    dataSizeVector.push_back((channels[2]*pow(imgWidth[3], 2)));
    
    /*** Read MNIST images ***/
    unsigned char** mnistData = read_mnist_images(mnistPath, numImages, imageSize);
    vector<double> readImage;
    std::ofstream outFile;
    outFile.open("./../results/lenet5/fhepredictions.txt", std::ios_base::app);

    for (int imageIndex = INDEX_VALUE; imageIndex < numtoShow; imageIndex++) {
        unsigned char* image = mnistData[imageIndex];
        readImage = read_single_mnist_image(image, imageSize);
        // display_mnist_image(image, imageSize, true);
        Ctext encryptedImage = fheonHEController.encrypt_input(readImage);
        cout << endl << imageIndex+1 << " - image Read, Normalized and Encrypt"<< endl;
        /************************************************************************************************ */
        /***** The first Convolution Layer takes  image=(1,28,28), kernel=(6,1,5,5) 
         * stride=1, pooling=0 output= (6,24,24) = 3456 vals */
        auto inference_time = startTime();
        startIn = get_current_time();
        auto convData = fheonANNController.he_convolution(encryptedImage, conv1_kernelData, conv1biasEncoded, imgWidth[0], channels[0], channels[1], kernelWidth);
        measuringTime.push_back(measureTime(startIn, get_current_time()));
        reluScale = fheonHEController.read_scaling_value(convData, dataSizeVector[0]);

        startIn = get_current_time();
        convData = fheonANNController.he_relu(convData, reluScale, dataSizeVector[0]);
        convData = fheonANNController.he_avgpool_optimzed(convData, imgWidth[1], channels[1], poolSize, poolSize);
        
        /*** Second Conv2d Layer input = (6,12,12), kernel=(16,6,5,5) 
         * striding =1, padding = 0 output = (16,8,8) ***/
        convData = fheonANNController.he_convolution(convData, conv2_kernelData, conv2biasEncoded, imgWidth[2],channels[1], channels[2], kernelWidth);
        measuringTime.push_back(measureTime(startIn, get_current_time()));

        reluScale = fheonHEController.read_scaling_value(convData, dataSizeVector[1]);
        convData = fheonHEController.bootstrap_function(convData);
        startIn = get_current_time();
        convData = fheonANNController.he_relu(convData, reluScale, dataSizeVector[1]);
        measuringTime.push_back(measureTime(startIn, get_current_time()));
        convData = fheonHEController.bootstrap_function(convData);

        startIn = get_current_time();
        convData = fheonANNController.he_avgpool_optimzed(convData, imgWidth[3], channels[2], poolSize, poolSize);
        
        /*** fully connected layers */
        convData = fheonANNController.he_linear(convData, fc1_kernelData, fc1baisVector, channels[3], channels[4], rotPositions);
        measuringTime.push_back(measureTime(startIn, get_current_time()));

        reluScale = fheonHEController.read_scaling_value(convData, channels[4]);
        convData = fheonHEController.bootstrap_function(convData);
        
        startIn = get_current_time();
        convData = fheonANNController.he_relu(convData, reluScale, channels[4]);
        convData = fheonANNController.he_linear(convData, fc2_kernelData, fc2baisVector, channels[4], channels[5], rotPositions);
        measuringTime.push_back(measureTime(startIn, get_current_time()));

        reluScale = fheonHEController.read_scaling_value(convData, channels[5]);
        convData = fheonHEController.bootstrap_function(convData);

        startIn = get_current_time();
        convData = fheonANNController.he_relu(convData, reluScale, channels[5]);
        convData = fheonANNController.he_linear(convData, fc3_kernelData, fc3baisVector, channels[5], channels[6], rotPositions);
        measuringTime.push_back(measureTime(startIn, get_current_time()));

        totalTime(measuringTime);
        measuringTime.clear();

        string infereMessage = "Total Run Time for Image " + to_string(imageIndex + 1);  
        printDuration(inference_time, infereMessage, false);
        fheonHEController.read_inferenced_label(convData, channels[6], outFile);
    }
    outFile.close();
    cout << "All predicted results printed to File." << endl;
    clear_mnist_images(mnistData, numImages);
   return 0;
 
}
