
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

#include "./../src/FHEONHEController.h"
#include "../src/FHEONANNController.h"

#ifndef DEFAULT_ARG
#define DEFAULT_ARG 100
#endif

using namespace std;
CryptoContext<DCRTPoly> context;
FHEONHEController fheonHEController(context);

Ctext convolution_relu_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer, Ctext encrytedVector, int &dataWidth, int &dataSize, int kernelWidth, int padding, 
                                            int striding, int inputChannels, int outputChannels, int reluScale, bool bootstrapState = true);
Ctext FClayer_relu_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer, Ctext encrytedVector, int inputChannels, int outputChannels, int reluScale, int rotPosition);

vector<int> measuringTime;
auto startIn = get_current_time();

int main(int argc, char *argv[]) {

    auto begin_time = startTime();
    printWelcomeMessage();
    int ringDegree = 15;
    int numSlots = 14;
    int circuitDepth = 11;
    int dcrtBits = 50;
    int firstMod = 54;
    int digitSize = 4;
    vector<uint32_t> levelBudget = {3, 3};
    int serialize = true;
    fheonHEController.generate_context(ringDegree, numSlots, circuitDepth, dcrtBits, firstMod, digitSize, levelBudget, serialize);
    context = fheonHEController.getContext();
    FHEONANNController fheonANNController(context);
    printDuration(begin_time, "Context Generation and Keys Serialization", false);
    cout << "---------------VGG11-------------"<< to_string(DEFAULT_ARG) << "--------------------" << endl; 
    
    /**** Read the CIFAR-10 Images and inference them */
    string cifar10tPath = "./../images/cifar-10-batches-bin/test_batch.bin";
    int numImages = DEFAULT_ARG;
    int img_depth = 3;
    int img_cols = 32;
    int dataWidth = img_cols;
    int dataSize = img_depth*pow(img_cols, 2);
    vector<vector<double>> imagesData = read_images(cifar10tPath, numImages, dataSize);
    ofstream outFile;
    outFile.open("./../results/vgg11/fhepredictions.txt", ios_base::app);
    Ctext convData;
    Ptext decryptedData;
    int padding = 1;
    int striding = 1;
    int kernelWidth = 3; 
    int avgpoolSize = 2;
    vector<int> channelValues = {16, 32, 64, 128, 1024, 10};
    int rotPositions = 32;
    
    //** generate rotation keys for conv_layer 1 */
    auto conv1_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, img_depth, channelValues[0], striding);
    auto avgpool1_keys = fheonANNController.generate_avgpool_optimized_rotation_positions(dataWidth,  channelValues[0], avgpoolSize, avgpoolSize);
    dataWidth = dataWidth/2;

    auto conv2_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channelValues[0], channelValues[1], striding);
    auto avgpool2_keys = fheonANNController.generate_avgpool_optimized_rotation_positions(dataWidth, channelValues[1], avgpoolSize, avgpoolSize);
    dataWidth = dataWidth/2;

    auto conv3_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth,  channelValues[1], channelValues[2],  striding);
    auto conv4_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channelValues[2], channelValues[2], striding);
    auto avgpool3_keys = fheonANNController.generate_avgpool_optimized_rotation_positions(dataWidth, channelValues[2], avgpoolSize, avgpoolSize);
    dataWidth = dataWidth/2;

    auto conv5_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channelValues[2], channelValues[3], striding);
    auto conv6_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channelValues[3], channelValues[3], striding);
    auto avgpool4_keys = fheonANNController.generate_avgpool_optimized_rotation_positions(dataWidth,  channelValues[3], avgpoolSize, avgpoolSize);
    dataWidth = dataWidth/2;

    auto conv7_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channelValues[3], channelValues[3], striding);
    auto conv8_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channelValues[3], channelValues[3], striding);
    auto avgpool5_keys = fheonANNController.generate_avgpool_optimized_rotation_positions(dataWidth,  channelValues[3], avgpoolSize,  avgpoolSize, true, "multi_channels", rotPositions);

    /** Generate Fully Connected Keys */
    auto fc_keys = fheonANNController.generate_linear_rotation_positions(channelValues[4], rotPositions);
    /************************************************************************************************ */
    vector<vector<int>> rotation_keys;
    rotation_keys.push_back(conv1_keys);
    rotation_keys.push_back(conv2_keys);
    rotation_keys.push_back(conv3_keys);
    rotation_keys.push_back(conv4_keys);
    rotation_keys.push_back(conv5_keys);
    rotation_keys.push_back(conv6_keys);
    rotation_keys.push_back(conv7_keys);
    rotation_keys.push_back(conv8_keys);
    rotation_keys.push_back(avgpool1_keys);
    rotation_keys.push_back(avgpool2_keys);
    rotation_keys.push_back(avgpool3_keys);
    rotation_keys.push_back(avgpool4_keys);
    rotation_keys.push_back(avgpool5_keys);
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

    /*********************************************** Key Generation ******************************************************************************/
    auto begin_rotkeygenerate_time = startTime();
    cout << "This is the rotation positions keys ("<< rotation_positions.size() << "): " << rotation_positions << endl;
    fheonHEController.generate_rotation_keys(rotation_positions, "rotation_keys.bin", true);
    printDuration(begin_rotkeygenerate_time, "Rotation KeyGen (position, gen, and load) Time", false);
    // fheonHEController.load_rotation_keys("rotation_keys.bin");
    /********************************************************************************************************************************************/
   
    int reluScale = 10;
    for (int imageIndex = 0; imageIndex < DEFAULT_ARG; imageIndex++) {
        auto image = imagesData[imageIndex];
        dataWidth = img_cols;
        dataSize = img_depth*pow(dataWidth, 2);
        // display_image(image, imageSize, true);
        Ctext encryptedImage = fheonHEController.encrypt_input(image);
        cout << endl << imageIndex+1 << " - image Read, Normalized and Encrypted with " << image.size() << " Elements" << endl;
        /************************************************************************************************ */
        auto inference_time = startTime();
        // cout<< "Layer 1" << endl;
        convData = convolution_relu_block(fheonHEController, fheonANNController, "conv1", encryptedImage, dataWidth, dataSize, kernelWidth, padding, striding, img_depth, channelValues[0], reluScale, false);
        startIn = get_current_time();
        convData = fheonANNController.he_avgpool_optimzed_with_multiple_channels(convData, dataWidth,  channelValues[0], avgpoolSize, avgpoolSize);
        measuringTime.push_back(measureTime(startIn, get_current_time()));
        dataWidth = dataWidth/2;
        dataSize = channelValues[0]*pow(dataWidth, 2);
        // fheonHEController.read_minmax(convData, dataSize);
        // printDuration(inference_time, "avgpool1", false);
        
        // cout<< "Layer 2" << endl;
        convData = convolution_relu_block(fheonHEController, fheonANNController, "conv2", convData, dataWidth, dataSize, kernelWidth, padding, striding, channelValues[0], channelValues[1], reluScale, true);
        convData = fheonHEController.bootstrap_function(convData);
        startIn = get_current_time();
        convData = fheonANNController.he_avgpool_optimzed_with_multiple_channels(convData, dataWidth,  channelValues[1], avgpoolSize, avgpoolSize);
        measuringTime.push_back(measureTime(startIn, get_current_time()));
        dataWidth = dataWidth/2;
        dataSize = channelValues[1]*pow(dataWidth, 2);
        // fheonHEController.read_minmax(convData, dataSize);
        // printDuration(inference_time, "avgpool2", false);
    
        // cout<< "Layer 3" << endl;
        convData = convolution_relu_block(fheonHEController, fheonANNController,  "conv3", convData, dataWidth, dataSize, kernelWidth, padding, striding, channelValues[1], channelValues[2], reluScale);
        convData = convolution_relu_block(fheonHEController, fheonANNController, "conv4", convData, dataWidth, dataSize, kernelWidth, padding, striding, channelValues[2], channelValues[2], reluScale);
        convData = fheonHEController.bootstrap_function(convData);
        startIn = get_current_time();
        convData = fheonANNController.he_avgpool_optimzed_with_multiple_channels(convData, dataWidth,  channelValues[2], avgpoolSize, avgpoolSize);
        measuringTime.push_back(measureTime(startIn, get_current_time()));
        dataWidth = dataWidth/2;
        dataSize = channelValues[2]*pow(dataWidth, 2);
        // fheonHEController.read_minmax(convData, dataSize);
        // printDuration(inference_time, "avgpool3", false);
        
        // cout<< "Layer 4" << endl;
        convData = convolution_relu_block(fheonHEController, fheonANNController,   "conv5",  convData, dataWidth, dataSize, kernelWidth, padding, striding, channelValues[2], channelValues[3], reluScale);
        convData = convolution_relu_block(fheonHEController, fheonANNController, "conv6",  convData, dataWidth, dataSize, kernelWidth, padding, striding, channelValues[3], channelValues[3],  reluScale);
        convData = fheonHEController.bootstrap_function(convData);
        startIn = get_current_time();
        convData = fheonANNController.he_avgpool_optimzed_with_multiple_channels(convData, dataWidth,  channelValues[3], avgpoolSize, avgpoolSize);
        measuringTime.push_back(measureTime(startIn, get_current_time()));
        dataWidth = dataWidth/2;
        dataSize = channelValues[3]*pow(dataWidth, 2);
        // fheonHEController.read_minmax(convData, dataSize);
        // printDuration(inference_time, "avgpool4", false);
        
        // cout<< "Layer 5" << endl;
        convData = convolution_relu_block(fheonHEController, fheonANNController, "conv7", convData, dataWidth, dataSize, kernelWidth, padding, striding, channelValues[3], channelValues[3], reluScale);
        convData = convolution_relu_block(fheonHEController, fheonANNController, "conv8", convData, dataWidth, dataSize, kernelWidth, padding, striding, channelValues[3], channelValues[3], reluScale);
        convData = fheonHEController.bootstrap_function(convData);
        startIn = get_current_time();
        convData = fheonANNController.he_globalavgpool(convData, dataWidth,  channelValues[3], avgpoolSize, rotPositions);
        measuringTime.push_back(measureTime(startIn, get_current_time()));
        // dataWidth = dataWidth/2;
        // dataSize = channelValues[3]* pow(dataWidth, 2);
        // fheonHEController.read_minmax(convData, dataSize);
        // printDuration(inference_time, "avgpool5", false);
        
        // cout << "FC" << endl;
        convData = FClayer_relu_block(fheonHEController, fheonANNController, "fc1", convData, channelValues[3], channelValues[4], reluScale, rotPositions);
        convData = FClayer_relu_block(fheonHEController, fheonANNController, "fc2", convData, channelValues[4], channelValues[4], reluScale, rotPositions);
        convData = FClayer_relu_block(fheonHEController, fheonANNController, "fc3", convData, channelValues[4], channelValues[5], 0, rotPositions);
        
        totalTime(measuringTime);
        measuringTime.clear();
        string infereMessage =  to_string(imageIndex + 1)+"  --  ";          
        printDuration(inference_time, infereMessage, false);
        decryptedData = fheonHEController.decrypt_data(convData, channelValues[5]);
        printPtextVector(decryptedData);
        fheonHEController.read_inferenced_label(convData, channelValues[5], outFile);
    }
    outFile.close();
    cout << "All predicted results printed to File." << endl;
    clear_images(imagesData, numImages);
   return 0;
}

Ctext convolution_relu_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer, Ctext encrytedVector, int &dataWidth, int &dataSize, int kernelWidth, int padding, 
                                int striding, int inputChannels, int outputChannels, int reluScale, bool bootstrapState){
    string dataPath = "./../weights/vgg11/"+layer;
    auto biasVector = load_bias(dataPath+"_bias.csv");
    auto rawKernel = load_weights(dataPath+"_weight.csv", outputChannels, inputChannels, kernelWidth, kernelWidth);
    
    int inputSize = pow(dataWidth, 2);
    vector<vector<Ptext>> kernelData;
    for(int i=0; i<outputChannels; i++){
        auto encodeKernel = fheonHEController.encode_kernel_optimized(rawKernel[i], inputSize);
        kernelData.push_back(encodeKernel);
    }
    rawKernel.clear();
    rawKernel.shrink_to_fit();

    if(bootstrapState){
        encrytedVector = fheonHEController.bootstrap_function(encrytedVector);
    }
    auto convbiasEncoded = fheonHEController.encode_bais_input(biasVector, inputSize);

    startIn = get_current_time();
    auto conv_data = fheonANNController.he_convolution_optimized(encrytedVector, kernelData, convbiasEncoded, dataWidth, inputChannels, outputChannels);
    measuringTime.push_back(measureTime(startIn, get_current_time()));
    // dataWidth = (((dataWidth) + (2*padding) - (kernelWidth - 1) - 1)/striding)+1;
    dataSize = (outputChannels*pow(dataWidth, 2));

    /** Temporal relu scale for max accuracy */
    reluScale = fheonHEController.read_scaling_value(conv_data, dataSize);

    startIn = get_current_time();
    conv_data = fheonANNController.he_relu(conv_data, reluScale, dataSize);
    measuringTime.push_back(measureTime(startIn, get_current_time()));

    // fheonHEController.read_minmax(conv_data, dataSize);
    kernelData.clear();
    kernelData.shrink_to_fit();
    biasVector.clear();
    return conv_data;
}

Ctext FClayer_relu_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer, Ctext encrytedVector, int inputChannels, int outputChannels, int reluScale, int rotPosition){
    string dataPath = "./../weights/vgg11/"+layer;
    auto fc_biasVector = load_bias(dataPath+"_bias.csv");
    auto fc_rawKernel = load_fc_weights(dataPath+"_weight.csv", outputChannels, inputChannels);
    vector<Ptext> fc_kernelData;
    for(int i=0; i < outputChannels; i++){
        auto encodeWeights = fheonHEController.encode_input(fc_rawKernel[i]);
        fc_kernelData.push_back(encodeWeights);
    }

    Ptext fcbaisVector = context->MakeCKKSPackedPlaintext(fc_biasVector, 1);
    encrytedVector = fheonHEController.bootstrap_function(encrytedVector);

    startIn = get_current_time();
    Ctext layer_data = fheonANNController.he_linear(encrytedVector, fc_kernelData, fcbaisVector, inputChannels, outputChannels, rotPosition);
    measuringTime.push_back(measureTime(startIn, get_current_time()));
    
    if(reluScale != 0){
        layer_data = fheonHEController.bootstrap_function(layer_data);
        
        /** Temporal relu scale for max accuracy */
        reluScale = fheonHEController.read_scaling_value(layer_data, outputChannels);

        startIn = get_current_time();
        layer_data = fheonANNController.he_relu(layer_data, reluScale, outputChannels);
        measuringTime.push_back(measureTime(startIn, get_current_time()));
    }
    
    fc_kernelData.clear();
    fc_kernelData.shrink_to_fit();
    fc_biasVector.clear();
    return layer_data;
}