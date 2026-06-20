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
#include "FHEONANNBatchController.h"

using namespace std;
CryptoContext<DCRTPoly> context;
FHEONHEController fheonHEController(context);

#ifndef DEFAULT_TEST_SIZE
#define DEFAULT_TEST_SIZE 256
#endif

#ifndef INDEX_VALUE
#define INDEX_VALUE 0
#endif

vector<Ctext> shortcut_convolution_block(FHEONHEController &fheonHEController, FHEONANNBatchController fheonANNBatchController, string layer, vector<Ctext>& encrytedInputs, int batchSize, 
                            int &dataWidth, int &dataSize, int inputChannels, int outputChannels, int stride);
vector<Ctext> convolution_block(FHEONHEController &fheonHEController, FHEONANNBatchController fheonANNBatchController, string layer, vector<Ctext>& encrytedInputs, int batchSize, 
                            int &inputdataWidth, int &inputdataSize, int inputChannels, int outputChannels, int stride);
vector<Ctext> resnet_block(FHEONHEController &fheonHEController, FHEONANNBatchController fheonANNBatchController, string layer, vector<Ctext>& encrytedInputs, int batchSize, int &dataWidth, int &dataSize,
                 int inputChannels, int outputChannels, int reluScale, bool bootstrapState, bool shortcutState);
Ctext fc_layer_block(FHEONHEController &fheonHEController, FHEONANNBatchController fheonANNBatchController, string layer, Ctext encrytedInput, int batchSize, int inputChannels, int outputChannels, int rotPositions);

vector<int> measuringTime;
auto startIn = get_current_time();

int batchSize = 16;

int main(int argc, char *argv[]) {

    auto begin_time = startTime();
    printWelcomeMessage();
    /*** Generate the context of the project in the FHEONHEController and pass it to the SnnController */
     int ringDegree = 15;
    int numSlots = 14;
    int circuitDepth = 11;
    int dcrtBits = 46;
    int firstMod = 50;
    int digitSize = 3;
    vector<uint32_t> levelBudget = {3, 3};
    int serialize = true;
    fheonHEController.generate_context(ringDegree, numSlots, circuitDepth, dcrtBits, firstMod, digitSize, levelBudget, serialize);
    context = fheonHEController.getContext();
    FHEONANNController fheonANNController(context);
    FHEONANNBatchController fheonANNBatchController( context);
    printDuration(begin_time, "Context Generation and Keys Serialization", false);
    cout << "---------------------------------RESNET20-------------"<< to_string(DEFAULT_TEST_SIZE) << "--------------------------" << endl; 
    
    /**** Read the CIFAR-10 Images and inference them */
    int img_cols = 32;
    int img_depth = 3;
    int kernelWidth = 3; 
    int paddingLen = 1;
    int stride = 1;
    int shortcutstride = 2;
    int avgpoolSize = 8;
    vector<int> channels = {3, 16, 32, 64, 10};
    vector<int> dataWidths = {32, 16, 8, 1};
    vector<int> dataSizes = {1024, 256, 64};
    int rotPositions = 16;
    
    //** generate rotation keys for conv_layer 1 */
    auto conv1_keys = fheonANNBatchController.generate_convolution_batch_rotation_positions(batchSize, dataWidths[0],  kernelWidth, paddingLen, stride);
    auto conv2_keys = fheonANNBatchController.generate_convolution_batch_rotation_positions(batchSize, dataWidths[0],  kernelWidth, paddingLen, shortcutstride);
    auto conv3_keys = fheonANNBatchController.generate_convolution_batch_rotation_positions(batchSize, dataWidths[1],  kernelWidth, paddingLen, stride);
    auto conv4_keys = fheonANNBatchController.generate_convolution_batch_rotation_positions(batchSize, dataWidths[1],  kernelWidth, paddingLen, shortcutstride);
    auto avg_keys = fheonANNBatchController.generate_avgpool_batch_optimized_rotation_positions(batchSize, dataWidths[2], avgpoolSize, avgpoolSize, true, rotPositions);
    auto convert_keys = fheonANNBatchController.generate_batch_inputs_converter_rotation_positions(batchSize,  channels[3], dataWidths[3]);
    auto fc_keys = fheonANNBatchController.generate_linear_batch_rotation_positions(batchSize, {channels[3], channels[4]}, channels[3], rotPositions);
    /************************************************************************************************ */
    vector<vector<int>> rotkeys;
    
    rotkeys.push_back(conv1_keys);
    rotkeys.push_back(conv2_keys);
    rotkeys.push_back(conv3_keys);
    rotkeys.push_back(conv4_keys);
    rotkeys.push_back(avg_keys);
    rotkeys.push_back(convert_keys);
    rotkeys.push_back(fc_keys);


    cout << "conv1_keys (" << conv1_keys.size() << ") " << conv1_keys << endl;
    cout << "conv2_keys (" << conv2_keys.size() << ") " << conv2_keys << endl;
    cout << "conv3_keys (" << conv3_keys.size() << ") " << conv3_keys << endl;
    cout << "conv4_keys (" << conv4_keys.size() << ") " << conv4_keys << endl;
    cout << "avg_keys (" << avg_keys.size() << ") " << avg_keys << endl;
    cout << "convert_keys (" << convert_keys.size() << ") " << convert_keys << endl;
    cout << "fc_keys (" << fc_keys.size() << ") " << fc_keys << endl;


    /*** join all keys and generate unique values only */
    /*********************************************** Key Generation ******************************************************************************/
    auto begin_rotkeygen_time = startTime();
    vector<int> rotation_positions = serialize_rotation_keys(rotkeys);
    cout << "This is the rotation positions (" << rotation_positions.size() << "): "  << rotation_positions << endl;
    fheonHEController.generate_rotation_keys(rotation_positions, "rotation_keys.bin",  true);
    printDuration(begin_rotkeygen_time, "Rotation KeyGen Time", false);
    /********************************************************************************************************************************************/;
    
    int numImages = DEFAULT_TEST_SIZE+INDEX_VALUE;
    int dataSize = img_depth*pow(img_cols, 2);
    string cifar10tPath = "./../images/cifar-10-batches-bin/test_batch.bin";
    vector<vector<double>> imagesData = read_images(cifar10tPath, numImages, dataSize);
    ofstream outFile;
    outFile.open("./../results/TresNet20/fhepredictions.txt", ios_base::app);
    Ctext convData;
    Ptext decryptedData;
    int polyDegee = 59;
    int reluScale = 10;
    int bootstrap_level = 2;

    for (int idx = 0; idx < numImages; idx++) {
        int imgIdx = idx*batchSize; 
        vector<vector<double>> batchedImages; 
        for (int b = 0; b < batchSize; b++) {
            auto img = imagesData[imgIdx + b];
            batchedImages.push_back(img);
        }

        /** adjust images */
        auto inputDatas = convert_inputData(batchedImages, batchSize, channels[0], dataSizes[0]);
        vector<Ctext>encryptedInputs;
        for(int i=0; i<channels[0]; i++){
            encryptedInputs.push_back(fheonHEController.encrypt_input(inputDatas[i]));
        }
        cout << endl << imgIdx+1  << " to " << imgIdx+batchSize << " - (" << encryptedInputs.size() << " input channes) images Read, Normalized and Encrypt"<< endl;
        
        /************************************************************************************************ */


        auto inference_time = startTime();
        cout<< "Layer 0" << endl;
        auto convData = convolution_block(fheonHEController, fheonANNBatchController,  "layer0_conv1", encryptedInputs, batchSize, dataWidths[0], dataSizes[0], channels[0], channels[1], stride);

        int totalSize = batchSize * dataSizes[0];
        auto scalingVals = fheonHEController.read_batch_scaling_values(convData, channels[1], totalSize);
    
        startIn = get_current_time();
        convData = fheonANNBatchController.he_batch_relu(convData, scalingVals, channels[1], totalSize, polyDegee);
        measuringTime.push_back(measureTime(startIn, get_current_time()));
        
        fheonHEController.read_batch_minmax(convData, channels[1], totalSize);
        printDuration(inference_time, "run time", false);

        // cout<< endl<<  "Layer 1" << endl;
        // cout <<"Block 1 " << endl;
        // convData = resnet_block(fheonHEController, fheonANNBatchController, "layer1_block1", convData, batchSize, dataWidths[0], dataSizes[0], channels[1], channels[1], reluScale, false, false);
        // fheonHEController.read_batch_minmax(convData, channels[1], totalSize);
        // cout <<"Block 2 " << endl;
        // convData = resnet_block(fheonHEController, fheonANNBatchController, "layer2_block1", convData, batchSize, dataWidths[0], dataSizes[0], channels[1], channels[1], reluScale, true, false);
        // cout <<"Block 3 " << endl;
        // convData = resnet_block(fheonHEController, fheonANNBatchController, "layer3_block1", convData, batchSize, dataWidths[0], dataSizes[0], channels[1], channels[1], reluScale, true, false);
        // fheonHEController.read_batch_minmax(convData, channels[1], totalSize);
        // printDuration(inference_time, "run time", false);

        cout<< endl<< "Layer 2" << endl;
        cout <<"Block 1 " << endl;
        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer2_block1", convData, batchSize, dataWidths[0], dataSizes[0], channels[1], channels[2], reluScale, true, true);
        totalSize = batchSize * dataSizes[1];
        fheonHEController.read_batch_minmax(convData, channels[2], totalSize);
        cout <<"Block 2 " << endl;
        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer2_block2", convData, batchSize, dataWidths[1], dataSizes[1], channels[2], channels[2], reluScale, true, false);
         cout <<"Block 3 " << endl;
        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer2_block3", convData, batchSize, dataWidths[1], dataSizes[1], channels[2], channels[2], reluScale, true, false);
        
        fheonHEController.read_batch_minmax(convData, channels[2], totalSize);
        printDuration(inference_time, "run time", false);

        cout<< endl<<  "Layer 3" << endl;
        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer3_block1", convData, batchSize, dataWidths[1], dataSizes[1], channels[2], channels[3], reluScale, true, true);
        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer3_block2", convData, batchSize, dataWidths[2], dataSizes[2], channels[3], channels[3], reluScale, true, false);
        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer3_block3", convData, batchSize, dataWidths[2], dataSizes[2], channels[3], channels[3], reluScale, true, false);
        totalSize = batchSize * dataSizes[2];
        fheonHEController.read_batch_minmax(convData, channels[3], totalSize);
        printDuration(inference_time, "run time", false);
        

        cout<< "Classification" << endl;
        convData = fheonHEController.batch_bootstrap_function(convData, channels[3], bootstrap_level);
        
        startIn = get_current_time();
        convData = fheonANNBatchController.he_batch_globalpool(convData, batchSize, dataWidths[2], channels[3], avgpoolSize, rotPositions);
        auto fcData = fheonANNBatchController.he_batch_inputs_converter(convData, batchSize, channels[3], dataWidths[3]);
        fcData = fc_layer_block(fheonHEController, fheonANNBatchController, "layer_fc", fcData, batchSize, channels[3], channels[4], rotPositions);
        measuringTime.push_back(measureTime(startIn, get_current_time()));
        totalTime(measuringTime);
        measuringTime.clear();

        string infereMessage = "Batch Size ("+ to_string(batchSize) +") -- Total Run Time for Images " + to_string(imgIdx + 1) + " - " +  to_string(imgIdx+1+batchSize);  
        printDuration(inference_time, infereMessage, false);
        auto predictions = fheonHEController.read_batch_inferenced_label(fcData, batchSize, channels[4], outFile);

        cout << "Batch Predictions: " << predictions << endl;
    }
    outFile.close();
    cout << "All predicted results printed to File." << endl;
    clear_images(imagesData, numImages);
   return 0;
}


// vector<Ctext> shortcut_convolution_block(FHEONHEController &fheonHEController, FHEONANNBatchController fheonANNBatchController, string layer, vector<Ctext>& encrytedInputs, int batchSize, 
//                             int &dataWidth, int &dataSize, int inputChannels, int outputChannels, int stride){
    
//     int outputWidth = dataWidth/stride;
//     int outputSize = pow(outputWidth, 2);
//     int totalOutputSize = batchSize*pow(outputWidth, 2);
//     int totalInputSize = batchSize * dataSize;
    
//     /******** DOUBLE CHECK THIS KERNEL ENCODING */
//     string dataPath = "./../weights/resnet20/"+layer;
//     auto rawKernelData = load_shortcut_batch_weights(dataPath+"_shortcut_weight.csv", batchSize, outputChannels, inputChannels, dataSize); 
//     auto rawBiasData = load_batch_bias(dataPath+"_shortcut_bias.csv", outputChannels, batchSize, outputSize);
//     int encode_level = encrytedInputs[0]->GetLevel();
//     vector<vector<Ptext>> shortcutkernelData;
//     vector<Ptext> biasData;

//     for(int i=0; i < outputChannels; i++){
//         vector<Ptext> inconData; 
//         for(int j=0; j<inputChannels; j++){
//             auto encodeWeights = fheonHEController.encode_input(rawKernelData[i][j], totalInputSize, encode_level);
//             inconData.push_back(encodeWeights);
//         }
//         shortcutkernelData.push_back(inconData);
//         // cout <<"i: " << i << " - totalOutputSize: " << totalOutputSize << " - rawBiasData: " << rawBiasData[i].size() << endl;
//         auto biasVectorEncoded = fheonHEController.encode_input(rawBiasData[i], totalOutputSize, encode_level);
//         biasData.push_back(biasVectorEncoded);
        
//     }

//     startIn = get_current_time();
//     auto conv_data = fheonANNBatchController.he_batch_convolution_shortcut_optimized(encrytedInputs, shortcutkernelData, biasData, batchSize, dataWidth, inputChannels, outputChannels, stride);
//     measuringTime.push_back(measureTime(startIn, get_current_time()));
    
//     rawKernelData.clear();
//     rawKernelData.shrink_to_fit();
//     shortcutkernelData.clear();
//     shortcutkernelData.shrink_to_fit();
//     rawBiasData.clear();
//     rawBiasData.shrink_to_fit();
//     biasData.clear();
//     biasData.shrink_to_fit();
//     return conv_data;
// }



// vector<Ctext> convolution_block(FHEONHEController &fheonHEController, FHEONANNBatchController fheonANNBatchController, string layer, vector<Ctext>& encrytedInputs, int batchSize, 
//                             int &dataWidth, int &dataSize, int inputChannels, int outputChannels, int stride){
    
    
//     int kernelWidth = 3;
//     int outputWidth = dataWidth/stride;
//     int outputSize = pow(outputWidth, 2);
//     int totalOutSize = batchSize*outputSize;
//     {
//         string dataPath = "./../weights/resnet20/"+layer;
//         auto rawKernelData = load_batch_weights(dataPath+"_weight.csv", outputChannels, inputChannels, batchSize, kernelWidth, kernelWidth);
//         auto rawBiasData = load_batch_bias(dataPath+"_bias.csv", outputChannels, batchSize, outputSize);
//         int encode_level = encrytedInputs[0]->GetLevel();
//         cout << "we got here" << endl;

//         vector<vector<vector<Ptext>>> kernelData;
//         vector<Ptext> biasData; 
//         for(int i=0; i<outputChannels; i++){
//             vector<vector<Ptext>> inconData; 
//             for(int j=0; j<inputChannels; j++){
//                 auto encodeKernel = fheonHEController.optimized_encode_kernel(rawKernelData[i][j], dataSize);
//                 inconData.push_back(encodeKernel);
//                 cout << "j: "<<j << endl; 
//             }
//             auto inbiasEncoded = fheonHEController.encode_input(rawBiasData[i], totalOutSize, encode_level);
//             kernelData.push_back(inconData);
//             biasData.push_back(inbiasEncoded);
//             cout << "outputChannels: "<<i << endl; 
//         }

//         rawKernelData.clear();
//         // rawKernelData.shrink_to_fit();
//         rawBiasData.clear();
//         // rawBiasData.shrink_to_fit();

//         startIn = get_current_time();
//         auto conv_data = fheonANNBatchController.he_batch_convolution_optimized(encrytedInputs, kernelData, biasData, batchSize, dataWidth, inputChannels, outputChannels, stride);
//         measuringTime.push_back(measureTime(startIn, get_current_time()));
        
//         kernelData.clear();
//         // kernelData.shrink_to_fit();
//         biasData.clear();
//         // biasData.shrink_to_fit();
//         return conv_data;
//     }
// }

vector<Ctext> shortcut_convolution_block(FHEONHEController &fheonHEController, FHEONANNBatchController fheonANNBatchController, string layer, vector<Ctext>& encrytedInputs, int batchSize, 
                            int &dataWidth, int &dataSize, int inputChannels, int outputChannels, int stride){
    
    int outputWidth = dataWidth/stride;
    int outputSize = pow(outputWidth, 2);
    
    /******** DOUBLE CHECK THIS KERNEL ENCODING */
    string dataPath = "./../weights/resnet20/"+layer;
    auto rawKernelData = load_shortcut_batch_weights(dataPath+"_shortcut_weight.csv", batchSize, outputChannels, inputChannels, dataSize); 
    auto rawBiasData = load_batch_bias(dataPath+"_shortcut_bias.csv", outputChannels, batchSize, outputSize);

    startIn = get_current_time();
    auto conv_data = fheonANNBatchController.he_batch_convolution_shortcut_optimized(fheonHEController, encrytedInputs, rawKernelData, rawBiasData, batchSize, dataWidth, inputChannels, outputChannels, stride);
    measuringTime.push_back(measureTime(startIn, get_current_time()));
    
    rawKernelData.clear();
    rawKernelData.shrink_to_fit();
    rawBiasData.clear();
    rawBiasData.shrink_to_fit();
    return conv_data;
}

vector<Ctext> convolution_block(FHEONHEController &fheonHEController, FHEONANNBatchController fheonANNBatchController, string layer, vector<Ctext>& encrytedInputs, int batchSize, 
                            int &dataWidth, int &dataSize, int inputChannels, int outputChannels, int stride){
    
    
    int kernelWidth = 3;
    int outputWidth = dataWidth/stride;
    int outputSize = pow(outputWidth, 2);
    string dataPath = "./../weights/resnet20/"+layer;
    auto rawKernelData = load_batch_weights(dataPath+"_weight.csv", outputChannels, inputChannels, batchSize, kernelWidth, kernelWidth);
    auto rawBiasData = load_batch_bias(dataPath+"_bias.csv", outputChannels, batchSize, outputSize);

    startIn = get_current_time();
    auto conv_data = fheonANNBatchController.he_batch_convolution_optimized(fheonHEController, encrytedInputs, rawKernelData, rawBiasData, batchSize, dataWidth, inputChannels, outputChannels, stride);
    measuringTime.push_back(measureTime(startIn, get_current_time()));
        
    rawKernelData.clear();
    rawBiasData.clear();

    return conv_data;
}


vector<Ctext> resnet_block(FHEONHEController &fheonHEController, FHEONANNBatchController fheonANNBatchController, string layer, vector<Ctext>& encrytedInputs, int batchSize, int &inputdataWidth, int &inputdataSize,
                 int inputChannels, int outputChannels, int reluScale, bool bootstrapState, bool shortcutState){

    int stride = 1;
    int shortcutStridindLen = 2; 
    int polyDegee = 59; 
    int bootstrap_level= 2;
    int dataWidth = inputdataWidth;
    int dataSize = inputdataSize;

    vector<Ctext> shortcuts = encrytedInputs;
    vector<Ctext> convData;

    if(shortcutState){
        encrytedInputs = fheonHEController.batch_bootstrap_function(encrytedInputs, inputChannels, bootstrap_level);
        
        cout << "convolution" << endl;
        convData = convolution_block(fheonHEController, fheonANNBatchController, layer+"_conv1", encrytedInputs, batchSize, dataWidth, dataSize, inputChannels, outputChannels, shortcutStridindLen);
        cout << "convolution" << endl;
        cout << "shortcut" << endl;
        shortcuts = shortcut_convolution_block(fheonHEController, fheonANNBatchController, layer, encrytedInputs, batchSize, dataWidth, dataSize, inputChannels, outputChannels, shortcutStridindLen);


        dataWidth = dataWidth/2;
        cout << "dataWidth: " << dataWidth << " dataSize: " << dataSize << endl;
        dataSize = pow(dataWidth, 2);
        
        cout << "dataWidth: " << dataWidth << " dataSize: " << dataSize << endl;

    }
    else{
        convData = convolution_block(fheonHEController, fheonANNBatchController, layer+"_conv1", encrytedInputs, batchSize, dataWidth, dataSize, inputChannels, outputChannels, stride);
    }
    if(bootstrapState){
        convData = fheonHEController.batch_bootstrap_function(convData, outputChannels, bootstrap_level);
    }



    int totalSize = (batchSize*dataSize); 
    auto scalingVals = fheonHEController.read_batch_scaling_values(convData, outputChannels, totalSize);
    cout << "Scaling Values: " << scalingVals << endl;

    

    startIn = get_current_time();
    convData = fheonANNBatchController.he_batch_relu(convData, scalingVals, outputChannels, totalSize, polyDegee);
    measuringTime.push_back(measureTime(startIn, get_current_time()));

    auto second_convData = convolution_block(fheonHEController, fheonANNBatchController, layer+"_conv2", convData, batchSize, dataWidth, dataSize, outputChannels, outputChannels, stride);

    vector<Ctext> sum_convData = fheonANNBatchController.he_batch_sum_ciphertexts(second_convData, shortcuts, outputChannels);
    sum_convData = fheonHEController.batch_bootstrap_function(sum_convData, outputChannels, bootstrap_level);

    scalingVals = fheonHEController.read_batch_scaling_values(sum_convData, outputChannels, totalSize);
    cout << "Scaling Values: " << scalingVals << endl;
    
    startIn = get_current_time();
    sum_convData = fheonANNBatchController.he_batch_relu(sum_convData, scalingVals, outputChannels, totalSize, polyDegee);
    measuringTime.push_back(measureTime(startIn, get_current_time()));

    return sum_convData;
}

Ctext fc_layer_block(FHEONHEController &fheonHEController, FHEONANNBatchController fheonANNBatchController, string layer, Ctext encrytedInput, int batchSize, int inputChannels, int outputChannels, int rotPositions){
   
    string dataPath = "./../weights/resnet20/"+layer;
    auto fc_rawKernelData = load_batch_fc_weights(dataPath+"FC1_weight.csv", outputChannels, batchSize, inputChannels);
    auto fc_rawBiasData = load_batch_fc_bias(dataPath+"FC1_bias.csv", outputChannels, batchSize);
    vector<Ptext> fc_kernelData;
    for(int i=0; i < outputChannels; i++){
        // cout << "fc kernel size: " << fc1_rawKernelData[i].size() << " -- bais size: " << fc1_BiasData.size() << endl;
        auto encodeWeights = fheonHEController.encode_input(fc_rawKernelData[i]);
        fc_kernelData.push_back(encodeWeights);
    }
    Ptext fc_BiasData = fheonHEController.encode_input(fc_rawBiasData);


    startIn = get_current_time();
    Ctext fcData = fheonANNBatchController.he_batch_linear(fcData, fc_kernelData, fc_BiasData, batchSize, inputChannels, outputChannels, rotPositions);
    measuringTime.push_back(measureTime(startIn, get_current_time()));
    fc_kernelData.clear();
    fc_kernelData.shrink_to_fit();
    return fcData;
}