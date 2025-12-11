
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

#ifndef DEFAULT_ARG
#define DEFAULT_ARG 250
#endif

#ifndef INDEX_VALUE
#define INDEX_VALUE 0
#endif

Ctext convolution_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer, Ctext encrytedVector, int &dataWidth, int &dataSize, int kernelWidth, int padding, 
                                    int striding, int inputChannels, int outputChannels, int reluScale, bool bootstrapState);
Ctext shortcut_convolution_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer, Ctext encrytedVector, int &dataWidth, int &dataSize, int inputChannels, int outputChannels);
Ctext resnet_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer, Ctext encrytedVector, int &dataWidth, int &dataSize,
                                    int inputChannels, int outputChannels, int reluScale, int bootstrapState, bool shortcutConv);
Ctext fc_layer_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer, Ctext encrytedVector, int inputChannels, int outputChannels, int rotPosition);
vector<Ctext> double_shortcut_convolution_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer, Ctext &encrytedVector, int &dataWidth, int &dataSize, int inputChannels, int outputChannels);

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
    cout << "---------------------------------RESNET20-------------"<< to_string(DEFAULT_ARG) << "--------------------------" << endl; 
    
    /**** Read the CIFAR-10 Images and inference them */
    int img_cols = 32;
    int img_depth = 3;
    int dataWidth = img_cols;
    int kernelWidth = 3; 
    int padding = 1;
    int striding = 1;
    int avgpoolSize = 8;
    vector<int> channelValues = {16, 32, 64, 10};
    int rotPositions = 16;
    
    //** generate rotation keys for conv_layer 1 */
    auto conv1_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, img_depth, channelValues[0]);
    auto conv2_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channelValues[0], channelValues[0]);
    auto conv3_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channelValues[0], channelValues[1], 2, "single_channel");
    dataWidth = dataWidth/2;
    auto conv4_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channelValues[1], channelValues[1]);
    auto conv5_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channelValues[1], channelValues[2], 2, "single_channel");
    dataWidth = dataWidth/2;
    auto conv6_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channelValues[2], channelValues[2]);

	auto avgpool1_keys = fheonANNController.generate_avgpool_optimized_rotation_positions(dataWidth, channelValues[2], avgpoolSize, avgpoolSize, true, "single_channel", rotPositions);
    auto fc_keys = fheonANNController.generate_linear_rotation_positions(channelValues[3], rotPositions);
    /************************************************************************************************ */
    vector<vector<int>> rotkeys;
    
    rotkeys.push_back(conv1_keys);
    rotkeys.push_back(conv2_keys);
    rotkeys.push_back(conv3_keys);
    rotkeys.push_back(conv4_keys);
    rotkeys.push_back(conv5_keys);
    rotkeys.push_back(conv6_keys);
    rotkeys.push_back(avgpool1_keys);
    rotkeys.push_back(fc_keys);

    /*** join all keys and generate unique values only */
    /*********************************************** Key Generation ******************************************************************************/
    auto begin_rotkeygenerate_time = startTime();
    vector<int> rotation_positions = serialize_rotation_keys(rotkeys);
    cout << "This is the rotation positions (" << rotation_positions.size() << "): "  << rotation_positions << endl;
    fheonHEController.generate_rotation_keys(rotation_positions, "rotation_keys.bin",  true);
    printDuration(begin_rotkeygenerate_time, "Rotation KeyGen Time", false);
    /********************************************************************************************************************************************/;
    
    int numImages = DEFAULT_ARG+INDEX_VALUE;
    dataWidth = img_cols;
    int dataSize = img_depth*pow(img_cols, 2);
    string cifar10tPath = "./../images/cifar-10-batches-bin/test_batch.bin";
    vector<vector<double>> imagesData = read_images(cifar10tPath, numImages, dataSize);
    ofstream outFile;
    outFile.open("./../results/resnet20/fhepredictions.txt", ios_base::app);
    Ctext convData;
    Ptext decryptedData;
    int reluScale = 10;

    for (int imageIndex = 0; imageIndex < numImages; imageIndex++) {
        dataWidth = img_cols;
        dataSize = img_depth*pow(dataWidth, 2);
        auto image = imagesData[imageIndex];
        convData = fheonHEController.encrypt_input(image);
        cout << endl << imageIndex+1 << " - image Read, Normalized and Encrypted with " << image.size() << " Elements" << endl;
        
        /************************************************************************************************ */
        auto inference_time = startTime();
        // cout<< "Layer 0" << endl;
        convData = convolution_block(fheonHEController, fheonANNController, "layer0_conv1", convData, dataWidth, dataSize, kernelWidth, padding, striding, img_depth, channelValues[0], reluScale, false);
        dataSize = channelValues[0]*pow(dataWidth, 2);
        reluScale = fheonHEController.read_scaling_value(convData, dataSize);
        
        startIn = get_current_time();
        convData = fheonANNController.he_relu(convData, reluScale, dataSize, 59);
        measuringTime.push_back(measureTime(startIn, get_current_time()));
        // fheonHEController.read_minmax(convData, dataSize);
		// printDuration(inference_time, "run time", false);

        // cout<< endl<<  "Layer 1" << endl;
        convData = resnet_block(fheonHEController, fheonANNController, "layer1_block1", convData, dataWidth, dataSize, channelValues[0], channelValues[0], reluScale, false, false);
        convData = resnet_block(fheonHEController, fheonANNController, "layer1_block2", convData, dataWidth, dataSize, channelValues[0], channelValues[0], reluScale, true, false);
        convData = resnet_block(fheonHEController, fheonANNController, "layer1_block3", convData, dataWidth, dataSize, channelValues[0], channelValues[0], reluScale, true, false);
        // fheonHEController.read_minmax(convData, dataSize);
		// printDuration(inference_time, "run time", false);

        // cout<< endl<< "Layer 2" << endl;
        convData = resnet_block(fheonHEController, fheonANNController, "layer2_block1", convData, dataWidth, dataSize, channelValues[0], channelValues[1], reluScale, true, true);
        convData = resnet_block(fheonHEController, fheonANNController, "layer2_block2", convData, dataWidth, dataSize, channelValues[1], channelValues[1], reluScale, true, false);
        convData = resnet_block(fheonHEController, fheonANNController, "layer2_block3", convData, dataWidth, dataSize, channelValues[1], channelValues[1], reluScale, true, false);
        // fheonHEController.read_minmax(convData, dataSize);
		// printDuration(inference_time, "run time", false);

        // cout<< endl<<  "Layer 3" << endl;
        convData = resnet_block(fheonHEController, fheonANNController, "layer3_block1", convData, dataWidth, dataSize, channelValues[1], channelValues[2], reluScale, true, true);
        convData = resnet_block(fheonHEController, fheonANNController, "layer3_block2", convData, dataWidth, dataSize, channelValues[2], channelValues[2], reluScale, true, false);
        convData = resnet_block(fheonHEController, fheonANNController, "layer3_block3", convData, dataWidth, dataSize, channelValues[2], channelValues[2], reluScale, true, false);
        // fheonHEController.read_minmax(convData, dataSize);
		// printDuration(inference_time, "run time", false);
        // totalTime(measuringTime);

        // cout<< "Classification" << endl;
        convData = fheonHEController.bootstrap_function(convData);
        startIn = get_current_time();
        convData = fheonANNController.he_globalavgpool(convData, dataWidth,  channelValues[2], avgpoolSize, rotPositions);
        convData = fc_layer_block(fheonHEController, fheonANNController, "layer_fc", convData, channelValues[2], channelValues[3], rotPositions);
        measuringTime.push_back(measureTime(startIn, get_current_time()));
        totalTime(measuringTime);
        measuringTime.clear();

        string infereMessage =  to_string(imageIndex + 1)+"  --  "; 
        printDuration(inference_time, infereMessage, false);
        decryptedData = fheonHEController.decrypt_data(convData, channelValues[3]);
        printPtextVector(decryptedData);
        fheonHEController.read_inferenced_label(convData, channelValues[3], outFile);
    }
    outFile.close();
    cout << "All predicted results printed to File." << endl;
    clear_images(imagesData, numImages);
   return 0;
}

Ctext shortcut_convolution_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer, Ctext encrytedVector, int &dataWidth, int &dataSize, int inputChannels, int outputChannels){
    string dataPath = "./../weights/resnet20/"+layer;

    auto biasVector = load_bias(dataPath+"_bias.csv");
    auto  rawKernel = load_fc_weights(dataPath+"_weight.csv",  outputChannels, inputChannels);
    int width_sq = pow(dataWidth, 2);
    vector<Ptext> kernelData;
    for(int i=0; i < outputChannels; i++){
        auto encodeWeights = fheonHEController.encode_shortcut_kernel(rawKernel[i], width_sq);
        kernelData.push_back(encodeWeights);
    }
    auto biasVectorEncoded = fheonHEController.encode_bais_input(biasVector, width_sq);

    startIn = get_current_time();
    auto conv_data = fheonANNController.he_shortcut_convolution(encrytedVector, kernelData, biasVectorEncoded, dataWidth, inputChannels, outputChannels);
    measuringTime.push_back(measureTime(startIn, get_current_time()));
    
    kernelData.clear();
    kernelData.shrink_to_fit();
    biasVector.clear();
    return conv_data;
}

Ctext convolution_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer, Ctext encrytedVector, int &dataWidth, int &dataSize, int kernelWidth, int padding, 
                                int striding, int inputChannels, int outputChannels, int reluScale, bool bootstrapState){
    string dataPath = "./../weights/resnet20/"+layer;
    auto biasVector = load_bias(dataPath+"_bias.csv");
    auto rawKernel = load_weights(dataPath+"_weight.csv", outputChannels, inputChannels, kernelWidth, kernelWidth);
    int width_sq = pow(dataWidth, 2);
    vector<vector<Ptext>> kernelData;
    int encode_level = encrytedVector->GetLevel();
    for(int i=0; i<outputChannels; i++){
        auto encodeKernel = fheonHEController.encode_kernel_optimized(rawKernel[i], width_sq, encode_level);
        kernelData.push_back(encodeKernel);
    }
    auto biasVectorEncoded = fheonHEController.encode_bais_input(biasVector, width_sq, encode_level);

    startIn = get_current_time();
    auto conv_data = fheonANNController.he_convolution_optimized(encrytedVector, kernelData, biasVectorEncoded, dataWidth, inputChannels, outputChannels, striding);
    measuringTime.push_back(measureTime(startIn, get_current_time()));
    
    kernelData.clear();
    kernelData.shrink_to_fit();
    biasVector.clear();
    return conv_data;
}

vector<Ctext> double_shortcut_convolution_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer, Ctext &encrytedVector, int &dataWidth, int &dataSize, int inputChannels, int outputChannels){
    
    string dataPath = "./../weights/resnet20/"+layer;
    int width_sq = pow(dataWidth, 2);
    int width_out_sq = pow((dataWidth/2), 2);
    int kernelWidth = 3;
    
    /*** convolution data */
    auto biasVector = load_bias(dataPath+"_conv1_bias.csv");
    auto rawKernel = load_weights(dataPath+"_conv1_weight.csv", outputChannels, inputChannels, kernelWidth, kernelWidth);
    vector<vector<Ptext>> kernelData;
    int encode_level = encrytedVector->GetLevel();
    for(int i=0; i<outputChannels; i++){
        auto encodeKernel = fheonHEController.encode_kernel_optimized(rawKernel[i], width_sq, encode_level);
        kernelData.push_back(encodeKernel);
    }
 
    /*** shortcut data */
    auto shortcutbiasVector = load_bias(dataPath+"_shortcut_bias.csv");
    auto  shortcutrawKernel = load_fc_weights(dataPath+"_shortcut_weight.csv",  outputChannels, inputChannels);
    vector<Ptext> shortcutkernelData;
    for(int i=0; i < outputChannels; i++){
        auto encodeWeights = fheonHEController.encode_bais_input(shortcutrawKernel[i], width_sq);
        shortcutkernelData.push_back(encodeWeights);
    }
   
    /*** Bias vectors */
    auto biasVectorEncoded = fheonHEController.encode_bais_input(biasVector, width_out_sq);
	auto shortcutbiasVectorEncoded = fheonHEController.encode_bais_input(shortcutbiasVector, width_out_sq);
    
    startIn = get_current_time();
    auto returnedCiphers = fheonANNController.he_convolution_and_shortcut_optimized(encrytedVector, kernelData, shortcutkernelData, biasVectorEncoded, shortcutbiasVectorEncoded, dataWidth, inputChannels, outputChannels);
    measuringTime.push_back(measureTime(startIn, get_current_time()));

    kernelData.clear();
    kernelData.shrink_to_fit();
    rawKernel.clear();
    rawKernel.shrink_to_fit();
    biasVector.clear();

    shortcutkernelData.clear();
    shortcutkernelData.shrink_to_fit();
    shortcutrawKernel.clear();
    shortcutrawKernel.shrink_to_fit();
    shortcutbiasVector.clear();
    return returnedCiphers;
}

Ctext resnet_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer, Ctext encrytedVector, int &dataWidth, int &dataSize,
                 int inputChannels, int outputChannels, int reluScale, int bootstrapState, bool shortcutConv){
    int kernelWidth = 3; 
    int padding = 1;
    int striding = 1;
    int polyDeg = 59; 

    Ctext shortcut_convData = encrytedVector;
    Ctext convData;
    if(shortcutConv){
        encrytedVector = fheonHEController.bootstrap_function(encrytedVector);
        auto doubleResults = double_shortcut_convolution_block(fheonHEController, fheonANNController, layer, encrytedVector, dataWidth, dataSize, inputChannels, outputChannels);
        dataWidth = dataWidth/2;
        dataSize = (outputChannels*pow(dataWidth, 2));

        convData = doubleResults[0]->Clone();
        shortcut_convData = doubleResults[1]->Clone();
    }
    else{
        convData = convolution_block(fheonHEController, fheonANNController, layer+"_conv1", encrytedVector, dataWidth, dataSize, kernelWidth, padding, striding, inputChannels, outputChannels, reluScale, bootstrapState);
    }
    if(bootstrapState){
        convData = fheonHEController.bootstrap_function(convData);
    }

    reluScale = fheonHEController.read_scaling_value(convData, dataSize);
    startIn = get_current_time();
    convData = fheonANNController.he_relu(convData, reluScale, dataSize, polyDeg);
    measuringTime.push_back(measureTime(startIn, get_current_time()));

    auto second_convData = convolution_block(fheonHEController, fheonANNController, layer+"_conv2", convData, dataWidth, dataSize, kernelWidth, padding, striding, outputChannels, outputChannels, reluScale, bootstrapState);

    Ctext sum_convData = fheonANNController.he_sum_two_ciphertexts(second_convData, shortcut_convData);
    sum_convData = fheonHEController.bootstrap_function(sum_convData);

    reluScale = fheonHEController.read_scaling_value(sum_convData, dataSize);
    startIn = get_current_time();
    sum_convData = fheonANNController.he_relu(sum_convData, reluScale, dataSize, polyDeg);

    measuringTime.push_back(measureTime(startIn, get_current_time()));
    return sum_convData;
}

Ctext fc_layer_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer, Ctext encrytedVector, int inputChannels, int outputChannels, int rotPosition){
    string dataPath = "./../weights/resnet20/"+layer;
    auto fc_biasVector = load_bias(dataPath+"_bias.csv");
    auto fc_rawKernel = load_fc_weights(dataPath+"_weight.csv", outputChannels, inputChannels);
    vector<Ptext> fc_kernelData;
    for(int i=0; i < outputChannels; i++){
        auto encodeWeights = fheonHEController.encode_input(fc_rawKernel[i]);
        fc_kernelData.push_back(encodeWeights);
    }
	Ptext encodedbaisVector = context->MakeCKKSPackedPlaintext(fc_biasVector, 1,  encrytedVector->GetLevel());

    startIn = get_current_time();
    Ctext layer_data = fheonANNController.he_linear_optimized(encrytedVector, fc_kernelData, encodedbaisVector, inputChannels, outputChannels);
    measuringTime.push_back(measureTime(startIn, get_current_time()));
    fc_kernelData.clear();
    fc_kernelData.shrink_to_fit();
    fc_biasVector.clear();
    return layer_data;
}