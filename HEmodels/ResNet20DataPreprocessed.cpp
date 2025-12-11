
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

void convolution_data_processing(FHEONHEController &fheonHEController, string layer,  vector<vector<Ptext>>& kernelData, Ptext &baisData, int &dataWidth, int inputChannels, int outputChannels, int kernelWidth);
void shorcut_data_processing(FHEONHEController &fheonHEController, string layer, vector<vector<Ptext>>& kernelData, Ptext &baisData, int &dataWidth, int inputChannels, int outputChannels);
void double_data_processing(FHEONHEController &fheonHEController, string layer,  vector<vector<Ptext>>&  convkernelData, Ptext &convbaisData, vector<vector<Ptext>>& shortcutkernelData, Ptext &shortcutbaisData,
                    int &dataWidth, int inputChannels, int outputChannels, int kernelWidth);
void fclayer_data_processing(FHEONHEController &fheonHEController, string layer, vector<Ptext>& fc_kernelData, Ptext& fc_baisData, int inputChannels, int outputChannels);

Ctext convolution_block(FHEONANNController &fheonANNController, Ctext& encrytedData, vector<vector<Ptext>>& kernelData, Ptext& baisData, int &dataWidth, int inputChannels, int outputChannels, int striding = 1);
Ctext resnet_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, Ctext& encrytedData, vector<vector<vector<Ptext>>>& blockkernelData, vector<Ptext>& blockbaisData, 
                                                int &dataWidth, int &dataSize, int inputChannels, int outputChannels, int reluScale, int bootstrapState, bool shortcutConv);
Ctext FClayer_block(FHEONANNController &fheonANNController, Ctext& encrytedData, vector<Ptext>& fc_kernelData, Ptext& fc_baisData, int inputChannels, int outputChannels, int rotPosition);

vector<int> measuringTime;
auto startIn = get_current_time();
int striding = 1;
int padding = 1;

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

    int avgpoolSize = 8;
    vector<int> channelValues = {16, 32, 64, 10};
    int rotPositions = 16;
    
    //** generate rotation keys for conv_layer 1 */
    auto conv1_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, img_depth, channelValues[0]);
    auto conv2_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channelValues[0], channelValues[0]);
    auto conv3_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channelValues[0], channelValues[1], 2);
    dataWidth = dataWidth/2;
    auto conv4_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channelValues[1], channelValues[1]);
    auto conv5_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channelValues[1], channelValues[2], 2);
    dataWidth = dataWidth/2;
    auto conv6_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channelValues[2], channelValues[2]);
    
	auto avgpool1_keys = fheonANNController.generate_avgpool_optimized_rotation_positions(dataWidth, channelValues[2], avgpoolSize, avgpoolSize, true, "multi_channels", rotPositions);
    auto fc_keys = fheonANNController.generate_linear_rotation_positions(channelValues[3], rotPositions);
    /************************************************************************************************ */
    vector<vector<int>> rotation_keys;
    rotation_keys.push_back(conv1_keys);
    rotation_keys.push_back(conv2_keys);
    rotation_keys.push_back(conv3_keys);
    rotation_keys.push_back(conv4_keys);
    rotation_keys.push_back(conv5_keys);
    rotation_keys.push_back(conv6_keys);
    rotation_keys.push_back(avgpool1_keys);
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
    cout << "This is the rotation positions (" << rotation_positions.size() <<"): " << rotation_positions << endl;
    fheonHEController.generate_rotation_keys(rotation_positions, "rotation_keys.bin", true);
    printDuration(begin_rotkeygenerate_time, "Rotation KeyGen (position, gen, and load) Time", false);
    /********************************************************************************************************************************************/

    /********************************************************MODEL WEIGHTS AND BAISES DATA PROCESSING ******************************************/
    vector<string> layer_names = {  "layer0_conv1",
                                    "layer1_block1",
                                    "layer1_block2",
                                    "layer1_block3",
                                    "layer2_block1",
                                    "layer2_block2",
                                    "layer2_block3",
                                    "layer3_block1",
                                    "layer3_block2",
                                    "layer3_block3",
                                    "layer_fc"
                                };

    vector<vector<Ptext>> conv0_kernelData;
    Ptext conv0_baisData;
    vector<vector<vector<vector<Ptext>>>> resnet_block1_kernelData(9, vector<vector<vector<Ptext>>>(3));
    vector<vector<Ptext>> resnet_block1_baisData(9, vector<Ptext>(3));
    vector<Ptext> fc_kernelData;
    Ptext fc_baisData;

    dataWidth = img_cols;
    convolution_data_processing(fheonHEController, layer_names[0], conv0_kernelData, conv0_baisData, dataWidth, img_depth,  channelValues[0], kernelWidth);
    
    /*** Block 1 Data */
    convolution_data_processing(fheonHEController, layer_names[1] + "_conv1", resnet_block1_kernelData[0][0], resnet_block1_baisData[0][0], dataWidth, channelValues[0], channelValues[0], kernelWidth);
    convolution_data_processing(fheonHEController, layer_names[1] + "_conv2", resnet_block1_kernelData[0][1], resnet_block1_baisData[0][1], dataWidth, channelValues[0], channelValues[0], kernelWidth);
    convolution_data_processing(fheonHEController, layer_names[2] + "_conv1", resnet_block1_kernelData[1][0], resnet_block1_baisData[1][0], dataWidth, channelValues[0], channelValues[0], kernelWidth);
    convolution_data_processing(fheonHEController, layer_names[2] + "_conv2", resnet_block1_kernelData[1][1], resnet_block1_baisData[1][1], dataWidth, channelValues[0], channelValues[0], kernelWidth);
    convolution_data_processing(fheonHEController, layer_names[3] + "_conv1", resnet_block1_kernelData[2][0], resnet_block1_baisData[2][0], dataWidth, channelValues[0], channelValues[0], kernelWidth);
    convolution_data_processing(fheonHEController, layer_names[3] + "_conv2", resnet_block1_kernelData[2][1], resnet_block1_baisData[2][1], dataWidth, channelValues[0], channelValues[0], kernelWidth);

    /*** Block 2 Data */
    double_data_processing(fheonHEController, layer_names[4], resnet_block1_kernelData[3][0], resnet_block1_baisData[3][0], resnet_block1_kernelData[3][2], resnet_block1_baisData[3][2], dataWidth, 
                                channelValues[0], channelValues[1], kernelWidth);
    convolution_data_processing(fheonHEController, layer_names[4] + "_conv2", resnet_block1_kernelData[3][1], resnet_block1_baisData[3][1], dataWidth, channelValues[1], channelValues[1], kernelWidth);
    
    convolution_data_processing(fheonHEController, layer_names[5] + "_conv1", resnet_block1_kernelData[4][0], resnet_block1_baisData[4][0], dataWidth, channelValues[1], channelValues[1], kernelWidth);
    convolution_data_processing(fheonHEController, layer_names[5] + "_conv2", resnet_block1_kernelData[4][1], resnet_block1_baisData[4][1], dataWidth, channelValues[1], channelValues[1], kernelWidth);
    convolution_data_processing(fheonHEController, layer_names[6] + "_conv1", resnet_block1_kernelData[5][0], resnet_block1_baisData[5][0], dataWidth, channelValues[1], channelValues[1], kernelWidth);
    convolution_data_processing(fheonHEController, layer_names[6] + "_conv2", resnet_block1_kernelData[5][1], resnet_block1_baisData[5][1], dataWidth, channelValues[1], channelValues[1], kernelWidth);

      /*** Block 3 Data */
    double_data_processing(fheonHEController, layer_names[7], resnet_block1_kernelData[6][0], resnet_block1_baisData[6][0], resnet_block1_kernelData[6][2], resnet_block1_baisData[6][2], dataWidth, 
                                channelValues[1], channelValues[2], kernelWidth);
    convolution_data_processing(fheonHEController, layer_names[7] + "_conv2", resnet_block1_kernelData[6][1], resnet_block1_baisData[6][1], dataWidth, channelValues[2], channelValues[2], kernelWidth);
    
    convolution_data_processing(fheonHEController, layer_names[8] + "_conv1", resnet_block1_kernelData[7][0], resnet_block1_baisData[7][0], dataWidth, channelValues[2], channelValues[2], kernelWidth);
    convolution_data_processing(fheonHEController, layer_names[8] + "_conv2", resnet_block1_kernelData[7][1], resnet_block1_baisData[7][1], dataWidth, channelValues[2], channelValues[2], kernelWidth);
    convolution_data_processing(fheonHEController, layer_names[9] + "_conv1", resnet_block1_kernelData[8][0], resnet_block1_baisData[8][0], dataWidth, channelValues[2], channelValues[2], kernelWidth);
    convolution_data_processing(fheonHEController, layer_names[9] + "_conv2", resnet_block1_kernelData[8][1], resnet_block1_baisData[8][1], dataWidth, channelValues[2], channelValues[2], kernelWidth);

    fclayer_data_processing(fheonHEController, layer_names[10], fc_kernelData, fc_baisData, channelValues[2], channelValues[3]);

    int numImages = DEFAULT_ARG+INDEX_VALUE;
    int dataSize = img_depth*pow(img_cols, 2);
    string cifar10tPath = "./../images/cifar-10-batches-bin/test_batch.bin";
    vector<vector<double>> imagesData = read_images(cifar10tPath, numImages, dataSize);
    ofstream outFile;
    outFile.open("./../results/resnet20/fhepredictions.txt", ios_base::app);
    Ctext convData;
    Ptext decryptedData;
    int reluScale = 10;

    for (int imageIndex = INDEX_VALUE; imageIndex < numImages; imageIndex++) {
        dataWidth = img_cols;
        dataSize = img_depth*pow(dataWidth, 2);
        auto image = imagesData[imageIndex];

        // display_image(image, imageSize, true);
        Ctext encryptedImage = fheonHEController.encrypt_input(image);
        cout << endl << imageIndex+1 << " - image Read, Normalized and Encrypted with " << image.size() << " Elements" << endl;
        /************************************************************************************************ */
        
        auto inference_time = startTime();
        // cout<< "Layer 0" << endl;
        convData = convolution_block(fheonANNController, encryptedImage, conv0_kernelData, conv0_baisData, dataWidth, img_depth, channelValues[0]);
        dataSize = channelValues[0]*pow(dataWidth, 2);
        reluScale = fheonHEController.read_scaling_value(convData, dataSize);

        startIn = get_current_time();
        convData = fheonANNController.he_relu(convData, reluScale, dataSize);
        measuringTime.push_back(measureTime(startIn, get_current_time()));

        // cout<< endl<<  "Layer 1" << endl;
        convData = resnet_block(fheonHEController, fheonANNController, convData,  resnet_block1_kernelData[0], resnet_block1_baisData[0], dataWidth, dataSize, channelValues[0], channelValues[0], reluScale, false, false);
        convData = resnet_block(fheonHEController, fheonANNController, convData,  resnet_block1_kernelData[1], resnet_block1_baisData[1], dataWidth, dataSize, channelValues[0], channelValues[0], reluScale, true, false);
        convData = resnet_block(fheonHEController, fheonANNController, convData,  resnet_block1_kernelData[2], resnet_block1_baisData[2], dataWidth, dataSize, channelValues[0], channelValues[0], reluScale, true, false);
        // fheonHEController.read_minmax(convData, dataSize);
		// printDuration(inference_time, "run time", false);

        // cout<< endl<< "Layer 2" << endl;
        convData = resnet_block(fheonHEController, fheonANNController, convData,  resnet_block1_kernelData[3], resnet_block1_baisData[3], dataWidth, dataSize, channelValues[0], channelValues[1], reluScale, true, true);
        convData = resnet_block(fheonHEController, fheonANNController, convData,  resnet_block1_kernelData[4], resnet_block1_baisData[4], dataWidth, dataSize, channelValues[1], channelValues[1], reluScale, true, false);
        convData = resnet_block(fheonHEController, fheonANNController, convData,  resnet_block1_kernelData[5], resnet_block1_baisData[5], dataWidth, dataSize, channelValues[1], channelValues[1], reluScale, true, false);
        // fheonHEController.read_minmax(convData, dataSize);
		// printDuration(inference_time, "run time", false);

        // cout<< endl<<  "Layer 3" << endl;
        convData = resnet_block(fheonHEController, fheonANNController, convData,  resnet_block1_kernelData[6], resnet_block1_baisData[6], dataWidth, dataSize, channelValues[1], channelValues[2], reluScale, true, true);
        convData = resnet_block(fheonHEController, fheonANNController, convData,  resnet_block1_kernelData[7], resnet_block1_baisData[7], dataWidth, dataSize, channelValues[2], channelValues[2], reluScale, true, false);
        convData = resnet_block(fheonHEController, fheonANNController, convData,  resnet_block1_kernelData[8], resnet_block1_baisData[8], dataWidth, dataSize, channelValues[2], channelValues[2], reluScale, true, false);
        // fheonHEController.read_minmax(convData, dataSize);
		// printDuration(inference_time, "run time", false);

        // cout<< "Classification" << endl;
        convData = fheonHEController.bootstrap_function(convData);
        startIn = get_current_time();
        convData = fheonANNController.he_globalavgpool(convData, dataWidth,  channelValues[2], avgpoolSize, rotPositions);
        convData = FClayer_block(fheonANNController, convData, fc_kernelData, fc_baisData, channelValues[2], channelValues[3], rotPositions);
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

void convolution_data_processing(FHEONHEController &fheonHEController, string layer, vector<vector<Ptext>>& kernelData, Ptext &baisData,
                    int &dataWidth, int inputChannels, int outputChannels, int kernelWidth){
    
    string dataPath = "./../weights/resnet20/"+layer;
    auto biasVector = load_bias(dataPath+"_bias.csv");
    auto rawKernel = load_weights(dataPath+"_weight.csv", outputChannels, inputChannels, kernelWidth, kernelWidth);
    int width_sq = pow(dataWidth, 2);
    for(int i=0; i<outputChannels; i++){
        auto encodeKernel = fheonHEController.encode_kernel_optimized(rawKernel[i], width_sq);
        kernelData.push_back(encodeKernel);
    }
    baisData = fheonHEController.encode_bais_input(biasVector, width_sq);

    rawKernel.clear();
    biasVector.clear();
}

void shorcut_data_processing(FHEONHEController &fheonHEController, string layer, vector<vector<Ptext>>& kernelData, Ptext &baisData,
                    int &dataWidth, int inputChannels, int outputChannels){
    
    string dataPath = "./../weights/resnet20/"+layer;
    auto biasVector = load_bias(dataPath+"_bias.csv");
    auto  rawKernel = load_fc_weights(dataPath+"_weight.csv",  outputChannels, inputChannels);
    int width_sq = pow(dataWidth, 2);
    
    vector<Ptext> shData;
    for(int i=0; i < outputChannels; i++){
        auto encodeWeights = fheonHEController.encode_bais_input(rawKernel[i], width_sq);
        shData.push_back(encodeWeights);
    }
    kernelData.push_back(shData);

    dataWidth = dataWidth/2;
    width_sq = pow(dataWidth, 2);
    baisData = fheonHEController.encode_bais_input(biasVector, width_sq);

    rawKernel.clear();
    biasVector.clear();
}

void double_data_processing(FHEONHEController &fheonHEController, string layer, vector<vector<Ptext>>& convkernelData, Ptext &convbaisData, 
                        vector<vector<Ptext>>& shortcutkernelData, Ptext &shortcutbaisData,
                        int &dataWidth, int inputChannels, int outputChannels, int kernelWidth){
    
    string dataPath = "./../weights/resnet20/"+layer;
    int width_sq = pow(dataWidth, 2);
    int width_out_sq = pow((dataWidth/2), 2);
    
    /*** convolution data */
    auto rawKernel = load_weights(dataPath+"_conv1_weight.csv", outputChannels, inputChannels, kernelWidth, kernelWidth);
    auto biasVector = load_bias(dataPath+"_conv1_bias.csv");
    for(int i=0; i<outputChannels; i++){
        auto encodeKernel = fheonHEController.encode_kernel_optimized(rawKernel[i], width_sq);
        convkernelData.push_back(encodeKernel);
    }
 
    /*** shortcut data */
    auto shortcutbiasVector = load_bias(dataPath+"_shortcut_bias.csv");
    auto  shortcutrawKernel = load_fc_weights(dataPath+"_shortcut_weight.csv",  outputChannels, inputChannels);
    vector<Ptext> shData;
    for(int i=0; i < outputChannels; i++){
        auto encodeWeights = fheonHEController.encode_bais_input(shortcutrawKernel[i], width_sq);
        shData.push_back(encodeWeights);
    }
    shortcutkernelData.push_back(shData);
   
    /*** Bias vectors */
    convbaisData = fheonHEController.encode_bais_input(biasVector, width_out_sq);
	shortcutbaisData = fheonHEController.encode_bais_input(shortcutbiasVector, width_out_sq);
    dataWidth = dataWidth/2;

    rawKernel.clear();
    biasVector.clear();
    shortcutrawKernel.clear();
    shortcutbiasVector.clear();
}

void fclayer_data_processing(FHEONHEController &fheonHEController, string layer, vector<Ptext>& fc_kernelData, Ptext& fc_baisData,
                    int inputChannels, int outputChannels){

    string dataPath = "./../weights/resnet20/"+layer;
    auto fc_rawKernel = load_fc_weights(dataPath+"_weight.csv", outputChannels, inputChannels);
     auto fc_biasVector = load_bias(dataPath+"_bias.csv");

    for(int i=0; i < outputChannels; i++){
        auto encodeWeights = fheonHEController.encode_input(fc_rawKernel[i]);
        fc_kernelData.push_back(encodeWeights);
    }
	fc_baisData = context->MakeCKKSPackedPlaintext(fc_biasVector, 1);

    fc_rawKernel.clear();
    fc_biasVector.clear();
}

Ctext convolution_block(FHEONANNController &fheonANNController, Ctext& encrytedData, vector<vector<Ptext>>& kernelData, Ptext& baisData,
                            int &dataWidth, int inputChannels, int outputChannels, int striding){

    startIn = get_current_time();
    auto conv_data = fheonANNController.he_convolution_optimized(encrytedData, kernelData, baisData, dataWidth, inputChannels, outputChannels, striding);
    measuringTime.push_back(measureTime(startIn, get_current_time()));
    return conv_data;
}

Ctext resnet_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, Ctext& encrytedData, 
                vector<vector<vector<Ptext>>>& blockkernelData, vector<Ptext>& blockbaisData, 
                int &dataWidth, int &dataSize, int inputChannels, int outputChannels, int reluScale, int bootstrapState, bool shortcutConv){

    vector<Ctext> blockData(2);
    blockData[1] = encrytedData;
    
    if(shortcutConv){
        encrytedData = fheonHEController.bootstrap_function(encrytedData);
        
        startIn = get_current_time();
        blockData = fheonANNController.he_convolution_and_shortcut_optimized_with_multiple_channels(encrytedData,  blockkernelData[0], blockkernelData[2][0], 
                                                        blockbaisData[0], blockbaisData[2], dataWidth, inputChannels, outputChannels);
        measuringTime.push_back(measureTime(startIn, get_current_time()));

        dataWidth = dataWidth/2;
        dataSize = (outputChannels*pow(dataWidth, 2));
    }
    else{
        startIn = get_current_time();
        blockData[0] = fheonANNController.he_convolution_optimized(encrytedData,  blockkernelData[0], blockbaisData[0], dataWidth, inputChannels, outputChannels, striding);
        measuringTime.push_back(measureTime(startIn, get_current_time()));
    }
    if(bootstrapState){
        blockData[0] = fheonHEController.bootstrap_function(blockData[0]);
    }
    reluScale = fheonHEController.read_scaling_value(blockData[0], dataSize);

    startIn = get_current_time();
    blockData[0] = fheonANNController.he_relu(blockData[0], reluScale, dataSize);
    blockData[0] = fheonANNController.he_convolution_optimized(blockData[0],  blockkernelData[1], blockbaisData[1], dataWidth, outputChannels, outputChannels, striding);
    encrytedData = fheonANNController.he_sum_two_ciphertexts(blockData[0], blockData[1]);
    measuringTime.push_back(measureTime(startIn, get_current_time()));
    
    encrytedData = fheonHEController.bootstrap_function(encrytedData);
    reluScale = fheonHEController.read_scaling_value(encrytedData, dataSize);
    
    startIn = get_current_time();
    encrytedData = fheonANNController.he_relu(encrytedData, reluScale, dataSize);
    measuringTime.push_back(measureTime(startIn, get_current_time()));
    
    blockData.clear();
    return encrytedData;
}

Ctext FClayer_block(FHEONANNController &fheonANNController, Ctext& encrytedData, vector<Ptext>& fc_kernelData, Ptext& fc_baisData, int inputChannels, int outputChannels, int rotPosition){
    startIn = get_current_time();
    Ctext layer_data = fheonANNController.he_linear(encrytedData, fc_kernelData, fc_baisData, inputChannels, outputChannels, rotPosition);
    measuringTime.push_back(measureTime(startIn, get_current_time()));
    return layer_data;
}
