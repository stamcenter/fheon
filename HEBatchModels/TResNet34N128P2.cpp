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

#ifndef DEFAULT_ARG
#define DEFAULT_ARG 250
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
vector<Ctext> fc_layer_block(FHEONHEController &fheonHEController, FHEONANNBatchController fheonANNBatchController, string layer, Ctext encrytedInput, int batchSize, int inputChannels, int outputChannels, int rotPositions);

vector<int> measuringTime;
vector<int> intermTime;
auto startIn = get_current_time();
vector<int> slotsSizes = {15, 15, 15, 15, 15};
int scalingFact = 2; 

int main(int argc, char *argv[]) {

    auto begin_time = startTime();
    printWelcomeMessage();
    /*** Generate the context of the project in the FHEONHEController and pass it to the SnnController */
     int ringDegree = 16;
    int numSlots = 15;
    int circuitDepth = 11;
    int dcrtBits = 46;
    int firstMod = 50;
    int digitSize = 4;
    vector<uint32_t> levelBudget = {3, 3};
    int serialize = true;
    fheonHEController.generate_context(ringDegree, numSlots, circuitDepth, dcrtBits, firstMod, digitSize, levelBudget, serialize);
    context = fheonHEController.getContext();
    FHEONANNController fheonANNController(context);
    FHEONANNBatchController fheonANNBatchController( context);
    printDuration(begin_time, "Context Generation and Keys Serialization", false);
    cout << "---------------------------------RESNET34 PIPELINE 2 - 128-------------"<< to_string(DEFAULT_ARG) << "--------------------------" << endl; 
    
    /**** Read the CIFAR-10 Images and inference them */
    int img_cols = 32;
    int img_depth = 3;
    int kernelWidth = 3; 
    int paddingLen = 1;
    int stride = 1;
    int shortcutstride = 2;
    int avgpoolSize = 4;
    vector<int> channels = {3, 16, 32, 64, 128, 100};
    vector<int> dataWidths = {32, 16, 8, 4, 1};
    vector<int> dataSizes = {1024, 256, 64, 16};
    vector<int> batchSizes = {32, 128, 128};
    int pipeVal = 4;
    int rotPositions = 32;
    int fcRotIndex = 1000;
    
    //** generate rotation keys for conv_layer 1 */
    auto conv1_keys = fheonANNBatchController.generate_convolution_batch_rotation_positions(batchSizes[0], dataWidths[0],  kernelWidth, paddingLen, stride);
    auto conv2_keys = fheonANNBatchController.generate_convolution_batch_rotation_positions(batchSizes[0], dataWidths[0],  kernelWidth, paddingLen, shortcutstride);
    auto conv3_keys = fheonANNBatchController.generate_convolution_batch_rotation_positions(batchSizes[1], dataWidths[1],  kernelWidth, paddingLen, stride);
    auto conv4_keys = fheonANNBatchController.generate_convolution_batch_rotation_positions(batchSizes[1], dataWidths[1],  kernelWidth, paddingLen, shortcutstride);
    auto conv5_keys = fheonANNBatchController.generate_convolution_batch_rotation_positions(batchSizes[1], dataWidths[2],  kernelWidth, paddingLen, stride);
    auto conv6_keys = fheonANNBatchController.generate_convolution_batch_rotation_positions(batchSizes[1], dataWidths[2],  kernelWidth, paddingLen, shortcutstride);
    auto conv7_keys = fheonANNBatchController.generate_convolution_batch_rotation_positions(batchSizes[1], dataWidths[3],  kernelWidth, paddingLen, stride);
    auto conv8_keys = fheonANNBatchController.generate_convolution_batch_rotation_positions(batchSizes[1], dataWidths[3],  kernelWidth, paddingLen, shortcutstride);

    auto avg_keys = fheonANNBatchController.generate_avgpool_batch_optimized_rotation_positions(batchSizes[1], dataWidths[3], avgpoolSize, avgpoolSize, true, rotPositions);
    auto convert_keys = fheonANNBatchController.generate_batch_inputs_converter_rotation_positions(batchSizes[1],  channels[4], dataWidths[4]);
    auto fc_keys = fheonANNBatchController.generate_linear_batch_rotation_positions(batchSizes[1], {channels[5]}, {channels[4]}, fcRotIndex);
    /************************************************************************************************ */

    vector<vector<int>> rkeys_layer1, rkeys_layer2, rkeys_layer3, rkeys_layer4, convert_layer, fc_layer;
    rkeys_layer1.push_back(conv1_keys);
    rkeys_layer1.push_back(conv2_keys);
    rkeys_layer1.push_back(conv3_keys);
    
    rkeys_layer2.push_back(conv3_keys);
    rkeys_layer2.push_back(conv4_keys);
    rkeys_layer2.push_back(conv5_keys);

    rkeys_layer3.push_back(conv5_keys);
    rkeys_layer3.push_back(conv6_keys);
    rkeys_layer3.push_back(conv7_keys);
    rkeys_layer3.push_back(conv8_keys);

    // rkeys_layer3.push_back(conv5_keys);
    rkeys_layer4.push_back(avg_keys);
    convert_layer.push_back(convert_keys);
    fc_layer.push_back(fc_keys);
/********************************************************************************************************************************************/;
    /*** join all keys and generate unique values only */
    vector<int> serkeys_layer1 = serialize_rotation_keys(rkeys_layer1); 
    vector<int> serkeys_layer2 = serialize_rotation_keys(rkeys_layer2);
    vector<int> serkeys_layer3 = serialize_rotation_keys(rkeys_layer3);
    vector<int> serkeys_layer4 = serialize_rotation_keys(rkeys_layer4);
    vector<int> serkeys_convert_layer = serialize_rotation_keys(convert_layer);
    vector<int> serkeys_fc_layer = serialize_rotation_keys(fc_layer);
    // /*********************************************** Key Generation ******************************************************************************/
    auto begin_rotkeygen_time = startTime();
    // cout << "This is the rotation positions (" << serkeys_block1.size() <<"+" << serkeys_block2.size() << "+" << serkeys_block3.size() << " = " << total_rkeys << "): " << endl;
    cout << "Layer 1 keys (" << serkeys_layer1.size() << ") " << serkeys_layer1 << endl;
    cout << "Layer 2 keys (" << serkeys_layer2.size() << ") " << serkeys_layer2 << endl;
    cout << "Layer 3 keys (" << serkeys_layer3.size() << ") " << serkeys_layer3 << endl;
    cout << "Layer 4 keys (" << serkeys_layer4.size() << ") " << serkeys_layer4 << endl;
    cout << "Converter keys (" << serkeys_convert_layer.size() << ") " << serkeys_convert_layer << endl;
    cout << "FC Layer keys (" << serkeys_fc_layer.size() << ") " << serkeys_fc_layer << endl;

    fheonHEController.generate_bootstrapping_and_rotation_keys(serkeys_layer1, slotsSizes[0], "layer1.bin", true);
    fheonHEController.clear_context(slotsSizes[0]);
    
    fheonHEController.generate_bootstrapping_and_rotation_keys(serkeys_layer2, slotsSizes[1], "layer2.bin",  true);
    fheonHEController.clear_context(slotsSizes[1]);
    
    fheonHEController.generate_bootstrapping_and_rotation_keys(serkeys_layer3, slotsSizes[2], "layer3.bin", true);
    fheonHEController.clear_context(slotsSizes[2]);

    fheonHEController.generate_bootstrapping_and_rotation_keys(serkeys_layer4, slotsSizes[2], "layer4.bin", true);
    fheonHEController.clear_context(slotsSizes[3]);

    fheonHEController.generate_bootstrapping_and_rotation_keys(serkeys_convert_layer, slotsSizes[2], "convert_layer.bin", true);
    fheonHEController.clear_context(slotsSizes[4]);

    fheonHEController.generate_bootstrapping_and_rotation_keys(serkeys_fc_layer, slotsSizes[3], "fc_layer.bin", true);
    fheonHEController.clear_context(slotsSizes[4]);
    printDuration(begin_rotkeygen_time, "Rotation KeyGen Time", false);
   
    int numImages = DEFAULT_ARG+INDEX_VALUE;
    int dataSize = img_depth*pow(img_cols, 2);
    string cifar100tPath = "./../images/cifar-100-binary/test.bin";
    vector<vector<double>> imagesData = read_images(cifar100tPath, numImages, dataSize);
    ofstream outFile;
    outFile.open("./../results/TresNet34/fhepredictions.txt", ios_base::app);
    Ptext decryptedData;
    vector<Ctext> convData;
    vector<vector<Ctext>> tconvData(pipeVal);
    int totalSize = dataSize;
    int imgIdx = 0;
    int polyDegee = 59;
    int reluScale = 10;
    int bootstrap_level = 2;
    
    for (int idx = 0; idx < 1; idx++) {
        
        auto inference_time = startTime();
        fheonHEController.clear_context(slotsSizes[4]);
        fheonHEController.load_bootstrapping_and_rotation_keys(slotsSizes[0], "layer1.bin", false);

        for(int tIdx=0; tIdx < pipeVal; tIdx++){
            int imgIdx = ( (idx*pipeVal) + tIdx ) * batchSizes[0]; 
            vector<vector<double>> batchedImages; 
            cout << endl << endl << endl;
            for (int b = 0; b < batchSizes[0]; b++) {
                int fImgIndx = imgIdx + b; 
                cout << "Loading Image: " << fImgIndx << endl; 
                auto img = imagesData[fImgIndx];
                batchedImages.push_back(img);
            }

            /** adjust images */
            auto inputDatas = convert_inputData(batchedImages, batchSizes[0], channels[0], dataSizes[0]);
            vector<Ctext>encryptedInputs;
            for(int i=0; i<channels[0]; i++){
                encryptedInputs.push_back(fheonHEController.encrypt_input(inputDatas[i]));
            }
            cout << endl << imgIdx+1  << " to " << imgIdx+batchSizes[0] << " - (" << encryptedInputs.size() << " input channes) images Read, Normalized and Encrypt"<< endl;
            cout << endl << endl;

            /************************************************************************************************ */
            cout<< "Layer 0" << endl;
            convData = convolution_block(fheonHEController, fheonANNBatchController,  "layer0_conv1", encryptedInputs, batchSizes[0], dataWidths[0], dataSizes[0], channels[0], channels[1], stride);
            totalSize = batchSizes[0] * dataSizes[0];
            auto scalingVals = fheonHEController.read_batch_scaling_values(convData, channels[1], totalSize);
        
            startIn = get_current_time();
            convData = fheonANNBatchController.he_batch_relu(convData, scalingVals, channels[1], totalSize, polyDegee);
            measuringTime.push_back(measureTime(startIn, get_current_time()));
            printDuration(inference_time, "run time", false);

            cout<< endl<<  "Layer 1" << endl;
            cout <<"Block 1 " << endl;
            convData = resnet_block(fheonHEController, fheonANNBatchController, "layer1_block1", convData, batchSizes[0], dataWidths[0], dataSizes[0], channels[1], channels[1], reluScale, false, false);
            cout <<"Block 2 " << endl;
            convData = resnet_block(fheonHEController, fheonANNBatchController, "layer1_block2", convData, batchSizes[0], dataWidths[0], dataSizes[0], channels[1], channels[1], reluScale, true, false);
            cout <<"Block 3 " << endl;
            convData = resnet_block(fheonHEController, fheonANNBatchController, "layer1_block3", convData, batchSizes[0], dataWidths[0], dataSizes[0], channels[1], channels[1], reluScale, true, false);
            printDuration(inference_time, "run time", false);

            cout<< endl<< "Layer 2" << endl;
            cout <<"Block 1 " << endl;
            convData = resnet_block(fheonHEController, fheonANNBatchController, "layer2_block1", convData, batchSizes[0], dataWidths[0], dataSizes[0], channels[1], channels[2], reluScale, true, true);
            tconvData[tIdx] = convData;
            convData.clear();
            encryptedInputs.clear();
            batchedImages.clear();
        }
        
        /*** create a joined new cipher */
        totalSize = batchSizes[0] * dataSizes[1];
        cout << endl << "Batch Size: "<< batchSizes[0] << " -- TotalSize: " << totalSize << endl; 
        vector<double> cleaningInVec = generate_mixed_mask(totalSize, (batchSizes[0]*dataSizes[0]));
        Ptext cleaningMask = context->MakeCKKSPackedPlaintext(cleaningInVec, 1, 1);
        vector<Ctext> interConvData(channels[2]);
        for(int chan=0; chan<channels[2]; chan++){
            vector<Ctext> pipData(pipeVal);
            pipData[0] = context->EvalMult(tconvData[0][chan], cleaningMask);
            for(int pIdx=1; pIdx < pipeVal; pIdx++){
                pipData[pIdx] = context->EvalRotate(context->EvalMult(tconvData[pIdx][chan], cleaningMask), -(pIdx*totalSize));
            }
            interConvData[chan] = context->EvalAddMany(pipData); 
            pipData.clear();
        }
        totalSize = batchSizes[1] * dataSizes[1];
        cout << "New Batch Size: "<< batchSizes[1] << " -- TotalSize: " << totalSize  <<  endl << endl;

        fheonHEController.clear_context(slotsSizes[0]);
        fheonHEController.load_bootstrapping_and_rotation_keys(slotsSizes[1], "layer2.bin", false);

        cout <<"Block 2 " << endl;
        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer2_block2", interConvData, batchSizes[1], dataWidths[1], dataSizes[1], channels[2], channels[2], reluScale, true, false);
        
        interConvData.clear();
        interConvData.shrink_to_fit();

        cout <<"Block 3 " << endl;
        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer2_block3", convData, batchSizes[1], dataWidths[1], dataSizes[1], channels[2], channels[2], reluScale, true, false);
        cout <<"Block 4 " << endl;
        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer2_block4", convData, batchSizes[1], dataWidths[1], dataSizes[1], channels[2], channels[2], reluScale, true, false);
        // fheonHEController.read_batch_minmax(convData, channels[2], totalSize);
        printDuration(inference_time, "run time", false);

        cout<< endl<<  "Layer 3" << endl;
        cout <<"Block 1 " << endl;
        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer3_block1", convData, batchSizes[1], dataWidths[1], dataSizes[1], channels[2], channels[3], reluScale, true, true);
        totalSize = batchSizes[1] * dataSizes[2];

        fheonHEController.clear_context(slotsSizes[0]);
        fheonHEController.load_bootstrapping_and_rotation_keys(slotsSizes[1], "layer3.bin", false);

        cout <<"Block 2 " << endl;
        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer3_block2", convData, batchSizes[1], dataWidths[2], dataSizes[2], channels[3], channels[3], reluScale, true, false);
        cout <<"Block 3" << endl;
        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer3_block3", convData, batchSizes[1], dataWidths[2], dataSizes[2], channels[3], channels[3], reluScale, true, false);
        cout <<"Block 4" << endl;
        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer3_block4", convData, batchSizes[1], dataWidths[2], dataSizes[2], channels[3], channels[3], reluScale, true, false);
        cout <<"Block 5" << endl;
        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer3_block5", convData, batchSizes[1], dataWidths[2], dataSizes[2], channels[3], channels[3], reluScale, true, false);
        cout <<"Block 6" << endl;
        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer3_block6", convData, batchSizes[1], dataWidths[2], dataSizes[2], channels[3], channels[3], reluScale, true, false);
        // fheonHEController.read_batch_minmax(convData, channels[3], totalSize);
        printDuration(inference_time, "run time", false);
        
        cout<< endl<<  "Layer 4" << endl;
        cout <<"Block 1 " << endl;
        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer4_block1", convData, batchSizes[1], dataWidths[2], dataSizes[2], channels[3], channels[4], reluScale, true, true);
        totalSize = batchSizes[1] * dataSizes[3];

        cout <<"Block 2 " << endl;
        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer4_block2", convData, batchSizes[1], dataWidths[3], dataSizes[3], channels[4], channels[4], reluScale, true, false);
        cout <<"Block 3" << endl;
        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer4_block3", convData, batchSizes[1], dataWidths[3], dataSizes[3], channels[4], channels[4], reluScale, true, false);
        // fheonHEController.read_batch_minmax(convData, channels[4], totalSize);
        printDuration(inference_time, "run time", false);

    
        // cout<< "Classification" << endl;
        printTimeWithMessage("ResNet34 Circuit: ", measuringTime);
        startIn = get_current_time();
        convData = fheonHEController.batch_bootstrap_function(convData, channels[4], bootstrap_level);
        intermTime.push_back(measureTime(startIn, get_current_time()));


        fheonHEController.clear_context(slotsSizes[2]);
        fheonHEController.load_bootstrapping_and_rotation_keys(slotsSizes[4], "layer4.bin", false);
        startIn = get_current_time();


        cout << " Global Pooling" << endl;
        convData = fheonANNBatchController.he_batch_globalpool(convData, batchSizes[1], dataWidths[3], channels[4], avgpoolSize, rotPositions);
        measuringTime.push_back(measureTime(startIn, get_current_time()));

        cout << "Ciphertext Converter..." << endl;
        fheonHEController.clear_context(slotsSizes[2]);
        fheonHEController.load_bootstrapping_and_rotation_keys(slotsSizes[4], "convert_layer.bin", false);
        auto fcData = fheonANNBatchController.he_batch_inputs_converter(convData, batchSizes[1], channels[4], dataWidths[4]);
        cout << "Converter done" << endl; 

        convData.clear();
        convData.shrink_to_fit();


        cout << "Fully Connected" << endl; 
        fheonHEController.clear_context(slotsSizes[2]);
        fheonHEController.load_bootstrapping_and_rotation_keys(slotsSizes[4], "fc_layer.bin", false);
        auto multi_fcData = fc_layer_block(fheonHEController, fheonANNBatchController, "layer_fc", fcData, batchSizes[1], channels[4], channels[5], fcRotIndex);

        printTimeWithMessage("ResNet34 Circuit: ", measuringTime);
        measuringTime.clear();
        printTimeWithMessage("ResNet34 Bootsrapping: ", intermTime);
        intermTime.clear();

        string infereMessage = "Batch Size ("+ to_string(batchSizes[1]) +") -- Total Run Time for Images " + to_string(imgIdx + 1) + " - " +  to_string(imgIdx+1+batchSizes[1]);  
        printDuration(inference_time, infereMessage, false);
        auto predictions = fheonHEController.read_batch_inferenced_label_multiple_outputs(multi_fcData, batchSizes[1], channels[5], outFile);

        cout << "Batch Predictions: " << predictions << endl;
    }
    outFile.close();
    cout << "All predicted results printed to File." << endl;
    clear_images(imagesData, numImages);
   return 0;
}

vector<Ctext> shortcut_convolution_block(FHEONHEController &fheonHEController, FHEONANNBatchController fheonANNBatchController, string layer, vector<Ctext>& encrytedInputs, int batchSize, 
                            int &dataWidth, int &dataSize, int inputChannels, int outputChannels, int stride){
    
    int outputWidth = dataWidth/stride;
    int outputSize = pow(outputWidth, 2);
    
    /******** DOUBLE CHECK THIS KERNEL ENCODING */
    string dataPath = "./../weights/resnet34/"+layer;
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
    string dataPath = "./../weights/resnet34/"+layer;
    auto rawKernelData = load_batch_weights(dataPath+"_weight.csv", outputChannels, inputChannels, batchSize, kernelWidth, kernelWidth);
    auto rawBiasData = load_batch_bias(dataPath+"_bias.csv", outputChannels, batchSize, outputSize);

    startIn = get_current_time();
    auto conv_data = fheonANNBatchController.he_batch_convolution_optimized(fheonHEController, encrytedInputs, rawKernelData, rawBiasData, batchSize, dataWidth, inputChannels, outputChannels, stride);
    measuringTime.push_back(measureTime(startIn, get_current_time()));
        
    rawKernelData.clear();
    rawKernelData.shrink_to_fit();
    rawBiasData.clear();
    rawBiasData.shrink_to_fit();
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
        startIn = get_current_time();
        encrytedInputs = fheonHEController.batch_bootstrap_function(encrytedInputs, inputChannels, bootstrap_level);
        intermTime.push_back(measureTime(startIn, get_current_time()));

        convData = convolution_block(fheonHEController, fheonANNBatchController, layer+"_conv1", encrytedInputs, batchSize, dataWidth, dataSize, inputChannels, outputChannels, shortcutStridindLen);
        shortcuts = shortcut_convolution_block(fheonHEController, fheonANNBatchController, layer, encrytedInputs, batchSize, dataWidth, dataSize, inputChannels, outputChannels, shortcutStridindLen);
        
        dataWidth = dataWidth/2;
        dataSize = pow(dataWidth, 2);
        auto short_scalingVals = fheonHEController.read_batch_scaling_values(shortcuts, outputChannels, (batchSize*dataSize));
        cout << "Shortcut SumScaling Values----: " << short_scalingVals << endl;
    }
    else{
        convData = convolution_block(fheonHEController, fheonANNBatchController, layer+"_conv1", encrytedInputs, batchSize, dataWidth, dataSize, inputChannels, outputChannels, stride);
    }
    if(bootstrapState){
        startIn = get_current_time();
        convData = fheonHEController.batch_bootstrap_function(convData, outputChannels, bootstrap_level);
        intermTime.push_back(measureTime(startIn, get_current_time()));
    }

    int totalSize = (batchSize*dataSize); 
    auto scalingVals = fheonHEController.read_batch_scaling_values(convData, outputChannels, totalSize);
    cout << "1st Convolution Scaling Values: " << scalingVals << endl;

    if(layer == "layer4_block2"){
        for(int i=0; i<outputChannels; i++){
            scalingVals[i] = scalingFact*scalingVals[i]; 
        }
        cout << "SumScaling Values * ScaleFact -------: " << scalingVals << endl;
    }

    startIn = get_current_time();
    convData = fheonANNBatchController.he_batch_relu(convData, scalingVals, outputChannels, totalSize, polyDegee);
    measuringTime.push_back(measureTime(startIn, get_current_time()));
    auto sum_convData = convolution_block(fheonHEController, fheonANNBatchController, layer+"_conv2", convData, batchSize, dataWidth, dataSize, outputChannels, outputChannels, stride);
    scalingVals = fheonHEController.read_batch_scaling_values(sum_convData, outputChannels, totalSize);
    cout << "2nd Convolution Scaling Values: " << scalingVals << endl;

    convData.clear();
    sum_convData = fheonANNBatchController.he_batch_sum_ciphertexts(sum_convData, shortcuts, outputChannels);
    encrytedInputs.clear();
    shortcuts.clear();

    startIn = get_current_time();
    sum_convData = fheonHEController.batch_bootstrap_function(sum_convData, outputChannels, bootstrap_level);
    intermTime.push_back(measureTime(startIn, get_current_time()));

    scalingVals = fheonHEController.read_batch_scaling_values(sum_convData, outputChannels, totalSize);
    cout << "After SumScaling Values-------: " << scalingVals << endl;
    
    startIn = get_current_time();
    sum_convData = fheonANNBatchController.he_batch_relu(sum_convData, scalingVals, outputChannels, totalSize, polyDegee);
    measuringTime.push_back(measureTime(startIn, get_current_time()));
    return sum_convData;
}

vector<Ctext> fc_layer_block(FHEONHEController &fheonHEController, FHEONANNBatchController fheonANNBatchController, string layer, Ctext encrytedInput, int batchSize, int inputChannels, int outputChannels, int rotPositions){
   
    string dataPath = "./../weights/resnet34/"+layer;
    auto fc_rawKernelData = load_batch_fc_weights(dataPath+"_weight.csv", outputChannels, batchSize, inputChannels);
    auto fc_rawBiasData = load_bias(dataPath+"_bias.csv");
    int encodeLevel = encrytedInput->GetLevel();
    vector<Ptext> fc_kernelData;
    for(int i=0; i < outputChannels; i++){
        auto encodeWeights = fheonHEController.encode_input(fc_rawKernelData[i], encodeLevel);
        fc_kernelData.push_back(encodeWeights);
    }
    Ptext fc_BiasData = fheonHEController.encode_input(fc_rawBiasData, encodeLevel);

    startIn = get_current_time();
    auto fcData = fheonANNBatchController.he_batch_linear_multiple_outputs(encrytedInput, fc_kernelData, fc_BiasData, batchSize, inputChannels, outputChannels);
    measuringTime.push_back(measureTime(startIn, get_current_time()));
    fc_kernelData.clear();
    fc_kernelData.shrink_to_fit();
    return fcData;
}