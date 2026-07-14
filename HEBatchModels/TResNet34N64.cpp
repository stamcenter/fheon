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

#ifndef DEFAULT_TEST_SIZE
#define DEFAULT_TEST_SIZE 250
#endif

#ifndef INDEX_VALUE
#define INDEX_VALUE 0
#endif

using namespace std;
CryptoContext<DCRTPoly> context;
FHEONHEController fheonHEController(context);
vector<Ctext> shortcut_convolution_block(FHEONHEController &fheonHEController, FHEONANNBatchController fheonANNBatchController, string layer, vector<Ctext>& encrytedInputs, int batchSize, 
                            int &dataWidth, int &dataSize, int inputChannels, int outputChannels, int stride);
vector<Ctext> convolution_block(FHEONHEController &fheonHEController, FHEONANNBatchController fheonANNBatchController, string layer, vector<Ctext>& encrytedInputs, int batchSize, 
                            int &inputdataWidth, int &inputdataSize, int inputChannels, int outputChannels, int stride);
vector<Ctext> resnet_block(FHEONHEController &fheonHEController, FHEONANNBatchController fheonANNBatchController, string layer, vector<Ctext>& encrytedInputs, int batchSize, int &dataWidth, int &dataSize,
                 int inputChannels, int outputChannels, int reluScale, bool bootstrapState, bool shortcutState);
Ctext fc_layer_block(FHEONHEController &fheonHEController, FHEONANNBatchController fheonANNBatchController, string layer, Ctext encrytedInput, int batchSize, int inputChannels, int outputChannels, int rotPositions);

vector<int> slotsSizes = {14, 14, 14, 14, 14};
int scalingFact = 2; 

int main(int argc, char *argv[]) {
    weights_folder = "./../weights/resnet34/";
    predictions_file = "./../results/TresNet34/fhepredictions.txt";
    auto runtime_config = parse_runtime_args(argc, argv, DEFAULT_TEST_SIZE, INDEX_VALUE);
    int test_size = runtime_config.test_size;
    int index_value = runtime_config.index_value;

    auto begin_time = startTime();
    printWelcomeMessage();
    /*** Generate the context of the project in the FHEONHEController and pass it to the SnnController */
    bool loadContext = runtime_config.load_context;
    FHEONHEController::HEConfig heinitConfig;
    heinitConfig.ringDim = 15;
    heinitConfig.numSlots = 14;
    heinitConfig.mlevelBootstrap = 11;
    heinitConfig.dcrtBits = 46;
    heinitConfig.firstMod = 50;
    heinitConfig.numDigits = 4;
    heinitConfig.levelBudget = {3, 3};
    heinitConfig.serialize = true;
    if (!runtime_config.keys_folder.empty()) {
        heinitConfig.keysFolder = runtime_config.keys_folder;
    }
    fheonHEController.load_context(heinitConfig, loadContext);
    context = fheonHEController.getContext();
    FHEONANNController fheonANNController(context);
    FHEONANNBatchController fheonANNBatchController( context);
    printDuration(begin_time, loadContext ? "Context Loaded" : "Context Generated and Keys Serialization", false);
    cout << "---------------------------------RESNET34-------------"<< to_string(test_size) << "--------------------------" << endl; 
    
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
    vector<int> batchSizes = {16, 64, 256};
    int pipeVal = 4;
    int rotPositions = 16;
    int fcRotIndex = 800;
    int baseIndex = 83;
    
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
    auto convert_keys = fheonANNBatchController.generate_batch_inputs_converter_rotation_positions(batchSizes[1],  channels[4], dataWidths[4], baseIndex);
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
    vector<int> serconverter_layer = serialize_rotation_keys(convert_layer);
    vector<int> serkeys_fc_layer = serialize_rotation_keys(fc_layer);
    // /*********************************************** Key Generation ******************************************************************************/
    auto begin_rotkeygen_time = startTime();
    // cout << "This is the rotation positions (" << serkeys_block1.size() <<"+" << serkeys_block2.size() << "+" << serkeys_block3.size() << " = " << total_rkeys << "): " << endl;
    cout << "Layer 1 keys (" << serkeys_layer1.size() << ") " << serkeys_layer1 << endl;
    cout << "Layer 2 keys (" << serkeys_layer2.size() << ") " << serkeys_layer2 << endl;
    cout << "Layer 3 keys (" << serkeys_layer3.size() << ") " << serkeys_layer3 << endl;
    cout << "Layer 4 keys (" << serkeys_layer4.size() << ") " << serkeys_layer4 << endl;
    cout << "Converter keys (" << serconverter_layer.size() << ") " << serconverter_layer << endl;
    cout << "FC Layer keys (" << serkeys_fc_layer.size() << ") " << serkeys_fc_layer << endl;

    fheonHEController.generate_bootstrapping_and_rotation_keys(serkeys_layer1, slotsSizes[0], "layer1.bin", true);
    fheonHEController.clear_context(slotsSizes[0]);
    
    fheonHEController.generate_bootstrapping_and_rotation_keys(serkeys_layer2, slotsSizes[1], "layer2.bin",  true);
    fheonHEController.clear_context(slotsSizes[1]);
    
    fheonHEController.generate_bootstrapping_and_rotation_keys(serkeys_layer3, slotsSizes[2], "layer3.bin", true);
    fheonHEController.clear_context(slotsSizes[2]);

    fheonHEController.generate_bootstrapping_and_rotation_keys(serkeys_layer4, slotsSizes[2], "layer4.bin", true);
    fheonHEController.clear_context(slotsSizes[3]);

    fheonHEController.generate_bootstrapping_and_rotation_keys(serconverter_layer, slotsSizes[2], "convert_layer.bin", true);
    fheonHEController.clear_context(slotsSizes[3]);

    fheonHEController.generate_bootstrapping_and_rotation_keys(serkeys_fc_layer, slotsSizes[3], "fc_layer.bin", true, true);
    fheonHEController.clear_context(slotsSizes[4]);
    printDuration(begin_rotkeygen_time, "Rotation KeyGen Time", false);
   
    int numImages = test_size+index_value;
    int dataSize = img_depth*pow(img_cols, 2);
    
    vector<vector<double>> imagesData = read_images(cifar100tPath, numImages, dataSize);
    
    
    Ptext decryptedData;
    vector<Ctext> convData;
    int totalSize = batchSizes[0] * dataSizes[1];
    vector<double> cleaningInVec = generate_mixed_mask(totalSize, (batchSizes[0]*dataSizes[0]));
    Ptext cleaningMask = context->MakeCKKSPackedPlaintext(cleaningInVec, 1, 1);
    vector<Ctext> interConvData(channels[2]);
    int imgIdx = 0;
    int polyDegee = 59;
    int reluScale = 10;
    int bootstrap_level = 2;
    
    for (int idx = 0; idx < 1; idx++) {
        
        auto inference_time = startTime();
        fheonHEController.clear_context(slotsSizes[4]);
        context = fheonHEController.read_evaluation_keys(context, "layer1.bin");
        bool isTime = true;

        for(int tIdx=0; tIdx < pipeVal; tIdx++){
            int imgIdx = ( (idx*pipeVal) + tIdx ) * batchSizes[0]; 
            vector<vector<double>> batchedImages; 

            for (int b = 0; b < batchSizes[0]; b++) {
                int fImgIndx = imgIdx + b; 
                cout << "Loading Image: " << fImgIndx << endl; 
                auto img = imagesData[fImgIndx];
                batchedImages.push_back(img);
            }
            if(tIdx != 0){
                isTime = false;
            }

            /** adjust images */
            auto inputDatas = convert_inputData(batchedImages, batchSizes[0], channels[0], dataSizes[0]);
            vector<Ctext>encryptedInputs;
            for(int i=0; i<channels[0]; i++){
                encryptedInputs.push_back(fheonHEController.encrypt_input(inputDatas[i], isTime));
            }
            cout << endl << imgIdx+1  << " to " << imgIdx+batchSizes[0] << " - (" << encryptedInputs.size() << " input channes) images Read, Normalized and Encrypt"<< endl;

            /************************************************************************************************ */
            convData = convolution_block(fheonHEController, fheonANNBatchController,  "layer0_conv1", encryptedInputs, batchSizes[0], dataWidths[0], dataSizes[0], channels[0], channels[1], stride);
            int reluTotalSize = batchSizes[0] * dataSizes[0];
            auto scalingVals = fheonHEController.read_batch_scaling_values(convData, channels[1], reluTotalSize);
        
            
            convData = fheonANNBatchController.he_batch_relu(convData, scalingVals, channels[1], reluTotalSize, polyDegee);
            

            convData = resnet_block(fheonHEController, fheonANNBatchController, "layer1_block1", convData, batchSizes[0], dataWidths[0], dataSizes[0], channels[1], channels[1], reluScale, false, false);
            convData = resnet_block(fheonHEController, fheonANNBatchController, "layer1_block2", convData, batchSizes[0], dataWidths[0], dataSizes[0], channels[1], channels[1], reluScale, true, false);
            convData = resnet_block(fheonHEController, fheonANNBatchController, "layer1_block3", convData, batchSizes[0], dataWidths[0], dataSizes[0], channels[1], channels[1], reluScale, true, false);

            convData = resnet_block(fheonHEController, fheonANNBatchController, "layer2_block1", convData, batchSizes[0], dataWidths[0], dataSizes[0], channels[1], channels[2], reluScale, true, true);
            
            for(int chan=0; chan<channels[2]; chan++){
                if(tIdx == 0){
                    interConvData[chan] = context->EvalMult(convData[chan], cleaningMask);
                }
                else {
                    convData[chan] =  context->EvalRotate(context->EvalMult(convData[chan], cleaningMask), -(tIdx*totalSize));
                    interConvData[chan] = context->EvalAdd(interConvData[chan], convData[chan]);
                }
            }
            convData.clear();
            encryptedInputs.clear();
            batchedImages.clear();
        }
        
        /*** create a joined new cipher */
        cout << endl << "Batch Size: "<< batchSizes[0] << " -- TotalSize: " << totalSize << endl; 
        totalSize = batchSizes[1] * dataSizes[1];
        cout << "New Batch Size: "<< batchSizes[1] << " -- TotalSize: " << totalSize  <<  endl << endl;

        fheonHEController.clear_context(slotsSizes[0]);
        context = fheonHEController.read_evaluation_keys(context, "layer2.bin");

        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer2_block2", interConvData, batchSizes[1], dataWidths[1], dataSizes[1], channels[2], channels[2], reluScale, true, false);
        
        interConvData.clear();
        interConvData.shrink_to_fit();

        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer2_block3", convData, batchSizes[1], dataWidths[1], dataSizes[1], channels[2], channels[2], reluScale, true, false);
        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer2_block4", convData, batchSizes[1], dataWidths[1], dataSizes[1], channels[2], channels[2], reluScale, true, false);

        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer3_block1", convData, batchSizes[1], dataWidths[1], dataSizes[1], channels[2], channels[3], reluScale, true, true);
        totalSize = batchSizes[1] * dataSizes[2];

        fheonHEController.clear_context(slotsSizes[0]);
        context = fheonHEController.read_evaluation_keys(context, "layer3.bin");

        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer3_block2", convData, batchSizes[1], dataWidths[2], dataSizes[2], channels[3], channels[3], reluScale, true, false);

        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer3_block3", convData, batchSizes[1], dataWidths[2], dataSizes[2], channels[3], channels[3], reluScale, true, false);

        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer3_block4", convData, batchSizes[1], dataWidths[2], dataSizes[2], channels[3], channels[3], reluScale, true, false);

        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer3_block5", convData, batchSizes[1], dataWidths[2], dataSizes[2], channels[3], channels[3], reluScale, true, false);

        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer3_block6", convData, batchSizes[1], dataWidths[2], dataSizes[2], channels[3], channels[3], reluScale, true, false);
        
        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer4_block1", convData, batchSizes[1], dataWidths[2], dataSizes[2], channels[3], channels[4], reluScale, true, true);
        totalSize = batchSizes[1] * dataSizes[3];

        // fheonHEController.clear_context(slotsSizes[2]);
        // context = fheonHEController.read_evaluation_keys(context, "layer4.bin");
        

        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer4_block2", convData, batchSizes[1], dataWidths[3], dataSizes[3], channels[4], channels[4], reluScale, true, false);

        convData = resnet_block(fheonHEController, fheonANNBatchController, "layer4_block3", convData, batchSizes[1], dataWidths[3], dataSizes[3], channels[4], channels[4], reluScale, true, false);

    
                
        convData = fheonHEController.batch_bootstrap_function(convData, channels[4], bootstrap_level);
        
        fheonHEController.clear_context(slotsSizes[2]);
        context = fheonHEController.read_evaluation_keys(context, "layer4.bin");
        

        convData = fheonANNBatchController.he_batch_globalpool(convData, batchSizes[1], dataWidths[3], channels[4], avgpoolSize, rotPositions);
        

        cout << "Ciphertext Converter..." << endl;
        fheonHEController.clear_context(slotsSizes[2]);
        context = fheonHEController.read_evaluation_keys(context, "convert_layer.bin");
        auto fcData = fheonANNBatchController.he_batch_inputs_converter(convData, batchSizes[1], channels[4], dataWidths[4], baseIndex);
        cout << "Converter done" << endl; 

        fheonHEController.clear_context(slotsSizes[2]);
        context = fheonHEController.read_evaluation_keys(context, "fc_layer.bin");

        convData.clear();
        convData.shrink_to_fit();

        fcData = fc_layer_block(fheonHEController, fheonANNBatchController, "layer_fc", fcData, batchSizes[1], channels[4], channels[5], fcRotIndex);

                
                
        string infereMessage = "Batch Size ("+ to_string(batchSizes[1]) +") -- Total Run Time for Images " + to_string(imgIdx + 1) + " - " +  to_string(imgIdx+1+batchSizes[1]);  
        printDuration(inference_time, infereMessage, false);
        auto predictions = fheonHEController.read_batch_inferenced_label(fcData, batchSizes[1], channels[5], predictions_file);

        cout << "Batch Predictions: " << predictions << endl;
    }
    
    cout << "All predicted results printed to File." << endl;
    clear_images(imagesData, numImages);
   return 0;
}

vector<Ctext> shortcut_convolution_block(FHEONHEController &fheonHEController, FHEONANNBatchController fheonANNBatchController, string layer, vector<Ctext>& encrytedInputs, int batchSize, 
                            int &dataWidth, int &dataSize, int inputChannels, int outputChannels, int stride){
    
    int outputWidth = dataWidth/stride;
    int outputSize = pow(outputWidth, 2);
    
    /******** DOUBLE CHECK THIS KERNEL ENCODING */
    string dataPath = weights_folder+layer;
    auto rawKernelData = load_shortcut_batch_weights(dataPath+"_shortcut_weight.csv", batchSize, outputChannels, inputChannels, dataSize); 
    auto rawBiasData = load_batch_bias(dataPath+"_shortcut_bias.csv", outputChannels, batchSize, outputSize);

    
    auto conv_data = fheonANNBatchController.he_batch_convolution_shortcut_optimized(fheonHEController, encrytedInputs, rawKernelData, rawBiasData, batchSize, dataWidth, inputChannels, outputChannels, stride);
    
    
    rawKernelData.clear();
    rawKernelData.shrink_to_fit();
    rawBiasData.clear();
    rawBiasData.shrink_to_fit();
    return conv_data;
}

vector<Ctext> convolution_block(FHEONHEController &fheonHEController, FHEONANNBatchController fheonANNBatchController, string layer, vector<Ctext>& encrytedInputs, int batchSize, 
                            int &dataWidth, int &dataSize, int inputChannels, int outputChannels, int stride){
    if (layer.find("block") == std::string::npos && layer.find("_conv") == std::string::npos) { printModelLayer(layer); }
     
    int kernelWidth = 3;
    int outputWidth = dataWidth/stride;
    int outputSize = pow(outputWidth, 2);
    string dataPath = weights_folder+layer;
    auto rawKernelData = load_batch_weights(dataPath+"_weight.csv", outputChannels, inputChannels, batchSize, kernelWidth, kernelWidth);
    auto rawBiasData = load_batch_bias(dataPath+"_bias.csv", outputChannels, batchSize, outputSize);

    
    auto conv_data = fheonANNBatchController.he_batch_convolution_optimized(fheonHEController, encrytedInputs, rawKernelData, rawBiasData, batchSize, dataWidth, inputChannels, outputChannels, stride);
    
        
    rawKernelData.clear();
    rawKernelData.shrink_to_fit();
    rawBiasData.clear();
    rawBiasData.shrink_to_fit();
    return conv_data;
}

vector<Ctext> resnet_block(FHEONHEController &fheonHEController, FHEONANNBatchController fheonANNBatchController, string layer, vector<Ctext>& encrytedInputs, int batchSize, int &inputdataWidth, int &inputdataSize,
                 int inputChannels, int outputChannels, int reluScale, bool bootstrapState, bool shortcutState){
    printModelLayer(layer);

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
        
        convData = fheonHEController.batch_bootstrap_function(convData, outputChannels, bootstrap_level);
            }

    int totalSize = (batchSize*dataSize); 
    auto scalingVals = fheonHEController.read_batch_scaling_values(convData, outputChannels, totalSize);

    if(layer == "layer4_block2"){
        for(int i=0; i<outputChannels; i++){
            scalingVals[i] = scalingFact*scalingVals[i]; 
        }
    }

    
    convData = fheonANNBatchController.he_batch_relu(convData, scalingVals, outputChannels, totalSize, polyDegee);
    
    auto sum_convData = convolution_block(fheonHEController, fheonANNBatchController, layer+"_conv2", convData, batchSize, dataWidth, dataSize, outputChannels, outputChannels, stride);
    scalingVals = fheonHEController.read_batch_scaling_values(sum_convData, outputChannels, totalSize);

    convData.clear();
    sum_convData = fheonANNBatchController.he_batch_sum_ciphertexts(sum_convData, shortcuts, outputChannels);
    encrytedInputs.clear();
    shortcuts.clear();

    
    sum_convData = fheonHEController.batch_bootstrap_function(sum_convData, outputChannels, bootstrap_level);
    
    scalingVals = fheonHEController.read_batch_scaling_values(sum_convData, outputChannels, totalSize);
    cout << "After SumScaling Values-------: " << scalingVals << endl;
    
    
    sum_convData = fheonANNBatchController.he_batch_relu(sum_convData, scalingVals, outputChannels, totalSize, polyDegee);
    
    return sum_convData;
}

Ctext fc_layer_block(FHEONHEController &fheonHEController, FHEONANNBatchController fheonANNBatchController, string layer, Ctext encrytedInput, int batchSize, int inputChannels, int outputChannels, int rotPositions){
    printModelLayer(layer);
   
    string dataPath = weights_folder+layer;
    auto fc_rawKernelData = load_batch_fc_weights(dataPath+"_weight.csv", outputChannels, batchSize, inputChannels);
    auto fc_rawBiasData = load_batch_fc_bias(dataPath+"_bias.csv", outputChannels, batchSize);
    int encodeLevel = encrytedInput->GetLevel();
    vector<Ptext> fc_kernelData;
    for(int i=0; i < outputChannels; i++){
        // cout << "fc kernel size: " << fc_rawKernelData[i].size() << endl;
        auto encodeWeights = fheonHEController.encode_input(fc_rawKernelData[i], encodeLevel);
        fc_kernelData.push_back(encodeWeights);
    }
    Ptext fc_BiasData = fheonHEController.encode_input(fc_rawBiasData, encodeLevel);

    
    Ctext fcData = fheonANNBatchController.he_batch_linear(encrytedInput, fc_kernelData, fc_BiasData, batchSize, inputChannels, outputChannels, rotPositions);
    
    fc_kernelData.clear();
    fc_kernelData.shrink_to_fit();
    return fcData;
}
