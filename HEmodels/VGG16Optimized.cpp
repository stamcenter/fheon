
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

using namespace std;
CryptoContext<DCRTPoly> context;
FHEONHEController fheonHEController(context);

#ifndef DEFAULT_ARG
#define DEFAULT_ARG 100
#endif

vector<int> measuringTime;
auto startIn = get_current_time();

vector<int> slotsValues = {14, 14, 14, 14, 14, 14};

Ctext convolution_relu_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer, Ctext encrytedVector, int &dataWidth, int &dataSize, int kernelWidth, int padding, 
                                            int striding, int inputChannels, int outputChannels, int reluScale, bool bootstrapState = true);
Ctext FClayer_relu_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer, Ctext encrytedVector, int inputChannels, int outputChannels, int reluScale, int rotPosition);

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
    cout << "---------------VGG16-------------"<< to_string(DEFAULT_ARG) << "--------------------" << endl; 
    
    /**** Read the CIFAR-10 Images and inference them */
    string cifar10tPath = "./../images/cifar-10-batches-bin/test_batch.bin";
    int numImages = DEFAULT_ARG;
    int img_cols = 32;
    int img_depth = 3;
    int dataWidth = img_cols;
    int dataSize = img_depth*pow(img_cols, 2);
    vector<vector<double>> imagesData = read_images(cifar10tPath, numImages, dataSize);
    ofstream outFile;
    outFile.open("./../results/vgg16/fhepredictions.txt", ios_base::app);
    Ctext convData;
    Ptext decryptedData;
    int kernelWidth = 3; 
    int padding = 1;
    int striding = 1;
    int avgpoolSize = 2;
    vector<int> channelValues = {16, 32, 64, 128, 1024, 10};
    int rotPositions = 32;
    
    //** generate rotation keys for conv_layer 1 */
    auto conv1_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, img_depth, channelValues[0]);
    auto avgpool1_keys = fheonANNController.generate_avgpool_optimized_rotation_positions(dataWidth,channelValues[0], avgpoolSize, avgpoolSize, false, "single_channel");
    dataWidth = dataWidth/2;

    auto conv2_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channelValues[0], channelValues[1]);
    auto conv2_1_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channelValues[1], channelValues[1]);
    auto avgpool2_keys = fheonANNController.generate_avgpool_optimized_rotation_positions(dataWidth, channelValues[1], avgpoolSize, avgpoolSize);
    dataWidth = dataWidth/2;

    auto conv3_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth,  channelValues[1], channelValues[2]);
    auto conv4_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channelValues[2], channelValues[2]);
    auto avgpool3_keys = fheonANNController.generate_avgpool_optimized_rotation_positions(dataWidth,  channelValues[2], avgpoolSize, avgpoolSize);
    dataWidth = dataWidth/2;

    auto conv5_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channelValues[2], channelValues[3]);
    auto avgpool4_keys = fheonANNController.generate_avgpool_optimized_rotation_positions(dataWidth,  channelValues[3], avgpoolSize, avgpoolSize);
    dataWidth = dataWidth/2;

    auto conv6_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channelValues[3], channelValues[3]);
    auto conv7_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channelValues[3], channelValues[3]);
    auto conv8_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth,  channelValues[3], channelValues[3]);
    auto avgpool5_keys = fheonANNController.generate_avgpool_optimized_rotation_positions(dataWidth,  channelValues[3],  avgpoolSize,  avgpoolSize, true, "multi_channels", rotPositions);
 
    /** Generate Fully Connected Keys */
    auto fc_keys = fheonANNController.generate_linear_rotation_positions(channelValues[4], rotPositions);
    fc_keys.push_back(channelValues[2]);
    fc_keys.push_back(channelValues[3]);
    fc_keys.push_back(channelValues[4]);
    /************************************************************************************************ */
    vector<vector<int>> rkeys_layer1, rkeys_layer2, rkeys_layer3, rkeys_layer4, rkeys_layer5,  fc_layer;
    
    rkeys_layer1.push_back(conv1_keys);
    rkeys_layer1.push_back(avgpool1_keys);

    rkeys_layer2.push_back(conv2_keys);
    rkeys_layer2.push_back(avgpool2_keys);
    rkeys_layer2.push_back(conv2_1_keys);

    rkeys_layer3.push_back(conv3_keys);
    rkeys_layer3.push_back(conv4_keys);
    rkeys_layer3.push_back(avgpool3_keys);

    rkeys_layer4.push_back(conv5_keys);
    rkeys_layer4.push_back(avgpool4_keys);

    rkeys_layer4.push_back(conv6_keys);
    rkeys_layer5.push_back(conv7_keys);
    rkeys_layer5.push_back(conv8_keys);
    rkeys_layer5.push_back(avgpool5_keys);

    fc_layer.push_back(fc_keys);

   /********************************************************************************************************************************************/
    /*** join all keys and generate unique values only */
    vector<int> serkeys_layer1 = serialize_rotation_keys(rkeys_layer1); 
    vector<int> serkeys_layer2 = serialize_rotation_keys(rkeys_layer2);
    vector<int> serkeys_layer3 = serialize_rotation_keys(rkeys_layer3);
    vector<int> serkeys_layer4 = serialize_rotation_keys(rkeys_layer4);
    vector<int> serkeys_layer5 = serialize_rotation_keys(rkeys_layer5);
    vector<int> serkeys_fc_layer = serialize_rotation_keys(fc_layer);
    // /*********************************************** Key Generation ******************************************************************************/
    auto begin_rotkeygenerate_time = startTime();
    // cout << "This is the rotation positions (" << serkeys_block1.size() <<"+" << serkeys_block2.size() << "+" << serkeys_block3.size() << " = " << total_rkeys << "): " << endl;
    cout << "Layer 1 keys (" << serkeys_layer1.size() << ") " << serkeys_layer1 << endl;
    cout << "Layer 2 keys (" << serkeys_layer2.size() << ") " << serkeys_layer2 << endl;
    cout << "Layer 3 keys (" << serkeys_layer3.size() << ") " << serkeys_layer3 << endl;
    cout << "Layer 4 keys (" << serkeys_layer4.size() << ") " << serkeys_layer4 << endl;
    cout << "Layer 5 keys (" << serkeys_layer5.size() << ") " << serkeys_layer5 << endl;
    cout << "FC keys (" << serkeys_fc_layer.size() << ") " << serkeys_fc_layer << endl;

    fheonHEController.generate_bootstrapping_and_rotation_keys(serkeys_layer1, slotsValues[0], "layer1.bin", true);
    fheonHEController.clear_context(slotsValues[0]);
    
    fheonHEController.generate_bootstrapping_and_rotation_keys(serkeys_layer2, slotsValues[1], "layer2.bin", true);
    fheonHEController.clear_context(slotsValues[1]);
    
    fheonHEController.generate_bootstrapping_and_rotation_keys(serkeys_layer3, slotsValues[2], "layer3.bin" , true );
    fheonHEController.clear_context(slotsValues[2]);

    fheonHEController.generate_bootstrapping_and_rotation_keys(serkeys_layer4, slotsValues[3], "layer4.bin", true);
    fheonHEController.clear_context(slotsValues[3]);

    fheonHEController.generate_bootstrapping_and_rotation_keys(serkeys_layer5, slotsValues[4], "layer5.bin", true);
    fheonHEController.clear_context(slotsValues[4]);

    fheonHEController.generate_bootstrapping_and_rotation_keys(serkeys_fc_layer, slotsValues[5], "fc_layer.bin",  true);
    fheonHEController.clear_context(slotsValues[5]);

    printDuration(begin_rotkeygenerate_time, "Rotation KeyGen Time", false);
    /********************************************************************************************************************************************/

    /********************************************************************************************************************************************/
    int reluScale = 20; // not in use at this time
    for (int imageIndex = 0; imageIndex < DEFAULT_ARG; imageIndex++) {
        auto image = imagesData[imageIndex];
        dataWidth = img_cols;
        // display_image(image, imageSize, true);
        Ctext encryptedImage = fheonHEController.encrypt_input(image);
        cout << endl << imageIndex+1 << " - image Read, Normalized and Encrypted with " << image.size() << " Elements" << endl;
        /************************************************************************************************ */
        auto inference_time = startTime();
        // cout<< "Layer 1" << endl;
        fheonHEController.clear_context(slotsValues[5]);
        fheonHEController.load_bootstrapping_and_rotation_keys(slotsValues[1], "layer1.bin", false);
        convData = convolution_relu_block(fheonHEController, fheonANNController, "conv1", encryptedImage, dataWidth, dataSize, kernelWidth, padding, striding, img_depth, channelValues[0], reluScale, false);
        convData = convolution_relu_block(fheonHEController, fheonANNController, "conv2", convData, dataWidth, dataSize, kernelWidth, padding, striding, channelValues[0], channelValues[0], reluScale, false);
        startIn = get_current_time();
        convData = fheonANNController.he_avgpool_optimzed(convData, dataWidth,  channelValues[0], avgpoolSize, avgpoolSize);
        measuringTime.push_back(measureTime(startIn, get_current_time()));
        dataWidth = dataWidth/2;
        dataSize = channelValues[0]* pow(dataWidth, 2);
        // fheonHEController.read_minmax(convData, dataSize);
        // printDuration(inference_time, "avgpool1", false);

        fheonHEController.clear_context(slotsValues[0]);
        fheonHEController.load_bootstrapping_and_rotation_keys(slotsValues[1], "layer2.bin", false);

        // cout<< "Layer 2" << endl;
        convData = convolution_relu_block(fheonHEController, fheonANNController, "conv3", convData, dataWidth, dataSize, kernelWidth, padding, striding, channelValues[0], channelValues[1], reluScale, true);
        convData = convolution_relu_block(fheonHEController, fheonANNController, "conv4", convData, dataWidth, dataSize, kernelWidth, padding, striding, channelValues[1], channelValues[1], reluScale, true);
        convData = fheonHEController.bootstrap_function(convData);
        startIn = get_current_time();
        convData = fheonANNController.he_avgpool_optimzed_with_multiple_channels(convData, dataWidth,  channelValues[1], avgpoolSize, avgpoolSize);
        measuringTime.push_back(measureTime(startIn, get_current_time()));
        dataWidth = dataWidth/2;
        dataSize = channelValues[1]* pow(dataWidth, 2);
        // fheonHEController.read_minmax(convData, dataSize);
        // printDuration(inference_time, "avgpool2", false);

        fheonHEController.clear_context(slotsValues[1]);
        fheonHEController.load_bootstrapping_and_rotation_keys(slotsValues[2], "layer3.bin", false);

        // cout<< "Layer 3" << endl;
        convData = convolution_relu_block(fheonHEController, fheonANNController, "conv5", convData, dataWidth, dataSize, kernelWidth, padding, striding, channelValues[1], channelValues[2], reluScale);
        convData = convolution_relu_block(fheonHEController, fheonANNController, "conv6", convData, dataWidth, dataSize, kernelWidth, padding, striding, channelValues[2], channelValues[2], reluScale);
        convData = convolution_relu_block(fheonHEController, fheonANNController, "conv7", convData, dataWidth, dataSize, kernelWidth, padding, striding, channelValues[2], channelValues[2], reluScale);
        convData = fheonHEController.bootstrap_function(convData);
        startIn = get_current_time();
        convData = fheonANNController.he_avgpool_optimzed_with_multiple_channels(convData, dataWidth,  channelValues[2], avgpoolSize, avgpoolSize);
        measuringTime.push_back(measureTime(startIn, get_current_time()));
        dataWidth = dataWidth/2;
        dataSize = channelValues[2]* pow(dataWidth, 2);
        // fheonHEController.read_minmax(convData, dataSize);
        // printDuration(inference_time, "avgpool3", false);

        fheonHEController.clear_context(slotsValues[2]);
        fheonHEController.load_bootstrapping_and_rotation_keys(slotsValues[3], "layer4.bin", false);

        // cout<< "Layer 4" << endl;
        convData = convolution_relu_block(fheonHEController, fheonANNController, "conv8",  convData, dataWidth, dataSize, kernelWidth, padding, striding, channelValues[2], channelValues[3], reluScale);
        convData = convolution_relu_block(fheonHEController, fheonANNController, "conv9",  convData, dataWidth, dataSize, kernelWidth, padding, striding, channelValues[3], channelValues[3], reluScale);
        convData = convolution_relu_block(fheonHEController, fheonANNController, "conv10", convData, dataWidth, dataSize, kernelWidth, padding, striding, channelValues[3], channelValues[3], reluScale);
        convData = fheonHEController.bootstrap_function(convData);
        startIn = get_current_time();
        convData = fheonANNController.he_avgpool_optimzed_with_multiple_channels(convData, dataWidth,  channelValues[3], avgpoolSize, avgpoolSize);
        measuringTime.push_back(measureTime(startIn, get_current_time()));
        dataWidth = dataWidth/2;
        dataSize = channelValues[3]* pow(dataWidth, 2);
        // fheonHEController.read_minmax(convData, dataSize);
        // printDuration(inference_time, "avgpool4", false);

        fheonHEController.clear_context(slotsValues[3]);
        fheonHEController.load_bootstrapping_and_rotation_keys(slotsValues[4], "layer5.bin", false);
        
        // cout<< "Layer 5" << endl;
        convData = convolution_relu_block(fheonHEController, fheonANNController, "conv11", convData, dataWidth, dataSize, kernelWidth, padding, striding, channelValues[3], channelValues[3], reluScale);
        convData = convolution_relu_block(fheonHEController, fheonANNController, "conv12", convData, dataWidth, dataSize, kernelWidth, padding, striding, channelValues[3], channelValues[3], reluScale);
        convData = convolution_relu_block(fheonHEController, fheonANNController, "conv13", convData, dataWidth, dataSize, kernelWidth, padding, striding, channelValues[3], channelValues[3], reluScale);
        convData = fheonHEController.bootstrap_function(convData);
        startIn = get_current_time();
        convData = fheonANNController.he_globalavgpool(convData, dataWidth,  channelValues[3], avgpoolSize, rotPositions);
        measuringTime.push_back(measureTime(startIn, get_current_time()));
        dataWidth = dataWidth/2;
        dataSize = channelValues[3]* pow(dataWidth, 2);
        // fheonHEController.read_minmax(convData, dataSize);
        // printDuration(inference_time, "avgpool5", false);

        fheonHEController.clear_context(slotsValues[4]);
        fheonHEController.load_bootstrapping_and_rotation_keys(slotsValues[5], "fc_layer.bin", false);
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
    string dataPath = "./../weights/vgg16/"+layer;
    auto biasVector = load_bias(dataPath+"_bias.csv");
    auto rawKernel = load_weights(dataPath+"_weight.csv", outputChannels, 
                        inputChannels, kernelWidth, kernelWidth);
    int inputSize = pow(dataWidth, 2);
    vector<vector<Ptext>> kernelData;
   for(int i=0; i<outputChannels; i++){
        auto encodeKernel = fheonHEController.encode_kernel_optimized(rawKernel[i], inputSize);
        kernelData.push_back(encodeKernel);
    }
    if(bootstrapState){
        encrytedVector = fheonHEController.bootstrap_function(encrytedVector);
    }
    auto convbiasEncoded = fheonHEController.encode_bais_input(biasVector, inputSize);

    startIn = get_current_time();
    auto conv_data = fheonANNController.he_convolution_optimized(encrytedVector, kernelData, convbiasEncoded, dataWidth, inputChannels, outputChannels);
    measuringTime.push_back(measureTime(startIn, get_current_time()));
    // dataWidth = (((dataWidth) + (2*padding) - (kernelWidth - 1) - 1)/striding)+1;
    dataSize = (outputChannels*pow(dataWidth, 2));

    if(bootstrapState){
        conv_data = fheonHEController.bootstrap_function(conv_data);
    }

    /** Temporal relu scale for max accuracy */
    reluScale = fheonHEController.read_scaling_value(conv_data, dataSize);
    startIn = get_current_time();
    conv_data = fheonANNController.he_relu(conv_data, reluScale, dataSize);
    measuringTime.push_back(measureTime(startIn, get_current_time()));
    kernelData.clear();
    kernelData.shrink_to_fit();
    biasVector.clear();
    return conv_data;
}

Ctext FClayer_relu_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer, Ctext encrytedVector, int inputChannels, int outputChannels, int reluScale, int rotPosition){
    string dataPath = "./../weights/vgg16/"+layer;
    auto fc_biasVector = load_bias(dataPath+"_bias.csv");
    auto fc_rawKernel = load_fc_weights(dataPath+"_weight.csv", outputChannels, inputChannels);
    vector<Ptext> fc_kernelData;
     int encoded_level = encrytedVector->GetLevel();
    int num_elements = nextPowerOf2(fc_rawKernel[0].size());
    for(int i=0; i < outputChannels; i++){
        auto encodeWeights = fheonHEController.encode_input(fc_rawKernel[i], num_elements, encoded_level);
        fc_kernelData.push_back(encodeWeights);
    }

    num_elements = nextPowerOf2(fc_biasVector.size());
    Ptext fcbaisVector = context->MakeCKKSPackedPlaintext(fc_biasVector, 1, encoded_level, nullptr, num_elements);
    encrytedVector = fheonHEController.bootstrap_function(encrytedVector);

    Ctext layer_data = fheonANNController.he_linear(encrytedVector, fc_kernelData, fcbaisVector, inputChannels, outputChannels, rotPosition);
    // fheonHEController.read_minmax(layer_data, outputChannels);
    if(reluScale != 0){
        layer_data = fheonHEController.bootstrap_function(layer_data);

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