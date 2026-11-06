
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

#ifndef DEFAULT_TEST_SIZE
#define DEFAULT_TEST_SIZE 250
#endif

#ifndef INDEX_VALUE
#define INDEX_VALUE 0
#endif

using namespace std;

CryptoContext<DCRTPoly> context;
FHEONHEController fheonHEController(context);

/** a single bootstrapping iteration: the second one doubles the cost without improving the precision when no precision is supplied */
const int bootstrapIterations = 1;
/** the ANN controller: every call below uses the shared interface of the ANN controllers */
using ANNController = ANNDiagonalController;

int main(int argc, char *argv[]) {
    weights_folder = "./../weights/lenet5/";
    predictions_file = "./../results/lenet5/fhepredictions.txt";
    auto runtime_config = parse_runtime_args(argc, argv, DEFAULT_TEST_SIZE, INDEX_VALUE);
    int test_size = runtime_config.test_size;
    int index_value = runtime_config.index_value;

    auto begin_time = startTime();
    printWelcomeMessage();
    bool loadContext = runtime_config.load_context;
    /** degree of the ReLU approximation: with the fully connected ReLU scales below, 27 gives the same accuracy as 59 on
     *  the first 100 test images (99/100) with one level less per ReLU; 13 loses accuracy. The Chebyshev evaluation uses
     *  6 levels at degree 59, 5 at degree 27 and 4 at degree 13, plus one when the ReLU scales its input */
    int polyDegree = 27;
    FHEONHEController::HEConfig heinitConfig;
    bool standardLevel = heinitConfig.securityLevel != 0;
    /** performance-efficient parameters: one bootstrapping after fc2 is faster than a leveled chain long enough for the
     *  whole network (2.2s against 2.6s). Standard levels: leveled, see the ring optimization below */
    bool useBootstrapping = !standardLevel;
    int chebyshevLevels = (polyDegree > 27) ? 6 : (polyDegree > 13) ? 5 : 4;
    int reluLevels = chebyshevLevels + 1;
    int preBootstrapLevels = 3*chebyshevLevels + 6;
    int postBootstrapLevels = chebyshevLevels + 1;
    /** depth of the bootstrapping with the level budget {3, 3} */
    int bootstrapDepth = 16;
    heinitConfig.ringDim = 13;
    heinitConfig.numSlots = 12;
    /** at a standard security level the ring holds more slots (2^15 at 128 bits): conv2 then runs its 16 channels as
     *  16 parallel groups instead of 4 groups of 4 interleaved channels (see the layout below) */
    heinitConfig.fillSlots = true;
    heinitConfig.mlevelBootstrap = useBootstrapping ? max(postBootstrapLevels + 2, preBootstrapLevels + 1 - bootstrapDepth) : preBootstrapLevels + postBootstrapLevels + 1;
    heinitConfig.dcrtBits = 46;
    heinitConfig.firstMod = 50;
    heinitConfig.numDigits = 4;
    heinitConfig.levelBudget = {3, 3};
    /** standard security levels: a leveled chain (no bootstrapping) with a small scale. The bootstrapping needs the
     *  46-bit scale (40 or 32 bits make the bootstrapped result undecryptable), which keeps the ring at 2^16 at 128 bits.
     *  Without it the 28-level chain stays precise down to a 22-bit scale (same predictions and logits within 0.05 on
     *  the first test images), so the ring optimization below brings it to 2^15 at 128 bits: scale 26 bits, first
     *  modulus 32 bits (6 bits above the scale for the logits of the last level), 8 key-switching digits. A 2^15 ring
     *  has too few NTT primes below 26 bits for the 29 towers, so 26 bits is the smallest scale OpenFHE builds there */
    if (standardLevel) {
        heinitConfig.firstMod = heinitConfig.dcrtBits + 6;
        heinitConfig.minScaleBits = 23;
    }
    heinitConfig.serialize = true;
    if (!runtime_config.keys_folder.empty()) {
        heinitConfig.keysFolder = runtime_config.keys_folder;
    }
    if (loadContext || useBootstrapping) {
        fheonHEController.load_context(heinitConfig, loadContext);
    } else {
        fheonHEController.generate_leveled_context(heinitConfig);
    }
    context = fheonHEController.getContext();
    ANNController fheonANNController(context);
    printDuration(begin_time, "Context Generated and Keys Serialization", false);

    int kernelWidth = 5;
    /** 2^12, or every slot of the ring of a standard security level (fillSlots) */
    int numSlots = fheonHEController.num_slots;
    vector<int> imgWidth = {28, 24, 12, 8, 4};
    vector<int> channels = {1, 6, 16, 256, 120, 84, 10};

    /*************************************************** Layout of the network ************************************************************
     * No layer compacts its output: the rows keep the 28-slot stride of the image all the way.
     *  conv1 : (1,28,28) -> (6,24,24), channel c at 672*c, value (y,x) at 28*y + x
     *  pool1 : window sums left in place, value (Y,X) at 56*Y + 2*X (the 1/4 is in the conv2 weights)
     *  conv2 : (6,12,12) -> (16,8,8), 4 groups of 4 channels at 672*g; the 4 channels of a group are
     *          interleaved in the gaps left by pool1, value (y,x) at 56*y + 2*x + {0, 1, 28, 29}.
     *          With 16384 slots or more (standard security levels): 16 groups of 1 channel, channel c at 672*c
     *  pool2 : fused into fc1, which reads the 4 slots of every window (the 1/4 is in the fc1 weights)
     *  fc1, fc2, fc3 : diagonals (their outputs rounded up to a power of two), outputs in the first slots of ciphertexts of 128, 128 and 16 slots
     **************************************************************************************************************************************/
    int rowStride = imgWidth[0];
    int colStride = 1;
    int channelStride = imgWidth[1]*rowStride;
    vector<int> conv1SubShifts = {0};
    /** multiplexed conv2: its 16 output channels are packed as numGroups groups of channelStride slots, the channels of a
     *  group interleaved in the gaps pool1 leaves (sub shifts). 4096 slots hold 4 groups of 4 channels: 4 passes of
     *  6 + 4 - 1 diagonals. With the slots of a standard security level all 16 channels run in parallel, one per group:
     *  a single pass of 6 + 16 - 1 diagonals (21 rotations and products of 25 taps instead of 36, and 1 final rotation
     *  instead of 4) */
    bool parallelConv2 = numSlots >= channels[2]*channelStride;
    vector<int> conv2SubShifts = parallelConv2 ? vector<int>{0} : vector<int>{0, colStride, rowStride, rowStride + colStride};
    int conv2Groups = channels[2]/conv2SubShifts.size();
    cout << "conv2 layout: " << conv2Groups << " groups of " << conv2SubShifts.size() << " channel(s) on " << numSlots << " slots" << endl;
    int fcDiagonals = 128;
    /** the pooling returns window sums: the next layer takes the 1/4 in its weights */
    double poolScale = fheonANNController.avgpool_weight_scale(2);
    /** scales of the two fully connected ReLUs: the largest pre-activations over the MNIST test set are 6.8 and 9.3 */
    int fc1ReluScale = 8;
    int fc2ReluScale = 10;

    /** slots holding every input of fc1: the 2x2 window of (channel, Y, X) in the conv2 output */
    vector<vector<int>> fc1Positions(channels[3]);
    for (int ch = 0; ch < channels[2]; ch++) {
        int groupStart = (ch/conv2SubShifts.size())*channelStride + conv2SubShifts[ch % conv2SubShifts.size()];
        for (int row = 0; row < imgWidth[3]; row++) {
            for (int col = 0; col < imgWidth[3]; col++) {
                int input = ch*imgWidth[4]*imgWidth[4] + (row/2)*imgWidth[4] + (col/2);
                fc1Positions[input].push_back(groupStart + row*2*rowStride + col*2*colStride);
            }
        }
    }

    /** strided layouts of the two "valid" convolutions and of the pooling (see the layout description above) */
    FHEONLayout conv1Layout, conv2Layout, pool1Layout, fc1Layout, fc2Layout, fc3Layout, fcKeysLayout, fc3KeysLayout;
    conv1Layout.rowStride = rowStride;
    conv1Layout.colStride = colStride;
    conv1Layout.channelStride = channelStride;
    conv1Layout.subShifts = conv1SubShifts;
    conv2Layout.rowStride = 2*rowStride;
    conv2Layout.colStride = 2*colStride;
    conv2Layout.channelStride = channelStride;
    conv2Layout.subShifts = conv2SubShifts;
    conv2Layout.numGroups = conv2Groups;
    conv2Layout.weightScale = poolScale;
    pool1Layout.rowStride = rowStride;
    pool1Layout.colStride = colStride;
    fcKeysLayout.inputSlots = numSlots;
    fc3KeysLayout.inputSlots = fcDiagonals;

    //** generate rotation keys*/
    vector<vector<int>> rotation_keys;
    rotation_keys.push_back(fheonANNController.generate_convolution_rotation_positions(imgWidth[0], channels[0], channels[1], kernelWidth, 0, 1, conv1Layout));
    rotation_keys.push_back(fheonANNController.generate_avgpool_rotation_positions(imgWidth[1], 2, 2, channels[1], pool1Layout));
    rotation_keys.push_back(fheonANNController.generate_convolution_rotation_positions(imgWidth[2], channels[1], channels[2], kernelWidth, 0, 1, conv2Layout));
    rotation_keys.push_back(fheonANNController.generate_linear_rotation_positions(channels[4], 0, fcKeysLayout));
    rotation_keys.push_back(fheonANNController.generate_linear_rotation_positions(channels[6], 0, fc3KeysLayout));

    /*** join all keys and generate unique values only */
    vector<int> rotation_positions =  serialize_rotation_keys(rotation_keys);
    
    /*** Generate the rotation positions, generate rotation keys, and load rotation keys */
    auto begin_rotkeygenerate_time = startTime();
    cout << "This is the rotation positions (" << rotation_positions.size() <<"): " << rotation_positions << endl;
    fheonHEController.generate_rotation_keys(rotation_positions, "rotation_keys.bin", true, true);
    /** the only bootstrapping of the network runs on the 128-slot output of fc2 */
    if (useBootstrapping) {
        fheonHEController.generate_bootstrapping_keys_for_slots(fcDiagonals);
    }
    printDuration(begin_rotkeygenerate_time, "Rotation KeyGen (position, gen, and load) Time", false);

    /*************************************************** Prepare Weights for the network **************************************************/
    /** the levels are fixed by the circuit: conv1 on the fresh ciphertext, 1 level per layer, 7 per ReLU that scales its
     *  input and 6 per ReLU whose input is already scaled. conv1 + ReLU + conv2 + ReLU + fc1 + ReLU + fc2 use 24 levels
     *  and have to end with 3 towers; only the last ReLU and fc3 follow the bootstrapping */
    auto wloading_time = startTime();
    string dataPath = weights_folder;
    int numTowers = context->GetCryptoParameters()->GetElementParams()->GetParams().size();
    int inputLevel = numTowers - (preBootstrapLevels + 2 + (useBootstrapping ? 0 : postBootstrapLevels));
    if (inputLevel < 0) {
        cerr << "The modulus chain is too short for the circuit before the bootstrapping." << endl;
        return 1;
    }
    int conv1Level = inputLevel;
    int conv2Level = conv1Level + 1 + reluLevels;
    int fc1Level = conv2Level + 1 + reluLevels;
    int fc2Level = fc1Level + 1 + (reluLevels - 1);

    auto conv1_biasVector = load_bias(dataPath+"conv1_bias.csv");
    auto conv1_rawKernelData = load_weights(dataPath+"conv1_weight.csv", channels[1], channels[0], kernelWidth, kernelWidth);
    auto conv1_kernelData = fheonANNController.encode_convolution_kernel(fheonHEController, conv1_rawKernelData, imgWidth[0], numSlots, conv1Level, 1, 0, conv1Layout);
    auto conv1biasEncoded = fheonANNController.encode_convolution_bias(fheonHEController, conv1_biasVector, imgWidth[1], numSlots, conv1Level, conv1Layout);

    auto conv2_rawKernelData = load_weights(dataPath+"conv2_weight.csv", channels[2], channels[1], kernelWidth, kernelWidth);
    auto conv2_biasVector = load_bias(dataPath+"conv2_bias.csv");
    auto conv2_kernelData = fheonANNController.encode_convolution_kernel(fheonHEController, conv2_rawKernelData, imgWidth[2], numSlots, conv2Level, 1, 0, conv2Layout);
    auto conv2biasEncoded = fheonANNController.encode_convolution_bias(fheonHEController, conv2_biasVector, imgWidth[3], numSlots, conv2Level, conv2Layout);

     /*** first fully layer connected kernel and bias */
    auto fc1_biasVector = load_bias(dataPath+"fc1_bias.csv");
    auto fc1_rawKernelData = load_fc_weights(dataPath+"fc1_weight.csv", channels[4], channels[3]);
    /** the fully connected layers produce their outputs already divided by the scale of their ReLU, and the next layer
     *  multiplies it back (ReLU(x / s) = ReLU(x) / s): the ReLUs save the level of the scaling, and the bootstrapping
     *  after fc2 works on values within [-1, 1] */
    for (auto &bias : fc1_biasVector) { bias /= fc1ReluScale; }
    fc1Layout.inputPositions = fc1Positions;
    fc1Layout.weightScale = poolScale/fc1ReluScale;
    auto fc1_kernelData = fheonANNController.encode_linear_weights(fheonHEController, fc1_rawKernelData, numSlots, fc1Level, fc1Layout);
    Ptext fc1baisVector = fheonANNController.encode_linear_bias(fheonHEController, fc1_biasVector, numSlots, fc1Level);
    
     /*** second fully layer connected weights and bias */
    auto fc2_biasVector = load_bias(dataPath+"fc2_bias.csv");
    auto fc2_rawKernelData = load_fc_weights(dataPath+"fc2_weight.csv", channels[5], channels[4]);
    for (auto &bias : fc2_biasVector) { bias /= fc2ReluScale; }
    fc2Layout.weightScale = static_cast<double>(fc1ReluScale)/fc2ReluScale;
    auto fc2_kernelData = fheonANNController.encode_linear_weights(fheonHEController, fc2_rawKernelData, fcDiagonals, fc2Level, fc2Layout);
    Ptext fc2baisVector = fheonANNController.encode_linear_bias(fheonHEController, fc2_biasVector, fcDiagonals, fc2Level);

     /*** third fully layer connected weights and bias: encoded once the level after the bootstrapping is known */
    auto fc3_biasVector = load_bias(dataPath+"fc3_bias.csv");
    auto fc3_rawKernelData = load_fc_weights(dataPath+"fc3_weight.csv", channels[6], channels[5]);
    vector<Ptext> fc3_kernelData;
    fc3Layout.weightScale = fc2ReluScale;
    Ptext fc3baisVector;
    int fc3Level = -1;

    printDuration(wloading_time, "Weights Loading Time", false);

    /************************************************************************************************ */
     /**** Read the MNIST Images and inference them */
    
    int numImages = 1;
    int imageSize = channels[0]*(imgWidth[0]* imgWidth[0]);
    int numtoShow = test_size+index_value;
    int reluScale = 10;
    int conv1DataSize = channels[1]*channelStride;
    
    /*** Read MNIST images ***/
    unsigned char** mnistData = read_mnist_images(mnistPath, numImages, imageSize);
    vector<double> readImage;

    for (int imageIndex = index_value; imageIndex < numtoShow; imageIndex++) {
        unsigned char* image = mnistData[imageIndex];
        readImage = read_single_mnist_image(image, imageSize);
        // display_mnist_image(image, imageSize, true);
        Ctext encryptedImage = fheonHEController.encrypt_input_at_level(readImage, inputLevel);
        cout << endl << imageIndex+1 << " - image Read, Normalized and Encrypt"<< endl;
        /************************************************************************************************ */
        /***** The first Convolution Layer takes  image=(1,28,28), kernel=(6,1,5,5) 
         * stride=1, pooling=0 output= (6,24,24) = 3456 vals */
        reluScale = 10;
        auto inference_time = startTime();
        
        printModelLayer("Layer", 1);
        auto convData = fheonANNController.he_convolution(encryptedImage, conv1_kernelData, conv1biasEncoded, imgWidth[0], channels[0], channels[1], kernelWidth, 0, 1, conv1Layout);
        reluScale = fheonHEController.read_scaling_value(convData, conv1DataSize);
        convData = fheonANNController.he_relu(convData, reluScale, convData->GetSlots(), polyDegree);
        convData = fheonANNController.he_avgpool(convData, imgWidth[1], channels[1], 2, 2, pool1Layout);
        
        /*** Second Conv2d Layer input = (6,12,12), kernel=(16,6,5,5) 
         * striding =1, padding = 0 output = (16,8,8) ***/
        printModelLayer("Layer", 2);
        convData = fheonANNController.he_convolution(convData, conv2_kernelData, conv2biasEncoded, imgWidth[2], channels[1], channels[2], kernelWidth, 0, 1, conv2Layout);
        convData = fheonANNController.he_relu(convData, reluScale, convData->GetSlots(), polyDegree);
        if (verbose_level >= level_two) { cout << "ReLU Scale 2: " << reluScale << endl; }
        
        /*** fully connected layers; the second pooling is part of the first one */
        printModelLayer("Layer", 3);
        convData = fheonANNController.he_linear(convData, fc1_kernelData, fc1baisVector, channels[3], channels[4], 0);
        convData = fheonANNController.he_relu(convData, 1, convData->GetSlots(), polyDegree);

        printModelLayer("Layer", 4);
        convData = fheonANNController.he_linear(convData, fc2_kernelData, fc2baisVector, channels[4], channels[5], 0);
        if (useBootstrapping) {
            convData = fheonHEController.bootstrap_function(convData, bootstrapIterations);
        }
        convData = fheonANNController.he_relu(convData, 1, convData->GetSlots(), polyDegree);

        printModelLayer("Layer", 5);
        int levelNow = fheonANNController.convolution_kernel_level(convData, imgWidth[0]);
        if (levelNow != fc3Level) {
            fc3Level = levelNow;
            fc3_kernelData = fheonANNController.encode_linear_weights(fheonHEController, fc3_rawKernelData, fcDiagonals, fc3Level, fc3Layout);
            vector<double> fc3_biasPadded = fc3_biasVector;
            fc3baisVector = fheonANNController.encode_linear_bias(fheonHEController, fc3_biasPadded, fcDiagonals, fc3Level);
        }
        convData = fheonANNController.he_linear(convData, fc3_kernelData, fc3baisVector, channels[5], channels[6], 0);

        string infereMessage = "Total Run Time for Image " + to_string(imageIndex + 1);  
        printDuration(inference_time, infereMessage, false);
        fheonHEController.read_inferenced_label(convData, channels[6], predictions_file);
    }
    
    cout << "All predicted results printed to File." << endl;
    clear_mnist_images(mnistData, numImages);
   return 0;
 
}
