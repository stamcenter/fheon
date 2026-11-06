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

/**
 * LeNet-5 with the diagonal primitives and a small memory footprint (the circuit and the results of LeNet5Diagonal).
 *
 * LeNet5Diagonal keeps every key of the network in memory for the whole run: the bootstrapping keys the context
 * generates for 4096, 2048 and 1024 slots (the network bootstraps only once, on 128 slots), the EvalSum keys of the
 * full batch (the diagonal primitives never call EvalSum) and the rotations of all the layers at once, plus every weight
 * encoded before the first image. At a standard security level the ring dimension grows from 2^13 to 2^16 or more, and
 * all of this no longer fits in memory.
 *
 * This variant keeps only what the current layers need:
 *  - the context generates no bootstrapping and no EvalSum keys (contextBootstrapKeys, contextSumKeys = false);
 *  - the keys are split into three files, generated one after the other and cleared from memory in between:
 *      lenet5_conv.bin : rotations of conv1, pool1 and conv2 (4096 slots)
 *      lenet5_fc.bin   : rotations of fc1 and fc2
 *      lenet5_boot.bin : bootstrapping keys for 128 slots and the rotations of fc3
 *    and during the inference only the file of the current layers is loaded: the rotation keys in memory are cleared
 *    (ClearEvalAutomorphismKeys, their memory returned with malloc_trim) before the next file is read. The context and
 *    the relinearization key stay as they are (clear_context and read_evaluation_keys also release the crypto contexts
 *    of OpenFHE, which this model must not do while its context is in use);
 *  - every layer encodes its weights right before it runs and releases them after, so the plaintexts of only one layer
 *    are in memory at a time.
 */

#include <iostream>
#include <malloc.h>
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

/** the key files of the three phases of the network */
const string convKeys = "lenet5_conv.bin";
const string fcKeys = "lenet5_fc.bin";
const string bootKeys = "lenet5_boot.bin";

/** drop the rotation (and bootstrapping) keys in memory and return their memory to the system */
static void clear_rotation_keys() {
    context->ClearEvalAutomorphismKeys();
    malloc_trim(0);
}

/** the rotation keys of the next layers in place of the current ones */
static void use_keys(const string &file) {
    auto t = startTime();
    clear_rotation_keys();
    fheonHEController.load_rotation_keys(file);
    if (verbose_level >= level_two) { printDuration(t, "Keys " + file + " loaded", false); }
}

int main(int argc, char *argv[]) {
    weights_folder = "./../weights/lenet5/";
    predictions_file = "./../results/lenet5/fhepredictions.txt";
    auto runtime_config = parse_runtime_args(argc, argv, DEFAULT_TEST_SIZE, INDEX_VALUE);
    int test_size = runtime_config.test_size;
    int index_value = runtime_config.index_value;

    auto begin_time = startTime();
    printWelcomeMessage();
    bool loadContext = runtime_config.load_context;
    FHEONHEController::HEConfig heinitConfig;
    bool standardLevel = heinitConfig.securityLevel != 0;
    /** the circuit of LeNet5Diagonal: ReLU degree 27, one bootstrapping after fc2 with the performance-efficient
     *  parameters; leveled (no bootstrapping) at the standard levels, see the ring optimization below */
    bool useBootstrapping = !standardLevel;
    int polyDegree = 27;
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
    heinitConfig.mlevelBootstrap = useBootstrapping ? max(postBootstrapLevels + 2, preBootstrapLevels + 1 - bootstrapDepth)
                                                    : preBootstrapLevels + postBootstrapLevels + 1;
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
    /** no keys with the context: the key files below hold exactly the keys the layers use */
    heinitConfig.contextBootstrapKeys = false;
    heinitConfig.contextSumKeys = false;
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
    printDuration(begin_time, loadContext ? "Context Loaded" : "Context Generated and Keys Serialization", false);

    int kernelWidth = 5;
    /** 2^12, or every slot of the ring of a standard security level (fillSlots) */
    int numSlots = fheonHEController.num_slots;
    vector<int> imgWidth = {28, 24, 12, 8, 4};
    vector<int> channels = {1, 6, 16, 256, 120, 84, 10};

    /*************************************************** Layout of the network (as LeNet5Diagonal) ***************************************
     *  conv1 : (1,28,28) -> (6,24,24), channel c at 672*c, value (y,x) at 28*y + x
     *  pool1 : window sums left in place, value (Y,X) at 56*Y + 2*X (the 1/4 is in the conv2 weights)
     *  conv2 : (6,12,12) -> (16,8,8), 4 groups of 4 channels at 672*g, value (y,x) at 56*y + 2*x + {0, 1, 28, 29};
     *          16 groups of 1 channel at 672*c when the slots of a standard security level allow it
     *  pool2 : fused into fc1, which reads the 4 slots of every window (the 1/4 is in the fc1 weights)
     *  fc1, fc2, fc3 : diagonals, outputs in the first slots of ciphertexts of 128, 128 and 16 slots
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
    double poolScale = fheonANNController.avgpool_weight_scale(2);
    int fc1ReluScale = 8;
    int fc2ReluScale = 10;

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

    /*********************************************** Key generation: one file per phase ********************************************************/
    vector<int> fc3Rotations = fheonANNController.generate_linear_rotation_positions(channels[6], 0, fc3KeysLayout);
    vector<int> convRotations = serialize_rotation_keys({
        fheonANNController.generate_convolution_rotation_positions(imgWidth[0], channels[0], channels[1], kernelWidth, 0, 1, conv1Layout),
        fheonANNController.generate_avgpool_rotation_positions(imgWidth[1], 2, 2, channels[1], pool1Layout),
        fheonANNController.generate_convolution_rotation_positions(imgWidth[2], channels[1], channels[2], kernelWidth, 0, 1, conv2Layout)});
    vector<int> fcRotations = serialize_rotation_keys({
        fheonANNController.generate_linear_rotation_positions(channels[4], 0, fcKeysLayout), fc3Rotations});
    vector<int> bootRotations = serialize_rotation_keys({fc3Rotations});

    auto begin_rotkeygenerate_time = startTime();
    if (!loadContext) {
        /** each phase is generated in an empty key set, serialized, then cleared before the next one */
        clear_rotation_keys();
        cout << "Keys " << convKeys << " (" << convRotations.size() << " rotations): " << convRotations << endl;
        fheonHEController.generate_rotation_keys(convRotations, convKeys, true, false);
        clear_rotation_keys();
        cout << "Keys " << fcKeys << " (" << fcRotations.size() << " rotations): " << fcRotations << endl;
        fheonHEController.generate_rotation_keys(fcRotations, fcKeys, true, false);
        clear_rotation_keys();
        /** leveled chain (standard levels): no bootstrapping, fc3 runs with the keys of the fully connected layers */
        if (useBootstrapping) {
            cout << "Keys " << bootKeys << " (bootstrapping for " << fcDiagonals << " slots, " << bootRotations.size()
                 << " rotations): " << bootRotations << endl;
            fheonHEController.generate_bootstrapping_and_rotation_keys(bootRotations, fcDiagonals, bootKeys, true, false);
            clear_rotation_keys();
        }
    }
    /** the bootstrapping precomputations of the 128 slots of fc2 (no keys: those are in lenet5_boot.bin) */
    if (useBootstrapping) {
        fheonHEController.prepare_bootstrapping(fcDiagonals);
    }
    printDuration(begin_rotkeygenerate_time, "Rotation KeyGen (3 key files)", false);

    /*************************************************** Levels of the layers **************************************************/
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

    /** the raw weights stay in memory (a few hundred KB); the encoded ones are made layer by layer */
    auto wloading_time = startTime();
    auto conv1_biasVector = load_bias(dataPath+"conv1_bias.csv");
    auto conv1_rawKernelData = load_weights(dataPath+"conv1_weight.csv", channels[1], channels[0], kernelWidth, kernelWidth);
    auto conv2_rawKernelData = load_weights(dataPath+"conv2_weight.csv", channels[2], channels[1], kernelWidth, kernelWidth);
    auto conv2_biasVector = load_bias(dataPath+"conv2_bias.csv");
    auto fc1_biasVector = load_bias(dataPath+"fc1_bias.csv");
    auto fc1_rawKernelData = load_fc_weights(dataPath+"fc1_weight.csv", channels[4], channels[3]);
    for (auto &bias : fc1_biasVector) { bias /= fc1ReluScale; }
    fc1Layout.inputPositions = fc1Positions;
    fc1Layout.weightScale = poolScale/fc1ReluScale;
    auto fc2_biasVector = load_bias(dataPath+"fc2_bias.csv");
    auto fc2_rawKernelData = load_fc_weights(dataPath+"fc2_weight.csv", channels[5], channels[4]);
    for (auto &bias : fc2_biasVector) { bias /= fc2ReluScale; }
    fc2Layout.weightScale = static_cast<double>(fc1ReluScale)/fc2ReluScale;
    auto fc3_biasVector = load_bias(dataPath+"fc3_bias.csv");
    auto fc3_rawKernelData = load_fc_weights(dataPath+"fc3_weight.csv", channels[6], channels[5]);
    fc3Layout.weightScale = fc2ReluScale;
    printDuration(wloading_time, "Weights Loading Time", false);

    /************************************************************************************************ */
    int numImages = 1;
    int imageSize = channels[0]*(imgWidth[0]* imgWidth[0]);
    int numtoShow = test_size+index_value;
    int conv1DataSize = channels[1]*channelStride;
    unsigned char** mnistData = read_mnist_images(mnistPath, numImages, imageSize);
    vector<double> readImage;

    for (int imageIndex = index_value; imageIndex < numtoShow; imageIndex++) {
        unsigned char* image = mnistData[imageIndex];
        readImage = read_single_mnist_image(image, imageSize);
        Ctext encryptedImage = fheonHEController.encrypt_input_at_level(readImage, inputLevel);
        cout << endl << imageIndex+1 << " - image Read, Normalized and Encrypt"<< endl;
        auto inference_time = startTime();
        Ctext convData;
        int reluScale = 10;

        /*** conv1 (1,28,28) -> (6,24,24), ReLU, pool1; conv2 (6,12,12) -> (16,8,8), ReLU: keys of the convolutions */
        use_keys(convKeys);
        printModelLayer("Layer", 1);
        {
            auto kernel = fheonANNController.encode_convolution_kernel(fheonHEController, conv1_rawKernelData, imgWidth[0], numSlots, conv1Level, 1, 0, conv1Layout);
            auto bias = fheonANNController.encode_convolution_bias(fheonHEController, conv1_biasVector, imgWidth[1], numSlots, conv1Level, conv1Layout);
            convData = fheonANNController.he_convolution(encryptedImage, kernel, bias, imgWidth[0], channels[0], channels[1], kernelWidth, 0, 1, conv1Layout);
        }
        reluScale = fheonHEController.read_scaling_value(convData, conv1DataSize);
        convData = fheonANNController.he_relu(convData, reluScale, convData->GetSlots(), polyDegree);
        convData = fheonANNController.he_avgpool(convData, imgWidth[1], channels[1], 2, 2, pool1Layout);

        printModelLayer("Layer", 2);
        {
            auto kernel = fheonANNController.encode_convolution_kernel(fheonHEController, conv2_rawKernelData, imgWidth[2], numSlots, conv2Level, 1, 0, conv2Layout);
            auto bias = fheonANNController.encode_convolution_bias(fheonHEController, conv2_biasVector, imgWidth[3], numSlots, conv2Level, conv2Layout);
            convData = fheonANNController.he_convolution(convData, kernel, bias, imgWidth[2], channels[1], channels[2], kernelWidth, 0, 1, conv2Layout);
        }
        convData = fheonANNController.he_relu(convData, reluScale, convData->GetSlots(), polyDegree);

        /*** fc1 (the second pooling is part of it), ReLU, fc2: keys of the fully connected layers */
        use_keys(fcKeys);
        printModelLayer("Layer", 3);
        {
            auto kernel = fheonANNController.encode_linear_weights(fheonHEController, fc1_rawKernelData, numSlots, fc1Level, fc1Layout);
            Ptext bias = fheonANNController.encode_linear_bias(fheonHEController, fc1_biasVector, numSlots, fc1Level);
            convData = fheonANNController.he_linear(convData, kernel, bias, channels[3], channels[4], 0);
        }
        convData = fheonANNController.he_relu(convData, 1, convData->GetSlots(), polyDegree);

        printModelLayer("Layer", 4);
        {
            auto kernel = fheonANNController.encode_linear_weights(fheonHEController, fc2_rawKernelData, fcDiagonals, fc2Level, fc2Layout);
            Ptext bias = fheonANNController.encode_linear_bias(fheonHEController, fc2_biasVector, fcDiagonals, fc2Level);
            convData = fheonANNController.he_linear(convData, kernel, bias, channels[4], channels[5], 0);
        }

        /*** bootstrapping on the 128 slots of fc2, ReLU, fc3: bootstrapping keys */
        if (useBootstrapping) {
            use_keys(bootKeys);
            convData = fheonHEController.bootstrap_function(convData, bootstrapIterations);
        }
        convData = fheonANNController.he_relu(convData, 1, convData->GetSlots(), polyDegree);

        printModelLayer("Layer", 5);
        {
            int fc3Level = fheonANNController.convolution_kernel_level(convData, imgWidth[0]);
            auto kernel = fheonANNController.encode_linear_weights(fheonHEController, fc3_rawKernelData, fcDiagonals, fc3Level, fc3Layout);
            vector<double> fc3_biasPadded = fc3_biasVector;
            Ptext bias = fheonANNController.encode_linear_bias(fheonHEController, fc3_biasPadded, fcDiagonals, fc3Level);
            convData = fheonANNController.he_linear(convData, kernel, bias, channels[5], channels[6], 0);
        }

        string infereMessage = "Total Run Time for Image " + to_string(imageIndex + 1);
        printDuration(inference_time, infereMessage, false);
        fheonHEController.read_inferenced_label(convData, channels[6], predictions_file);
    }

    clear_rotation_keys();
    cout << "All predicted results printed to File." << endl;
    clear_mnist_images(mnistData, numImages);
    return 0;
}
