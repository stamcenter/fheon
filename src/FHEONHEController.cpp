
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
 * @brief FHE controller for defining basic FHE functions used across different neural networks.
 *
 * This class provides fundamental methods for context generation, encryption, encoding, 
 * bootstrapping, and other FHE operations that are utilized throughout the ANN development.
 *
 */

#include <fstream>
#include <filesystem> 
#include <iostream>
// #include <thread>
namespace fs = std::filesystem;

#include "FHEONHEController.h"

/**
 * @brief Update absolute file paths for FHE context and serialization keys.
 *
 * Appends trailing slashes to keys_folder if necessary and constructs full paths
 * for public, secret, multiplication, and sum keys.
 */
void FHEONHEController::update_key_paths() {
    fs::path base(keys_folder);
    if (!base.empty() && base.filename() == ".") {
        base = base.parent_path();
    }
    if (!base.empty() && base.string().back() == '/') {
        keys_folder = base.string();
    } else if (!base.empty()) {
        keys_folder = base.string() + "/";
    }

    crypto_context_file = keys_folder + "crypto-context.bin";
    multi_key_file = keys_folder + "mult-keys.bin";
    sum_key_file = keys_folder + "sum-keys.bin";
    public_key_file = keys_folder + "public-key.bin";
    secret_key_file = keys_folder + "secret-key.bin";
}

/**
 * @brief Compute the PQ value, which defines the application's security level.
 *
 * This function calculates the PQ value from the given polynomial, which is used 
 * to determine the security level of the application.
 *
 * @param poly  Input polynomial used to compute the PQ value.
 *
 * @return The computed PQ value as a double.
 */
double getlogPQ(const DCRTPoly& poly) {
    int n = poly.GetNumOfElements();
    double logPQ = 0;
    for (int i = 0; i < n; i++) {
        auto qi = poly.GetParams()->GetParams()[i]->GetModulus(); 
        logPQ += log(qi.ConvertToDouble()) / log(2);
    }
    return logPQ;
}

/**
 * @brief Set the count of bootstrapping operations performed.
 *
 * @param count The number of bootstrapping operations to set.
 */
void FHEONHEController::set_bootstrap_count(int count){
    num_bootsraps = count;
}


/**
 * @brief Generate FHE crypto context using an HEConfig structure.
 *
 * @param config Configuration structure containing ring dimension, slots, etc.
 */
void FHEONHEController::generate_context(HEConfig config) {
    if (!config.keysFolder.empty()) {
        keys_folder = config.keysFolder;
    }
    update_key_paths();
    generate_context(config.ringDim, config.numSlots, config.mlevelBootstrap, config.dcrtBits, 
                    config.firstMod, config.numDigits, config.levelBudget, config.serialize);
    return;
}

/**
 * @brief Generate the full FHE context with advanced parameters.
 *
 * @param ringDim Ring dimension for CKKS context.
 * @param numSlots Number of slots for packing.
 * @param mlevelBootstrap Multiplication depth/level for bootstrapping.
 * @param dcrtBits Bit size of the DCRT moduli.
 * @param firstMod Scaling factor/modulus for the first level.
 * @param numDigits Number of digits for key switching.
 * @param levelBudget Level budget for bootstrapping.
 * @param serialize Whether to serialize context to files.
 */
void FHEONHEController::generate_context(int ringDim, int numSlots, int mlevelBootstrap, 
                        int dcrtBits, int firstMod, int numDigits, vector<uint32_t> levelBudget,
                        bool serialize) {

    CCParams<CryptoContextCKKSRNS> parameters;
    auto secretKeyDist = SPARSE_TERNARY;

    ScalingTechnique rescaleTech = FLEXIBLEAUTO;
    // ScalingTechnique rescaleTech = FLEXIBLEAUTOEXT;
    level_budget = levelBudget;
    num_slots = 1 << numSlots;
    mult_depth = mlevelBootstrap;

    parameters.SetRingDim(1 << ringDim);
    parameters.SetBatchSize(num_slots);
    parameters.SetScalingModSize(dcrtBits);
    parameters.SetFirstModSize(firstMod);
    parameters.SetNumLargeDigits(numDigits);
    
    parameters.SetSecretKeyDist(secretKeyDist);
    parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
    // parameters.SetSecurityLevel(lbcrypto::HEStd_128_classic);
    parameters.SetScalingTechnique(rescaleTech);

    circuit_depth = mult_depth + FHECKKSRNS::GetBootstrapDepth(level_budget, secretKeyDist);
    parameters.SetMultiplicativeDepth(circuit_depth);

    cout << "Building the FHE Context" << endl;
    cout << "dcrtBits: "<< dcrtBits << " -- firstMod: " << firstMod << endl << "Ciphertexts depth: " 
         << circuit_depth << ", available multiplications: " << circuit_depth - 2 << endl;
   
    context = GenCryptoContext(parameters);
    context->Enable(PKE);
    context->Enable(KEYSWITCH);
    context->Enable(LEVELEDSHE);
    context->Enable(ADVANCEDSHE);
    context->Enable(FHE);
    
    keyPair = context->KeyGen();
    context->EvalMultKeyGen(keyPair.secretKey);
    context->EvalSumKeyGen(keyPair.secretKey);

    ringDim = context->GetRingDimension();
    numSlots = num_slots;
    usint halfnumSlots = numSlots/2;
    context->EvalBootstrapSetup(level_budget, bsgsDim, numSlots);
    context->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);
    
    auto sec_level = parameters.GetSecurityLevel();
    auto logq = context->GetModulus().GetMSB();
     double logPQ = getlogPQ(keyPair.publicKey->GetPublicElements()[0]);
    cout << "Keys Generated." << endl;
    cout << "Cyclotomic Order: " << context->GetCyclotomicOrder() << endl;
    cout << "CKKS scheme is using ring dimension: " << ringDim  << endl;
    cout << "Avaliable numSlots: " << numSlots << "  - halfnumSlots: " << halfnumSlots << endl;
    cout << "LogQ: "<< logq << " - Security Level: " << endl;
    cout << "Security Level: " << sec_level << endl;
    cout << "Ciphertexts depth: " << circuit_depth << endl; 
    cout << "Multiplication Depth: " << mult_depth - 2 << endl;
    cout << "log PQ = " << logPQ << std::endl << std::endl;
    cout << "-----------------------------------------------------------" << endl;
    
    if(serialize){
        write_to_file(keys_folder + "/mult_depth.txt", to_string(mult_depth));
        write_to_file(keys_folder + "/num_slots.txt", to_string(num_slots));
        write_to_file(keys_folder + "/level_budget.txt", to_string(level_budget[0]) + "," + to_string(level_budget[1]));
        keys_serialization();
    }

    context_loaded = false;
    return;
}

/**
 * @brief Generate a standard FHE context with defaults.
 *
 * @param ringDim Ring dimension for CKKS context.
 * @param numSlots Number of slots for packing.
 * @param mlevelBootstrap Multiplication depth/level for bootstrapping.
 * @param serialize Whether to serialize context to files.
 */
void FHEONHEController::generate_context(int ringDim, int numSlots, int mlevelBootstrap, bool serialize) {
    CCParams<CryptoContextCKKSRNS> parameters;

    num_slots = 1 << numSlots;
    int dcrtBits               = 46;
    int firstMod               = 50;

    auto secretKeyDist = SPARSE_TERNARY;
    parameters.SetSecretKeyDist(secretKeyDist);
    parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
    parameters.SetNumLargeDigits(3);
    parameters.SetRingDim(1 << ringDim);
    parameters.SetBatchSize(num_slots);
    ScalingTechnique rescaleTech = FLEXIBLEAUTO; 
    parameters.SetScalingModSize(dcrtBits);
    parameters.SetFirstModSize(firstMod);
    parameters.SetScalingTechnique(rescaleTech);
    mult_depth = mlevelBootstrap;
    uint32_t levelsAvailableAfterBootstrap = mult_depth;

    circuit_depth = levelsAvailableAfterBootstrap + FHECKKSRNS::GetBootstrapDepth(level_budget, secretKeyDist);

    cout << "Context built, generating keys..." << endl;
    cout << endl << "dcrtBits: "<< dcrtBits << " -- firstMod: " << firstMod << endl << "Ciphertexts depth: " 
         << circuit_depth << ", available multiplications: " << levelsAvailableAfterBootstrap - 2 << endl;

    parameters.SetMultiplicativeDepth(circuit_depth);
    context = GenCryptoContext(parameters);

    context->Enable(PKE);
    context->Enable(KEYSWITCH);
    context->Enable(LEVELEDSHE);
    context->Enable(ADVANCEDSHE);
    context->Enable(FHE);

    keyPair = context->KeyGen();
    context->EvalMultKeyGen(keyPair.secretKey);
    context->EvalSumKeyGen(keyPair.secretKey);

    numSlots = num_slots;
    usint halfnumSlots = numSlots/2;
    cout << "numSlots: " << numSlots << "  - halfnumSlots: " << halfnumSlots << endl;
    context->EvalBootstrapSetup(level_budget, bsgsDim, numSlots);
    context->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);

    cout << " Keys Generated." << endl;
    ringDim = context->GetRingDimension();
    cout << " CKKS scheme is using ring dimension: " << ringDim  << endl;
    cout << " Ciphertexts depth: " << circuit_depth << endl; 
    cout << " Multiplication Depth: " << levelsAvailableAfterBootstrap - 2 << endl;
    cout << " Cyclotomic Order: " << context->GetCyclotomicOrder() << endl;
    cout << " -----------------------------------------------------------" << endl;

   if(serialize){
        write_to_file(keys_folder + "/mult_depth.txt", to_string(mult_depth));
        write_to_file(keys_folder + "/num_slots.txt", to_string(num_slots));
        write_to_file(keys_folder + "/level_budget.txt", to_string(level_budget[0]) + "," + to_string(level_budget[1]));
        keys_serialization();
    }
    context_loaded = false;
    return;
}


/**
 * @brief Serialize generated FHE keys (public, secret, mult, sum) to files.
 */

void FHEONHEController::keys_serialization(){
    
    cout << "------------------------------------------------------------" << endl;
    cout << "Now serializing keys ..." << endl;

    fs::path keysDir = keys_folder;
    if (!fs::exists(keysDir)) {
        if (!fs::create_directories(keysDir)) {
            std::cerr << "Failed to create directory: " << keys_folder << std::endl;
            return;
        }
    }

    if (!Serial::SerializeToFile(crypto_context_file, context, SerType::BINARY)) {
        cerr << "Error writing serialization of the crypto context to crypto-context.bin" << endl;
    } else {
        cout << "Crypto Context have been serialized" << std::endl;
    }

    ofstream multKeyFile(multi_key_file, ios::out | ios::binary);
    if (multKeyFile.is_open()) {
        if (!context->SerializeEvalMultKey(multKeyFile, SerType::BINARY)) {
            cerr << "Error writing eval mult keys" << std::endl;
            exit(1);
        }
        cout << "Relinearization Keys have been serialized" << std::endl;
        multKeyFile.close();
    }
    else {
        cerr << "Error serializing EvalMult keys in \"" << keys_folder + "/mult-keys.bin" << "\"" << endl;
        exit(1);
    }

    ofstream sumKeysFile(sum_key_file, ios::out | ios::binary);
    if (sumKeysFile.is_open()) {
        if (!context->SerializeEvalSumKey(sumKeysFile, SerType::BINARY)) {
            cerr << "Error writing sum keys" << std::endl;
            exit(1);
        }
        cout << "sum keys have been serialized" << std::endl;
    } else {
        cerr << "Error serializing sum keys \"" << keys_folder + "/sum-keys" << "\"" << std::endl;
        exit(1);
    }

    if (!Serial::SerializeToFile(public_key_file, keyPair.publicKey, SerType::BINARY)) {
        cerr << "Error writing serialization of public key to public-key.bin" << endl;
    } else {
        cout << "Public Key has been serialized" << std::endl;
    }

    if (!Serial::SerializeToFile(secret_key_file, keyPair.secretKey, SerType::BINARY)) {
        cerr << "Error writing serialization of public key to secret-key.bin" << endl;
    } else {
        cout << "Secret Key has been serialized" << std::endl;
    }
    return;
}


/**
 * @brief Load or generate FHE context based on configuration.
 *
 * @param config Configuration parameters for initialization.
 * @param loadContext If true, load context and keys from files; if false, generate new ones.
 */
void FHEONHEController::load_context(HEConfig config, bool loadContext){
    if (!config.keysFolder.empty()) {
        keys_folder = config.keysFolder;
    }
    update_key_paths();
    if (loadContext) {
        load_context();
    } else {
        generate_context(config);
    }
}


/**
 * @brief Load serialized FHE context and keys from default paths.
 */
void FHEONHEController::load_context() {
    
    context->ClearEvalMultKeys();
    context->ClearEvalAutomorphismKeys();
    CryptoContextFactory<lbcrypto::DCRTPoly>::ReleaseAllContexts();

    cout << "------------------------------------------------------------" << endl;
    cout << "Loading context from: " << keys_folder << endl;
    cout << "------------------------------------------------------------" << endl;

    if (utils::verbose_level >= 1) {
        cout << "------------------------------------------------------------" << endl;
        cout << "Reading serialized context..." << endl;
    }

    if (!Serial::DeserializeFromFile(crypto_context_file, context, SerType::BINARY)) {
        cerr << "I cannot read serialized data from: " << crypto_context_file << endl;
        exit(1);
    }

    PublicKey<DCRTPoly> clientPublicKey;
    if (!Serial::DeserializeFromFile(public_key_file, clientPublicKey, SerType::BINARY)) {
        cerr << "I cannot read serialized data from public-key.bin" << endl;
        exit(1);
    }

    PrivateKey<DCRTPoly> serverSecretKey;
    if (Serial::DeserializeFromFile(secret_key_file, serverSecretKey, SerType::BINARY)) {
        keyPair.secretKey = serverSecretKey;
    } else {
        if (utils::verbose_level >= 1) {
            cout << "Warning: Could not read secret-key.bin. Private key not loaded." << endl;
        }
    }

    keyPair.publicKey = clientPublicKey;

    std::ifstream multKeyIStream(multi_key_file, ios::in | ios::binary);
    if (!multKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " << multi_key_file << endl;
        exit(1);
    }
    if (!context->DeserializeEvalMultKey(multKeyIStream, SerType::BINARY)) {
        cerr << "Could not deserialize eval multkey file" << endl;
        exit(1);
    }

    ifstream sumKeyIStream(sum_key_file, ios::in | ios::binary);
    if (!sumKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " << sum_key_file << std::endl;
        exit(1);
    }
    if (!context->DeserializeEvalSumKey(sumKeyIStream, SerType::BINARY)) {
        cerr << "Could not deserialize eval rot key file" << std::endl;
        exit(1);
    }

    mult_depth = stoi(read_from_file(keys_folder + "/mult_depth.txt"));
    level_budget[0] = read_from_file(keys_folder + "/level_budget.txt").at(0) - '0';
    level_budget[1] = read_from_file(keys_folder + "/level_budget.txt").at(2) - '0';

    uint32_t approxBootstrapDepth = 4 + 4;  
    uint32_t levelsUsedBeforeBootstrap = mult_depth;
    circuit_depth = levelsUsedBeforeBootstrap + FHECKKSRNS::GetBootstrapDepth(approxBootstrapDepth, level_budget, SPARSE_TERNARY);

    num_slots = stoi(read_from_file(keys_folder + "/num_slots.txt"));
    if (utils::verbose_level >= 1) {
        cout << "Setting up bootstrapping for " << num_slots << " slots..." << endl;
    }
    context->EvalBootstrapSetup(level_budget, bsgsDim, num_slots);

    if (utils::verbose_level >= 1) {
        cout << "Circuit depth: " << circuit_depth << ", available multiplications: " << levelsUsedBeforeBootstrap - 2 << endl;
        cout << "Context Loaded" << endl;
        cout << "------------------------------------------------------------" << endl;
    }
    context_loaded = true;
}

/**
 * @brief Generate evaluation keys required for bootstrapping.
 *
 * @param bootstrap_slots Number of slots reserved for bootstrapping.
 * @param filename Base filename/directory path for serialization.
 * @param serialize If true, serialize and save keys to disk.
 */
void FHEONHEController::generate_bootstrapping_keys(int bootstrap_slots, string filename, bool serialize) {
    // Instead, (re-)run the bootstrap setup for the specific slot count used at this
    // layer boundary so the precomputations are valid when EvalBootstrap is called.
    uint32_t numSlots = context->GetRingDimension() / 2;
    // Setup bootstrap precomputations for the exact slot count
    context->EvalBootstrapSetup(level_budget, bsgsDim, numSlots);
    // Generate bootstrap keys for the secret key
    context->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);
    // Ensure relinearization keys exist (safe to call, OpenFHE will ignore duplicates)
    context->EvalMultKeyGen(keyPair.secretKey);
}


/**
 * @brief Generate evaluation keys for homomorphic rotation operations.
 *
 * @param rotations List of rotation step sizes to generate keys for.
 * @param filename Base filename/directory path for serialization.
 * @param serialize If true, serialize and save keys to disk.
 * @param sum_key If true, generate additional keys required for evaluation sum.
 */
void FHEONHEController::generate_rotation_keys(const vector<int> rotations, std::string filename, bool serialize, bool sum_key) {
    if (context_loaded) {
        load_rotation_keys(filename, false);
        return;
    }
    
    if (serialize && filename.size() == 0) {
        cout << "Filename cannot be empty when serializing rotation keys." << endl;
        return;
    }
  
    context->EvalRotateKeyGen(keyPair.secretKey, rotations);
    if (sum_key) {
        // Generate sum keys when requested (useful for fully-connected layers)
        context->EvalSumKeyGen(keyPair.secretKey);
    }
    
    if (serialize) {
        ofstream rotationKeyFile(keys_folder + rotation_prefix + filename, ios::out | ios::binary);
        if (rotationKeyFile.is_open()) {
            if (!context->SerializeEvalAutomorphismKey(rotationKeyFile, SerType::BINARY)) {
                cerr << "Error writing rotation keys" << std::endl;
                exit(1);
            }
            cout << "Rotation keys \"" << filename << "\" have been serialized" << std::endl;
        } else {
            cerr << "Error serializing Rotation keys" << keys_folder + rotation_prefix + filename << std::endl;
            exit(1);
        }
    }
}


/**
 * @brief Generate both bootstrapping and rotation keys.
 *
 * @param rotations List of rotation step sizes.
 * @param bootstrap_slots Number of slots for bootstrapping.
 * @param filename Base filename/directory path.
 * @param serialize If true, serialize keys to disk.
 * @param sum_key If true, generate evaluation sum keys.
 */
void FHEONHEController::generate_bootstrapping_and_rotation_keys(vector<int> rotations, int bootstrap_slots, const string& filename,  bool serialize, bool sum_key) {
    if (context_loaded) {
        // load_bootstrapping_and_rotation_keys(bootstrap_slots, filename, false);
        return;
    }

    generate_bootstrapping_keys(bootstrap_slots, filename, serialize);
    if(sum_key){
        context->EvalSumKeyGen(keyPair.secretKey);
    }
    generate_rotation_keys(rotations, filename, serialize, sum_key);
}


/**
 * @brief Load both bootstrapping and rotation evaluation keys from files.
 *
 * @param bootstrap_slots Number of slots for bootstrapping.
 * @param filename Base filename/path.
 * @param verbose If true, print loading progress.
 */
void FHEONHEController::load_bootstrapping_and_rotation_keys(int bootstrap_slots, const string& filename, bool verbose) {
    if (utils::verbose_level >= 1) cout << endl << "    Loading bootstrapping and rotations keys from " << filename << "..." << endl;

    // int numSlots =  1 << bootstrap_slots;
    // context->EvalBootstrapSetup(level_budget, bsgsDim, numSlots);

    context->ClearEvalMultKeys();
	context->ClearEvalAutomorphismKeys();

    if (utils::verbose_level >= 1)  cout << "    (1/4) Bootstrapping precomputations completed!" << endl;
    
    ifstream multKeyIStream(multi_key_file, ios::in | ios::binary);
    if (!multKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " << keys_folder+ "/" << mult_prefix << filename << std::endl;
        exit(1);
    }
    if (!context->DeserializeEvalMultKey(multKeyIStream, SerType::BINARY)) {
        cerr << "Could not deserialize eval rot key file" << std::endl;
        exit(1);
    }
    if (utils::verbose_level >= 1)  cout << "    (2/4) MultKey deserialized and loaded!" << endl;

    ifstream rotKeyIStream(keys_folder + rotation_prefix + filename, ios::in | ios::binary);
    if (!rotKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " << keys_folder+ "/" << rotation_prefix << filename << std::endl;
        exit(1);
    }
    if (!context->DeserializeEvalAutomorphismKey(rotKeyIStream, SerType::BINARY)) {
        cerr << "Could not deserialize eval rot key file" << std::endl;
        exit(1);
    }
    if (utils::verbose_level >= 1) cout << "    (3/4) Rotation keys deserialized and loaded!" << endl;
    if (utils::verbose_level >= 1) cout << "    (4/4) Loaded keys for " << filename  << endl;
}

/**
 * @brief Helper function to read evaluation keys from file.
 *
 * @param crypto_context Context to associate keys with.
 * @param rot_file Path to rotation keys file.
 * @return Configured crypto context.
 */
CryptoContext<DCRTPoly> FHEONHEController::read_evaluation_keys(CryptoContext<DCRTPoly> crypto_context, const string &rot_file) {

    // Clear any existing eval keys in the provided context to avoid conflicts
    crypto_context->ClearEvalMultKeys();
    crypto_context->ClearEvalAutomorphismKeys();
    CryptoContextFactory<lbcrypto::DCRTPoly>::ReleaseAllContexts();
    if (utils::verbose_level >= 1) cout << "    (1/4) Clear previous Keys and context!" << endl;

    // Open files in READ mode
    ifstream multkey_file(multi_key_file, ios::in | ios::binary);
    if (!multkey_file.is_open()) {
        throw std::runtime_error("Failed to open mult key file: " + multi_key_file);
    }
    if (!crypto_context->DeserializeEvalMultKey(multkey_file, SerType::BINARY)) {
        throw std::runtime_error("Failed to deserialize relinearization key from " + multi_key_file);
    }
    if (utils::verbose_level >= 1) cout << "    (2/4) MultKey deserialized and loaded!" << endl;

    ifstream rotkey_file(keys_folder + rotation_prefix + rot_file, ios::in | ios::binary);
    if (!rotkey_file.is_open()) {
        throw std::runtime_error("Failed to open rotation key file: " + keys_folder + rotation_prefix + rot_file);
    }
    if (!crypto_context->DeserializeEvalAutomorphismKey(rotkey_file, SerType::BINARY)) {
        throw std::runtime_error("Failed to deserialize rotation keys from " + keys_folder + rotation_prefix + rot_file);
    }
    if (utils::verbose_level >= 1) cout << "    (3/4) Rotation keys deserialized and loaded!" << endl;
    if (utils::verbose_level >= 1) cout << "    (4/4) Loaded context for " << rot_file  << endl;

    return crypto_context;
}


/**
 * @brief Load rotation evaluation keys from file.
 *
 * @param filename Path to rotation keys file.
 * @param verbose If true, print loading progress.
 */
void FHEONHEController::load_rotation_keys(const string& filename, bool verbose) {

    if (utils::verbose_level >= 1) cout << endl << "Loading rotations keys from " << filename << "..." << endl;
    
    ifstream rotKeyIStream(keys_folder + rotation_prefix + filename, ios::in | ios::binary);
    if (!rotKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " <<keys_folder + "/" << rotation_prefix << filename << std::endl;
        exit(1);
    }
    if (!context->DeserializeEvalAutomorphismKey(rotKeyIStream, SerType::BINARY)) {
        cerr << "Could not deserialize eval rot key file" << std::endl;
        exit(1);
    }

    if (utils::verbose_level >= 1) {
        cout << "(1/1) Rotation keys read!" << endl;
        cout << endl;
    }
}

/**
 * @brief Clear rotation evaluation keys from current context.
 */
void FHEONHEController::clear_rotation_keys() {
    context->ClearEvalMultKeys();
    context->ClearEvalAutomorphismKeys();
    CryptoContextFactory<lbcrypto::DCRTPoly>::ReleaseAllContexts();
}

/**
 * @brief Clear bootstrapping and rotation keys from current context.
 *
 * @param bootstrap_num_slots Number of bootstrapping slots.
 */
void FHEONHEController::clear_bootstrapping_and_rotation_keys(int bootstrap_num_slots) {
    //This lines would free more or less 1GB or precomputations, but requires access to the GetFHE function
    // FHECKKSRNS* derivedPtr = dynamic_cast<FHECKKSRNS*>(context->GetScheme()->GetFHE().get());
    // derivedPtr->m_bootPrecomMap.erase(bootstrap_num_slots);

    context->ClearEvalMultKeys();
    context->ClearEvalAutomorphismKeys();
    CryptoContextFactory<lbcrypto::DCRTPoly>::ReleaseAllContexts();
}


/**
 * @brief Clear keys and release crypto context.
 *
 * @param bootstrapping_key_slots Number of bootstrapping slots.
 */
void FHEONHEController::clear_context(int bootstrapping_key_slots) {
    
    if (bootstrapping_key_slots != 0)
        clear_bootstrapping_and_rotation_keys(bootstrapping_key_slots);
    else
        clear_rotation_keys();
}


/**
 * @brief Perform homomorphic bootstrapping (refreshing noise level) on ciphertext.
 *
 * @param encryptedInput Ciphertext to bootstrap.
 * @param encode_level Level budget for encoding.
 * @return Refreshed ciphertext.
 */
Ctext FHEONHEController::bootstrap_function(Ctext& encryptedInput, int encode_level){
    num_bootsraps++;
    auto start_bootstrap = startTime();
    Ctext boots_ciphertext = context->EvalBootstrap(encryptedInput, encode_level);
    printBootstrapTiming("    Bootstrapping ciphertext... " + to_string(num_bootsraps), start_bootstrap);
    return boots_ciphertext;
}

/**
 * @brief Perform parallel bootstrapping on a batch of ciphertexts.
 *
 * @param encryptedInputs Vector of ciphertexts to bootstrap.
 * @param inputChannels Number of channels/ciphertexts.
 * @param encode_level Level budget.
 * @return Vector of refreshed ciphertexts.
 */

vector<Ctext> FHEONHEController::batch_bootstrap_function(vector<Ctext>& encryptedInputs, int inputChannels, int encode_level) {
    auto start_bootstrap = startTime();
    vector<Ctext> batch_ciphertexts(inputChannels);
    int numThreads = min(inputChannels, (int)thread::hardware_concurrency());
    vector<thread> threads(numThreads);

    // Worker function for a range of channels
    auto worker = [&](int start, int end) {
        for (int b = start; b < end; b++) {
            batch_ciphertexts[b] = context->EvalBootstrap(encryptedInputs[b], encode_level);
        }
    };

    // Divide channels evenly among threads
    int block = (inputChannels + numThreads - 1) / numThreads;
    for (int t = 0; t < numThreads; t++) {
        int start = t * block;
        int end = min(start + block, inputChannels);
        threads[t] = thread(worker, start, end);
    }
    for (auto &th : threads) th.join();

    printBootstrapTiming("  Batch Bootstrapping " + to_string(inputChannels) + " ciphertexts...", start_bootstrap);
    return batch_ciphertexts;
}

/**
 * @brief Encrypt a vector of double values into a packed CKKS ciphertext.
 *
 * @param inputData Data vector to be homomorphically encrypted.
 * @param isTimeMeasurement If true, initialize timing statistics for the image run.
 * @return homomorphically encrypted ciphertext.
 */
Ctext FHEONHEController::encrypt_input(vector<double>& inputData, bool isTimeMeasurement) {
   
    Ptext plaintext = context->MakeCKKSPackedPlaintext(inputData, 1, 1);
    plaintext->SetLength(inputData.size());
    auto encryptImage = context->Encrypt(keyPair.publicKey, plaintext);

     if(isTimeMeasurement) {
        image_start_time = std::chrono::steady_clock::now();
        program_start_time = std::chrono::steady_clock::now();
        image_circuit_time_secs = 0;
        image_bootstrap_time_ms = 0.0;
        num_bootsraps = 0;
        operation_count = 0;
        accumulated_circuit_time_ms = 0.0;
    }
    return encryptImage;
}


/**
 * @brief Encode a double vector into packed CKKS plaintext.
 *
 * @param inputData Data vector to encode.
 * @param encode_level Level budget.
 * @return Encoded plaintext.
 */
Ptext FHEONHEController::encode_input(vector<double>& inputData, int encode_level) {
    Ptext plaintext = context->MakeCKKSPackedPlaintext(inputData, 1, encode_level);
    return plaintext;
}

/**
 * @brief Encode a double vector with explicit slot count.
 *
 * @param inputData Data vector to encode.
 * @param num_slots Number of slots to use.
 * @param encode_level Level budget.
 * @return Encoded plaintext.
 */

Ptext FHEONHEController::encode_input(vector<double>& inputData, int num_slots, int encode_level) {
    int numElements = nextPowerOf2(num_slots);
    Ptext plaintext = context->MakeCKKSPackedPlaintext(inputData, 1, encode_level, nullptr, numElements);
    return plaintext;
}

/**
 * @brief Encode kernel weights for ResNet shortcut connections.
 *
 * @param inputData Flattened weight vector.
 * @param cols_square Square dimension of output features.
 * @return Encoded plaintext.
 */
Ptext FHEONHEController::encode_shortcut_kernel(vector<double>& inputData, int cols_square) {
    int dim1 = inputData.size();
    vector<double> main_kernel;
    for(int t =0; t< dim1; t++){
        double cell_value = inputData[t];
        vector<double> repeated(cols_square, cell_value);
        main_kernel.insert(main_kernel.end(), repeated.begin(), repeated.end());
    }
    Ptext plaintext = context->MakeCKKSPackedPlaintext(main_kernel, 1, 1);
    return plaintext;
}

/**
 * @brief Encode bias inputs for convolution or linear layers.
 *
 * @param inputData Bias values vector.
 * @param cols_square Square dimension of output features.
 * @param encode_level Level budget.
 * @return Encoded plaintext.
 */
Ptext FHEONHEController::encode_bais_input(vector<double>& inputData, int cols_square, int encode_level) {
    int dim1 = inputData.size();
    vector<double> main_kernel;
    for(int t =0; t< dim1; t++){
        double cell_value = inputData[t];
        vector<double> repeated(cols_square, cell_value);
        main_kernel.insert(main_kernel.end(), repeated.begin(), repeated.end());
    }

    Ptext plaintext = context->MakeCKKSPackedPlaintext(main_kernel, 1, encode_level);
    return plaintext;
}

/**
 * @brief Re-encrypt plaintext data to ciphertext.
 *
 * @param plaintextData Plaintext to encrypt.
 * @return Encrypted ciphertext.
 */
Ctext FHEONHEController::reencrypt_data(Ptext plaintextData) {
    
    auto encryptedData = context->Encrypt(keyPair.publicKey, plaintextData);
    return encryptedData;
}

/**
 * @brief Decrypt ciphertext into packed plaintext values.
 *
 * @param encryptedinputData Ciphertext to decrypt.
 * @param cols Number of slots to retrieve.
 * @return Decrypted plaintext.
 */
Ptext FHEONHEController::decrypt_data(Ctext encryptedinputData, int cols) {
    
    Ptext plaintextDec;
    context->Decrypt(keyPair.secretKey, encryptedinputData, &plaintextDec);
    plaintextDec->SetLength(cols);
    return plaintextDec;
}

/**
 * @brief Encrypt a 3D weight kernel matrix.
 *
 * @param kernelData 3D raw weights vector.
 * @param cols_square Square dimension of features.
 * @return Encrypted weights matrix.
 */
vector<vector<Ctext>> FHEONHEController::encrypt_kernel(vector<vector<vector<double>>>& kernelData, int cols_square){
    size_t dim1 = kernelData.size();
    if (dim1 == 0) return {};
    size_t dim2 = kernelData[0].size();
    if (dim2 == 0) return {};
    size_t dim3 = kernelData[0][0].size();
    if (dim3 == 0) return {}; 

    vector<vector<Ctext>> encrypt_kernel; 
    for (size_t k=0; k< dim1; k++){
        vector<Ctext> filters;
        for (size_t i=0; i< dim2; i++){
            for (size_t j=0; j< dim3 ; j++){
                double cell_value  = kernelData[k][i][j];
                vector<double> repeated(cols_square, cell_value);
                Ctext encrypted_val = encrypt_input(repeated);
                filters.push_back(encrypted_val);
            }
        }
        encrypt_kernel.push_back(filters);
    }
    return encrypt_kernel;
}

/**
 * @brief Encode a 1D fully connected weight vector.
 *
 * @param kernelData Weight values vector.
 * @param cols_square Square dimension.
 * @return Vector of encoded plaintexts.
 */
vector<Ptext> FHEONHEController::encode_kernel(vector<double>& kernelData, int cols_square){
    size_t dim1 = kernelData.size();
    if (dim1 == 0) return {};

    vector<Ptext> encrypt_kernel; 
     for (size_t j=0; j< dim1 ; j++){
        double cell_value  = kernelData[j];
        vector<double> repeated(cols_square, cell_value);
        Ptext encrypted_val = encode_input(repeated);
        encrypt_kernel.push_back(encrypted_val);
    }
    return encrypt_kernel;
}

/**
 * @brief Encode a 3D kernel weight matrix.
 *
 * @param kernelData 3D raw weights vector.
 * @param cols_square Square dimension.
 * @return Vector of encoded plaintexts.
 */
vector<Ptext> FHEONHEController::encode_kernel(vector<vector<vector<double>>>& kernelData, int cols_square){
    size_t dim1 = kernelData.size();
    if (dim1 == 0) return {};
    size_t dim2 = kernelData[0].size();
    if (dim2 == 0) return {};
    size_t dim3 = kernelData[0][0].size();
    if (dim3 == 0) return {}; 
    // cout <<"input kernel shape: " << dim1 << "*" << dim2 << "*" << dim3 <<endl;

    int kernelWidth_sq = pow(dim2, 2);
    vector<vector<double>> main_kernel(kernelWidth_sq, vector<double>());
    for (size_t k=0; k< dim1; k++){
        vector<vector<double>> filters;
        for (size_t i=0; i< dim2; i++){
            for (size_t j=0; j< dim3 ; j++){
                double cell_value  = kernelData[k][i][j];
                vector<double> repeated(cols_square, cell_value);
                filters.push_back(repeated);
            }
        }
        for(int t =0; t< kernelWidth_sq; t++){   
            main_kernel[t].insert(main_kernel[t].end(), filters[t].begin(), filters[t].end());
        }
    }
    vector<Ptext> encoded_kernel;
    for( int s =0; s< kernelWidth_sq; s++){
        // cout << "Kernel size: " << main_kernel[s].size() << endl;
        Ptext encoded_val = encode_input(main_kernel[s]);
        encoded_kernel.push_back(encoded_val);
    }
    return encoded_kernel;
}

/**
 * @brief Dynamically change the number of active slots in a ciphertext.
 *
 * Useful for reducing ring dimension/size in pooled layers.
 *
 * @param encryptedInput The ciphertext to modify.
 * @param num_slots The new number of slots.
 * @return Modifed ciphertext with changed slots.
 */

Ctext FHEONHEController::change_num_slots(Ctext& encryptedInput, uint32_t num_slots){
    encryptedInput->SetSlots(1 << num_slots);
    return encryptedInput;
}

/**
 * @brief Encode a 3D weight kernel using optimized masking for 3x3 convolution.
 *
 * @param kernelData 3D raw weights vector.
 * @param cols_square Square dimension.
 * @param encode_level Level budget.
 * @return Vector of encoded plaintexts.
 */
vector<Ptext> FHEONHEController::encode_kernel_optimized(vector<vector<vector<double>>>& kernelData, int cols_square, int encode_level) {
    size_t dim1 = kernelData.size();
    if (dim1 == 0) return {};
    size_t dim2 = kernelData[0].size();
    if (dim2 == 0) return {};
    size_t dim3 = kernelData[0][0].size();
    if (dim3 == 0) return {};

    int kernelWidth_sq = pow(dim2, 2);
    vector<vector<double>> main_kernel(kernelWidth_sq, vector<double>());
    for (size_t k=0; k< dim1; k++){
        vector<vector<double>> filters;
        for (size_t i=0; i< dim2; i++){
            for (size_t j=0; j< dim3 ; j++){
                double cell_value  = kernelData[k][i][j];
                // if(cell_value == 0)
                //     cell_value = 1e-40;
                vector<double> repeated(cols_square, cell_value);
                filters.push_back(repeated);
            }
        }
        for(int t =0; t< kernelWidth_sq; t++){   
            main_kernel[t].insert(main_kernel[t].end(), filters[t].begin(), filters[t].end());
        }
    }

    int vector_width = sqrt(cols_square);
    vector<vector<double>> bin_masks = {
        build_tiled_mask(vector_width + 1, 0, vector_width - 1, cols_square, dim1),
        build_tiled_mask(vector_width, 0, cols_square, cols_square, dim1),
        build_tiled_mask(vector_width, 0, vector_width - 1, cols_square, dim1),
        build_tiled_mask(1, 0, vector_width - 1, cols_square, dim1),
        build_tiled_mask(0, 0, cols_square, cols_square, dim1),
        build_tiled_mask(0, 1, vector_width - 1, cols_square, dim1),
        build_tiled_mask(1, vector_width - 1, vector_width - 1, cols_square, dim1),
        build_tiled_mask(0, vector_width, cols_square, cols_square, dim1),
        build_tiled_mask(0, vector_width + 1, vector_width - 1, cols_square, dim1)
    };
        
    vector<Ptext> encoded_kernel;
    for (int s = 0; s < kernelWidth_sq; ++s) {
        if (s >= static_cast<int>(bin_masks.size())) {
            std::cerr << "Error: bin_mask index out of range!" << std::endl;
            return encoded_kernel;
        }

        // Multiply main_kernel[s] element-wise with bin_masks[s]
        
        vector<double> cleaned_kernel(main_kernel[s].size());
        for (size_t i = 0; i < main_kernel[s].size(); ++i) {
            cleaned_kernel[i] = main_kernel[s][i] * bin_masks[s][i];
        }

        // Encode the cleaned kernel
        int numElements = nextPowerOf2(main_kernel[s].size());
        Ptext encoded_val = encode_input(cleaned_kernel, numElements, encode_level);
        encoded_kernel.push_back(encoded_val);
    }

    return encoded_kernel;
}

/**
 * @brief Decrypt and read the predicted label (argmax) from encrypted output.
 *
 * Prints prediction info based on verbosity levels. Optionally appends result to predictions file.
 *
 * @param inferencedData Ciphertext of inference results.
 * @param num_slots Number of slots to decode.
 * @param predictions_file_path File path to write the predicted label to.
 * @return 0 on success.
 */
int FHEONHEController::read_inferenced_label(Ctext inferencedData, int num_slots,  const string& predictions_file_path){
    auto decryptedValue = decrypt_data(inferencedData, num_slots);
    auto decryptedVector = decryptedValue->GetRealPackedValue();
    auto maxElementIt = max_element(decryptedVector.begin(), decryptedVector.end());
    int maxIndex = distance(decryptedVector.begin(), maxElementIt);
    
    if (verbose_level == level_zero) {
        cout << "Predicted Value : " << maxIndex << endl;
    } else if (verbose_level == level_one) {
        cout << "Predicted Value : " << maxIndex << " Weight:  " << decryptedVector[maxIndex] << endl;
    } else if (verbose_level >= level_two) {
        cout << "Predicted Value : " << maxIndex << " Weight:  " << decryptedVector[maxIndex] << endl;
        cout << "Weights for the entire vector: " << decryptedVector << endl;
    }

    double total_bts_secs = image_bootstrap_time_ms / 1000.0;
    double image_crt_time_secs = accumulated_circuit_time_ms / 1000.0;
    image_runtime_secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - image_start_time).count();
    cout << "Total Bootstrapping Time: " << total_bts_secs << "s" << endl;
    cout << "Total Circuit Time: " << image_crt_time_secs << "s" << endl;
    cout << "Total Runtime: " << image_runtime_secs << "s" << endl;

    if (!predictions_file_path.empty()) {
        ofstream outFile(predictions_file_path, ios_base::app);
        if (outFile.is_open()) {
            outFile << maxIndex << endl;
        }
    }
    return 0;
}

/**
 * @brief Decrypt and print minimum and maximum values of ciphertext slots.
 *
 * @param inferencedData Ciphertext to analyze.
 * @param num_slots Number of slots to decrypt.
 * @return 0 on success.
 */
int FHEONHEController::read_minmax(Ctext inferencedData, int num_slots) {
    auto decryptedValue = decrypt_data(inferencedData, num_slots);
    auto decryptedVector = decryptedValue->GetRealPackedValue();

    // cout << "Decrypted Vector " << decryptedVector << endl;
    auto maxElementIt = max_element(decryptedVector.begin(), decryptedVector.end());
    int maxIndex = distance(decryptedVector.begin(), maxElementIt);
    auto minElementIt = min_element(decryptedVector.begin(), decryptedVector.end());
    int minIndex = distance(decryptedVector.begin(), minElementIt);
    cout << "------------------------------------------------------------------ " << endl;
    cout << "Range [ " << decryptedVector[minIndex] << " , " << decryptedVector[maxIndex] <<" ]" << endl;
    cout << "Index: " << maxIndex << endl;
    cout << "------------------------------------------------------------------ " << endl;
    return 0;
}

/**
 * @brief Decrypt and read scale value (max absolute ceiling) for ReLU activation.
 *
 * @param inferencedData Ciphertext to analyze.
 * @param num_slots Number of slots to decrypt.
 * @return Max absolute value rounded up.
 */
int FHEONHEController::read_scaling_value(Ctext inferencedData, int num_slots){
    auto decryptedValue = decrypt_data(inferencedData, num_slots);
    auto decryptedVector = decryptedValue->GetRealPackedValue();

    double maxAbsValue = *std::max_element(decryptedVector.begin(), decryptedVector.end(), [](int a, int b) {
        return std::abs(a) < std::abs(b);
    });
    int roundedMaxAbsValue = static_cast<int>(std::ceil(std::abs(maxAbsValue)));
    return roundedMaxAbsValue;
}

/**
 * @brief Generate a tiled binary mask for convolution operations.
 *
 * @param starting_padding Zeros at the start.
 * @param ending_padding Zeros at the end.
 * @param window_length Length of the convolution window of 1s.
 * @param max_length Total slots per channel.
 * @param tile_count Number of channels.
 * @return Tiled mask data.
 */
vector<double> FHEONHEController::build_tiled_mask(int starting_padding, int ending_padding, 
    int window_length, int max_length, int tile_count) {
   
    vector<double> mask;

    // Add starting padding
    for (int i = 0; i < starting_padding; ++i) {
        mask.push_back(0.0);
    }

    // Add windows of 1s and a trailing 0
    while (mask.size() < static_cast<size_t>(max_length - ending_padding)) {
        for (int j = 0; j < window_length; ++j) {
            mask.push_back(1.0);
        }
        mask.push_back(0.0);
    }

    // Trim or pad the mask to match max_length
    while (mask.size() > static_cast<size_t>(max_length)) {
        mask.pop_back();
    }
    while (mask.size() < static_cast<size_t>(max_length)) {
        mask.push_back(0.0);
    }

    // Add ending padding
    for (int i = 0; i < ending_padding; ++i) {
        mask[max_length - i - 1] = 0.0;
    }

    // Tile the mask
    std::vector<double> tiled_mask;
    for (int i = 0; i < tile_count; ++i) {
        tiled_mask.insert(tiled_mask.end(), mask.begin(), mask.end());
    }
    return tiled_mask;
}


/**
 * @brief Decrypt and read predicted labels for a batch of inferences.
 *
 * Extracts the argmax class index for each sample in the packed batch.
 *
 * @param inferencedData Ciphertext containing packed batch results.
 * @param batchSize Number of samples in the batch.
 * @param numClasses Number of output classes.
 * @param predictions_file_path Path to write the predicted labels.
 * @param baseIndex Index offset for batch tracking.
 * @return Vector of predicted class labels.
 */
vector<int> FHEONHEController::read_batch_inferenced_label(Ctext inferencedData,  int batchSize, int numClasses,  const string& predictions_file_path, int baseIndex){
     int totalElements = batchSize * numClasses;

    // Decrypt
    auto decryptedVal = decrypt_data(inferencedData, totalElements);
    auto decryptedVec = decryptedVal->GetRealPackedValue();
    if (verbose_level >= level_two) {
        cout << "DecryptedVec: " << decryptedVec << endl;
    }

    vector<int> predictedLabels(batchSize);

    // Process each batch separately
    for (int b = 0; b < batchSize; b++) {
        int startIdx = b * numClasses;
        int endIdx   = startIdx + numClasses;

        // Find max in this batch slice
        
        auto maxIt = max_element(
            decryptedVec.begin() + startIdx, 
            decryptedVec.begin() + endIdx
        );
        int label = distance(decryptedVec.begin() + startIdx, maxIt);
        
        predictedLabels[b] = label;

        if (verbose_level == level_zero) {
            cout << "Batch " << baseIndex+b << " -> Predicted Label: " << label << endl;
        } else if (verbose_level == level_one) {
            cout << "Batch " << baseIndex+b << " -> Predicted Label: " << label << " (Score: " << *maxIt << ")" << endl;
        } else if (verbose_level >= level_two) {
            cout << "Batch " << baseIndex+b << " -> Predicted Label: " << label << " (Score: " << *maxIt << ")" << endl;
            cout << "Weights for Batch " << baseIndex+b << ": [ ";
            for (int i = startIdx; i < endIdx; i++) {
                cout << decryptedVec[i] << " ";
            }
            cout << "]" << endl;
        }

        if (!predictions_file_path.empty()) {
            ofstream outFile(predictions_file_path, ios_base::app);
            if (outFile.is_open()) {
                outFile << label << endl;
            }
        }
    }

    double total_bts_secs = image_bootstrap_time_ms / 1000.0;
    double image_crt_time_secs = accumulated_circuit_time_ms / 1000.0;
    image_runtime_secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - image_start_time).count();
    cout << "Total Bootstrapping Time: " << total_bts_secs << "s" << endl;
    cout << "Total Circuit Time: " << image_crt_time_secs << "s" << endl;
    cout << "Total Runtime: " << image_runtime_secs << "s" << endl;

    return predictedLabels;
}


/**
 * @brief Decrypt and read predicted labels from multiple independent ciphertexts.
 *
 * Each sample is represented by a separate ciphertext.
 *
 * @param inferencedData Vector of ciphertexts, one per batch sample.
 * @param batchSize Number of samples in the batch.
 * @param numClasses Number of output classes.
 * @param predictions_file_path Path to write the predicted labels.
 * @param baseIndex Index offset for batch tracking.
 * @return Vector of predicted class labels.
 */
vector<int> FHEONHEController::read_batch_inferenced_label_multiple_outputs(vector<Ctext> inferencedData,  int batchSize, int numClasses,  const string& predictions_file_path, int baseIndex){
    
    vector<int> predictedLabels(batchSize);
    // Process each batch separately
    for (int b = 0; b < batchSize; b++) {
        auto decryptedVal = decrypt_data(inferencedData[b], numClasses);
        auto decryptedVec = decryptedVal->GetRealPackedValue();
        int startIdx = 0;
        int endIdx   = startIdx + numClasses;

        // Find max in this batch slice
        auto maxIt = max_element(
            decryptedVec.begin() + startIdx, 
            decryptedVec.begin() + endIdx
        );
        int label = distance(decryptedVec.begin() + startIdx, maxIt);
        predictedLabels[b] = label;

        if (verbose_level == level_zero) {
            cout << "Batch " << baseIndex+b << " -> Predicted Label: " << label << endl;
        } else if (verbose_level == level_one) {
            cout << "Batch " << baseIndex+b << " -> Predicted Label: " << label << " (Score: " << *maxIt << ")" << endl;
        } else if (verbose_level >= level_two) {
            cout << "Batch " << baseIndex+b << " -> Predicted Label: " << label << " (Score: " << *maxIt << ")" << endl;
            cout << "Weights for Batch " << baseIndex+b << ": " << decryptedVec << endl;
        }

        if (!predictions_file_path.empty()) {
            ofstream outFile(predictions_file_path, ios_base::app);
            if (outFile.is_open()) {
                outFile << label << endl;
            }
        }
    }

    double total_bts_secs = image_bootstrap_time_ms / 1000.0;
    double image_crt_time_secs = accumulated_circuit_time_ms / 1000.0;
    image_runtime_secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - image_start_time).count();
    cout << "Total Bootstrapping Time: " << total_bts_secs << "s" << endl;
    cout << "Total Circuit Time: " << image_crt_time_secs << "s" << endl;
    cout << "Total Runtime: " << image_runtime_secs << "s" << endl;

    return predictedLabels;
}


/**
 * @brief Decrypt and read scaling values for batch ReLU operations.
 *
 * @param encryptedInputs Vector of ciphertexts.
 * @param inputChannels Number of channels.
 * @param num_slots Number of slots per channel.
 * @return Vector of scaling factors.
 */
vector<int> FHEONHEController::read_batch_scaling_values(vector<Ctext>& encryptedInputs, int inputChannels, int num_slots){
    vector<int> maxVec;
    for(int i=0; i<inputChannels; i++){
        auto decryptedVal = decrypt_data(encryptedInputs[i], num_slots);
        auto decryptedVec = decryptedVal->GetRealPackedValue();
        double maxAbsVal = *std::max_element(decryptedVec.begin(), decryptedVec.end(), [](double a, double b) {
            return std::abs(a) < std::abs(b);
        });
        int roundedMax = static_cast<int>(std::ceil(std::abs(maxAbsVal)));
        maxVec.push_back(roundedMax);
    }
    return maxVec;
}

/**
 * @brief Decrypt and print min/max values for batch ciphertexts.
 *
 * @param encryptedInputs Vector of ciphertexts.
 * @param inputChannels Number of channels.
 * @param num_slots Number of slots per channel.
 * @return 0 on success.
 */
int FHEONHEController::read_batch_minmax(vector<Ctext>& encryptedInputs, int inputChannels,  int num_slots) {
    
    vector<int> maxVec;
    for(int i=0; i<inputChannels; i++){
        auto decryptedVal = decrypt_data(encryptedInputs[i], num_slots);
        auto decryptedVec = decryptedVal->GetRealPackedValue();

        auto maxElementIt = max_element(decryptedVec.begin(), decryptedVec.end());
        int maxIndex = distance(decryptedVec.begin(), maxElementIt);
        auto minElementIt = min_element(decryptedVec.begin(), decryptedVec.end());
        int minIndex = distance(decryptedVec.begin(), minElementIt);
        cout << "------------------------------------------------------------------ " << endl;
        cout << "Range [ " << decryptedVec[minIndex] << " , " << decryptedVec[maxIndex] <<" ]" << endl;
        cout << "Index: " << maxIndex << endl;
        cout << "------------------------------------------------------------------ " << endl;
    }
    return 0;
}

/**
 * @brief Decrypt and print multiple ciphertexts representing channels or batches.
 *
 * @param encryptedInputs Vector of ciphertexts to decrypt.
 * @param inputChannels Number of ciphertexts/channels.
 * @param num_slots Number of slots to decode per ciphertext.
 * @return 0 on success.
 */
int FHEONHEController::decrypt_batch_data(vector<Ctext>& encryptedInputs, int inputChannels,  int num_slots) {
    
    vector<int> maxVec;
    for(int i=0; i<inputChannels; i++){
        auto decryptedVal = decrypt_data(encryptedInputs[i], num_slots);
        auto decryptedVec = decryptedVal->GetRealPackedValue();
        // double val = accumulate(decryptedVec.begin(), decryptedVec.end(), 0.0);
        cout << "------------------------------------------------------------------ " << endl;
        cout << decryptedVec << endl;
        cout << "------------------------------------------------------------------ " << endl;
        // cout << "Accumulated: " << val << endl; 
    }
    return 0;
}


/**
 * @brief Decrypt and print complex packed values of a ciphertext.
 *
 * @param encryptedpackedVector Ciphertext to print.
 * @param num_slots Number of slots to print.
 * @return Vector of real decrypted values.
 */
vector<double> FHEONHEController::decrypt_and_print(Ctext encryptedpackedVector, int num_slots) {
    
    Ptext plaintextDec;
    context->Decrypt(keyPair.secretKey, encryptedpackedVector, &plaintextDec);
    plaintextDec->SetLength(num_slots);
    vector<complex<double>> finalResult = plaintextDec->GetCKKSPackedValue();
    cout << finalResult << endl;
    vector<double> realResult;
    // add only the number of slots specified by num_slots
    for (int i = 0; i < num_slots; i++) {
        realResult.push_back(finalResult[i].real());
    }
    cout << endl;
    return realResult;
}

/**
 * @brief Decrypt ciphertext using an explicit private key.
 *
 * @param sk Explicit private key for decryption.
 * @param encryptedinputData Ciphertext to decrypt.
 * @param cols Number of slots to retrieve.
 * @return Decrypted plaintext object.
 */
Ptext FHEONHEController::decrypt_data_with_key(PrivateKey<DCRTPoly> &sk,
                                               Ctext encryptedinputData,
                                               int cols) {

    Ptext plaintextDec;
    context->Decrypt(sk, encryptedinputData, &plaintextDec);
    plaintextDec->SetLength(cols);
    return plaintextDec;
}

/**
 * @brief Decrypt and read scaling value using an explicit private key.
 *
 * @param sk Explicit private key.
 * @param inferencedData Ciphertext to analyze.
 * @param num_slots Number of slots.
 * @return Max absolute value rounded up.
 */
int FHEONHEController::read_scaling_value_with_key(PrivateKey<DCRTPoly> &sk,
                                                   Ctext inferencedData,
                                                   int num_slots) {
    // int roundedMaxAbsValue = 10; // Temporary hardcoded value for testing
    auto decryptedValue = decrypt_data_with_key(sk, inferencedData, num_slots);
    auto decryptedVector = decryptedValue->GetRealPackedValue();

    cout << endl
        << "--------------------------------------------------- " << endl
        << endl;
    cout << "Decrypted Vector for Scaling Value: " << decryptedVector << endl;
    cout << endl
        << "--------------------------------------------------- " << endl;

    double maxAbsValue =
        *std::max_element(decryptedVector.begin(), decryptedVector.end(),
                            [](int a, int b) { return std::abs(a) < std::abs(b); });
    int roundedMaxAbsValue = static_cast<int>(std::ceil(std::abs(maxAbsValue)));
    return roundedMaxAbsValue;
}


/**
 * @brief Decrypt and read prediction index using an explicit private key.
 *
 * @param sk Explicit private key for decryption.
 * @param inferencedData Ciphertext of inference results.
 * @param num_slots Number of slots to decode.
 * @param predictions_file_path File path to write the predicted label to.
 * @return Index corresponding to the predicted label.
 */
int FHEONHEController::read_inferenced_label_with_key(PrivateKey<DCRTPoly> &sk,
                                                      Ctext inferencedData,
                                                      int num_slots,
                                                      const string& predictions_file_path) {
    auto decryptedValue = decrypt_data_with_key(sk, inferencedData, num_slots);
    auto decryptedVector = decryptedValue->GetRealPackedValue();
    auto maxElementIt =
        max_element(decryptedVector.begin(), decryptedVector.end());
    int maxIndex = distance(decryptedVector.begin(), maxElementIt);

    if (verbose_level == level_zero) {
        cout << "Predicted Value : " << maxIndex << endl;
    } else if (verbose_level == level_one) {
        cout << "Predicted Value : " << maxIndex << " Weight:  " << decryptedVector[maxIndex] << endl;
    } else if (verbose_level >= level_two) {
        cout << "Predicted Value : " << maxIndex << " Weight:  " << decryptedVector[maxIndex] << endl;
        cout << "Decrypted Vector: " << decryptedVector << endl;
    }

    double total_bts_secs = image_bootstrap_time_ms / 1000.0;
    double total_run_secs = duration_cast<milliseconds>(steady_clock::now() - image_start_time).count() / 1000.0;
    cout << "Total Bootstrapping Time: " << total_bts_secs << "s" << endl;
    cout << "Total Circuit Time: " << image_circuit_time_secs << "s" << endl;
    cout << "Total Runtime: " << total_run_secs << "s" << endl;

    if (!predictions_file_path.empty()) {
        ofstream outFile(predictions_file_path, ios_base::app);
        if (outFile.is_open()) {
            outFile << maxIndex << endl;
        }
    }
    return maxIndex;
}

/**
 * @brief Clear automorphism keys from the given crypto context.
 *
 * @param crypto_context Context to clear keys from.
 */
void FHEONHEController::harness_clear_bootstrapping_and_rotation_keys(CryptoContext<DCRTPoly> &crypto_context) {
  // crypto_context->ClearEvalMultKeys();
  crypto_context->ClearEvalAutomorphismKeys();
  // CryptoContextFactory<lbcrypto::DCRTPoly>::ReleaseAllContexts();
}

