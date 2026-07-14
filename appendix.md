# Artifact Appendix (FHEON — Single-Input Models)

Paper title: **FHEON: A Configurable Framework for Developing Privacy-Preserving Encrypted Neural Networks**

Requested Badge(s):
- [x] **Available**
- [x] **Functional**
- [x] **Reproduced**

## Description

FHEON is a configurable framework for building **privacy-preserving convolutional neural networks (CNNs)** using **homomorphic encryption (HE)**. 
The framework enables neural network inference to be executed entirely over encrypted data, ensuring that sensitive inputs remain protected throughout computation. 
By combining configurable neural network components with secure encrypted execution, FHEON supports privacy-preserving machine learning deployments for sensitive and security-critical applications.

This appendix documents FHEON in the **SINGLE_INPUTS** mode of the framework. The single-input pipeline performs encrypted inference on one sample at a time and is suitable for low-latency per-sample deployments, benchmarking, and controlled research comparisons.

Models covered in this appendix:
- `LeNet5` with MNIST.
- `ResNet20` with CIFAR-10.
- `VGG11` with CIFAR-10.
- `VGG16` with CIFAR-10.
- `ResNet34` with CIFAR-100.

The C++ executables are generated from `CMakeLists.txt` when building with `-DMODE=SINGLE_INPUTS` or `-DMODE=ALL -DSINGLE_MODEL=...`.

## Security / Privacy Notes

- All inference in this pipeline uses CKKS via OpenFHE, so keys, weights, and datasets should be protected carefully.
- This artifact is research-grade and is not production-hardened.
- Encrypted-model evaluation still requires secure handling of generated parameters, model artifacts, and any derived outputs.
- Datasets may include sensitive or restricted content depending on their source and usage context; follow institutional and legal requirements.

## Hardware & Software Requirements

- OS: Linux, with Ubuntu 20.04, 22.04, or 24.04 recommended.
- Compiler: `gcc` 9+ or `clang` 10+.
- CMake: 3.18+.
- Python: 3.8+ for auxiliary scripts.
- Libraries: OpenFHE installed and discoverable via `find_package`.
- Recommended hardware: at least 8 CPU cores and 16 GB RAM for comfortable native builds.


### Estimated Time and Storage Consumption (Required for Functional and Reproduced badges)

- Minimal end-to-end setup (using Docker): ~30–90 minutes machine speed.
- Native build (OpenFHE + project): ~1 hours depending on CPU cores.
- Storage: ~50 GB (native) to ~50 GB (Docker image + data)


## Environment (Required for all badges)

Clone the repository and follow the README or use the Docker image for the quickest reproducible environment. You can either build OpenFHE locally and compile the C++ binaries, or use the `Dockerfile` included to produce a ready-to-run image.

### Set up the environment (Required for Functional and Reproduced badges)

#### Option A — Native build (manual)

#### Build OpenFHE Project

After installing OpenFHE as in [OpenFHE](https://github.com/openfheorg/openfhe-development)

```
git clone https://github.com/openfheorg/openfhe-development.git
cd openfhe
sudo apt-get install build-essential #this already includes g++
sudo apt-get install cmake
sudo apt-get install clang
sudo apt-get install libomp5
sudo apt-get install libomp-dev
mkdir build
cd build
cmake ..
make
sudo make install
```

## Build: SINGLE_INPUTS

From the repository root:

```bash
git clone https://github.com/stamcenter/fheon.git
cd fheon
mkdir -p build
cd build
cmake -DMODE=SINGLE_INPUTS -DSINGLE_MODEL=ALL ..
make -j$(nproc)
```

To build one specific model, for example `ResNet20`:

```bash
cmake -DMODE=SINGLE_INPUTS -DSINGLE_MODEL=ResNet20 ..
make -j$(nproc)
```

Executable profiles for the FHEON paper correspond to the source file names with `.cpp` removed. Depending on the build configuration, expected names include:
- `LeNet5`
- `ResNet20Optimized`
- `ResNet34Optimized`
- `VGG11`
- `VGG16`

## Run Examples

```bash
./build/LeNet5
./build/ResNet20Optimized
./build/ResNet34Optimized
./build/VGG11Optimized
./build/VGG16Optimized
```

Each binary produces the following outputs:

- It begins by displaying the CKKS crypto context parameters used in this work.
- It reports the required rotation keys that are generated, stored, and used for during inference.
- It loads the model weights when required.
- It reports the time required to inference every image required to interpret the results.

These results are summarized in Table 5. The default number of images run for each model (the test size) is 10. You can change this test size in two ways:

1. **At Build Time (CMake):**
   Configure the default test size when running CMake:
   ```bash
   cmake -DMODE=SINGLE_INPUTS -DTEST_SIZE=20 ..
   make -j$(nproc)
   ```

2. **At Run Time:**
   Override the default test size by passing the `--test_size` argument to the executable:
   ```bash
   ./build/LeNet5 --test_size 20
   ```


## Datasets

- `MNIST` for `LeNet5` (28×28 grayscale, 10 classes).
- `CIFAR-10`  for `ResNet20` and `VGG11` (3×32×32 RGB,  10 classes).
- `CIFAR-100` for `ResNet34` and `VGG16` (3×32×32 RGB, 100 classes).


## Option B — Docker

To build the Docker image and run single-input model binaries inside a container, you can use the helper script `run_docker_script.sh`. You can also configure the test size at build time or run time:

```bash
# Build the Docker image (compiles OpenFHE and FHEON inside the container)
./run_docker_script.sh build

# Build with a custom default test size (e.g., 20 instead of 10)
./run_docker_script.sh build --test_size 20

# Run specific single-input model binaries in Docker
./run_docker_script.sh run-lenet5
./run_docker_script.sh run-resnet20

# Run with a custom test size at run time (overrides the built-in default)
./run_docker_script.sh run-lenet5 --test_size 20
./run_docker_script.sh run-resnet20 --test_size 20
./run_docker_script.sh run-resnet34 --test_size 20
./run_docker_script.sh run-vgg11 --test_size 20
./run_docker_script.sh run-vgg16 --test_size 20

# Run accuracy checking script in Docker
./run_docker_script.sh run-accuracy

# Start an interactive bash session in the container
./run_docker_script.sh run
```

*(Note: If your user is not in the `docker` group, you will need to prefix these commands with `sudo`.)*


## Reproduction Checklist (Required for Functional and Reproduced badges)

1. Confirm that the target executable starts correctly and prints the expected HE configuration.
2. Verify that the correct model weights are present.
3. Run at each binary end to end and confirm that encrypted inference completes successfully.



### Main Results and Claims

- Use the executable’s internal logging to measure latency for each stage of encrypted inference.
- Measure peak memory with `/usr/bin/time -v` or `htop`.
- For timing comparisons consistent with the paper, use comparable hardware and minimize background system load.
- For Accuracy run `python3 results/accuracy.py`


## Limitations (Required for Functional and Reproduced badges)

- Exact runtime numbers depend on hardware, compiler flags, OpenFHE version, and system load.
- Latency may vary across machines even when using the same model configuration.
- Some builds may require careful configuration of OpenFHE installation paths and CMake variables.


## Notes on Reusability (Encouraged for all badges)

FHEON is structured to support reuse of its encrypted inference components in new privacy-preserving ML experiments.

- Modular model definitions make it straightforward to add new CNN architectures.
- The HE inference path can be adapted by replacing the model-specific layers while preserving the CKKS execution pipeline.
- Weight-export script can be reused to generate compatible encrypted-inference parameters.

## Extensibility

To extend the framework:
1. Add a new model implementation in the appropriate source directory.
2. Update CMake to build the corresponding executable.
3. Export model weights into the expected format.
4. Verify encrypted inference on a small benchmark set before running full experiments.
