
# FHEON Framework Documentation

## FHEON

FHEON is a configurable framework for building **privacy-preserving convolutional neural networks (CNNs)** using **Homomorphic Encryption (HE)**.  

At its core, FHEON leverages the **Cheon–Kim–Kim–Song (CKKS) scheme**, a widely adopted approximate homomorphic encryption method designed for efficient computation on real-valued data.  
It enables computations directly on encrypted data, ensuring data confidentiality while performing inference in the encrypted domain.  
This capability allows users to run complete inference tasks in the encrypted domain without ever exposing the underlying inputs, thereby ensuring strong data confidentiality.  
In doing so, FHEON enables secure deployment of machine learning models in sensitive environments, bridging the gap between utility and privacy in encrypted inference.


### Key Features

- **Optimized HE CNN Layers**: Multiple variants of secure convolution, average pooling, ReLU, and fully connected layers.  
- **Configurable Architecture**: All CNN layers can be customized with standard parameters such as input/output channels, kernel size, stride, and padding.  
- **Versatile Evaluation**: Tested on multiple architectures including VGG-11, VGG-16, ResNet-20, and ResNet-34. Also tested on MNIST, CIFAR-10, and CIFAR-100 datasets.

FHEON provides a **flexible and efficient platform** for researchers and developers to build HE-friendly neural networks without sacrificing accuracy or privacy.

---

## Documentation

The full documentation for FHEON can be found on our website at:

[Read the FHEON Documentation](https://fheon.pqcsecure.org)

---

## Building FHEON

To build FHEON, follow the instructions at 

[Build FHEON Documentation](https://fheon.pqcsecure.org/getting_started.html)


---

## Build FHEON Examplary Models

FHEON provides flexible build options to compile specific models and configurations. The build process is controlled through CMake configuration flags.

### Basic Build

To build all models with default settings (both single-input and high-throughput modes):

```bash
mkdir build && cd build
cmake ..
make
```

### Build Configuration Options

The CMake configuration supports the following options:

#### MODE (default: ALL)
Controls which mode(s) to build:
- **ALL**: Builds both single-input and high-throughput models (default)
- **SINGLE_INPUT**: Builds only single-input models
- **HIGH_THROUGHPUT**: Builds only high-throughput models

#### SINGLE_MODEL (default: ALL)
For single-input mode models:
- **ALL**: Build all single-input model variants (LeNet5, ResNet20, ResNet34, VGG11, VGG16)
- **LeNet5**: Build only LeNet5 variants
- **ResNet20**: Build only ResNet20 variants
- **ResNet34**: Build only ResNet34 variants
- **VGG11**: Build only VGG11 variants
- **VGG16**: Build only VGG16 variants

#### BATCH_MODEL (default: ALL)
For high-throughput mode models:
- **ALL**: Build all batch model variants
- **ResNet20**: Build only ResNet20 batch variants
- **ResNet34**: Build only ResNet34 batch variants

#### BATCH_SIZE (default: ALL)
Batch size for high-throughput models:
- **ALL**: Build all available batch sizes (16, 32, 64, 128, 256, 512) - default
- **16, 32, 64, 128, 256, 512**: Choose a specific batch size

### Build Examples

**Build all models (default)**
```bash
cmake ..
make
```

**Build only single-input models**
```bash
cmake -DMODE=SINGLE_INPUT ..
make
```

**Build only ResNet20 single-input model**
```bash
cmake -DMODE=SINGLE_INPUT -DSINGLE_MODEL=ResNet20 ..
make
```

**Build high-throughput ResNet20 with batch size 64**
```bash
cmake -DMODE=HIGH_THROUGHPUT -DBATCH_MODEL=ResNet20 -DBATCH_SIZE=64 ..
make
```

**Build all high-throughput models with batch size 128**
```bash
cmake -DMODE=HIGH_THROUGHPUT -DBATCH_SIZE=128 ..
make
```

**Build all models - single-input and high-throughput with batch size 256**
```bash
cmake -DMODE=ALL -DBATCH_SIZE=256 ..
make
```

### Static vs Shared Libraries

By default, FHEON links against OpenFHE's shared libraries. To build with static libraries:

```bash
cmake -DBUILD_STATIC=ON ..
make
```

### Build Output

All compiled executables are placed in the `build/` directory. Each model is compiled as a separate executable with a name corresponding to the model configuration (e.g., `ResNet20Basic`, `TResNet20N64`).

---

## Supported Models

FHEON supports a variety of CNN architectures across different evaluation modes:

[Visit FHEON Documentation for more information on the different models and variants](https://fheon.pqcsecure.org)

---

## Citation

If you use FHEON in your work, please cite the following papers:


### Encrypted Single-input Inference
```bibtex
@misc{njungle2025fheonconfigurableframeworkdeveloping,
      title={FHEON: A Configurable Framework for Developing Privacy-Preserving Neural Networks Using Homomorphic Encryption}, 
      author={Nges Brian Njungle and Eric Jahns and Michel A. Kinsy},
      year={2025},
      eprint={2510.03996},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2510.03996}, 
}
```

Also avaliable online at
[FHEON: A Configurable Framework for Developing Encrypted Privacy-Preserving Neural Networks Using Homomorphic Encryption](https://arxiv.org/abs/2510.03996)


### Encrypted High-throughput Inference
```bibtex
@misc{njungle2026deepencryptedtraininglowlatency,
      title={Towards Deep Encrypted Training: Low-Latency, Memory-Efficient, and High-Throughput Inference for Privacy-Preserving Neural Networks}, 
      author={Nges Brian Njungle and Eric Jahns and Michel A. Kinsy},
      year={2026},
      eprint={2604.16834},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2604.16834}, 
}
```

Also avaliable online at
[Towards Deep Encrypted Training: Low-Latency, Memory-Efficient, and High-Throughput Inference for Privacy-Preserving Neural Networks](https://arxiv.org/abs/2604.16834)

