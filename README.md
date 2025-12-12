
# FHEON - A Configurable Framework for Developing Encrypted Neural Networks

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

[Bluid FHEON Documentation](https://fheon.pqcsecure.org/getting_started.html)


---


## Citation

If you use FHEON in your work, please cite the following paper:

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

Which is also avaliable online at
[FHEON: A Configurable Framework for Developing Encrypted Privacy-Preserving Neural Networks Using Homomorphic Encryption](https://arxiv.org/abs/2510.03996)

