
# ***********************************************************************************************************************
#  @author: Nges Brian, Njungle
# 
#  MIT License
# Copyright (c) 2025 Secure, Trusted and Assured Microelectronics, Arizona State University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ***********************************************************************************************************************


## This file serves as an example demonstrating how to export model weights from a trained PyTorch model for use with FHEON.
## This is how all weights used in the FHEON examples where exported

import torch
import os
import csv
import torch.nn as nn
from resnet20Model import ResNet20 

def fold_batch_norm(model: nn.Module, data_path):
    """
    Fold BatchNorm layers into Conv layers, then save the weights and biases
    of each layer into separate CSV files in the specified `data_path`.
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)  # Create the directory if it doesn't exist

    conv_module = None
    bn_next = False

    model.eval()

    for index, (name, module) in enumerate(model.named_modules()):
        # Identify Conv2d layers and handle BatchNorm folding
        if isinstance(module, nn.Conv2d):
            bn_next = True
            conv_module = module
        if isinstance(module, nn.BatchNorm2d) and bn_next:
            bn_next = False

            gamma = module.weight.data  # The scale factor
            beta = module.bias.data  # The shift factor
            mean = module.running_mean  # The moving average of means
            var = module.running_var  # The moving average of variances
            epsilon = module.eps  # Epsilon to avoid division by zero

            # Compute the new Conv2d weights (fold BatchNorm into Conv)
            with torch.no_grad():
                # Scale Conv weights by gamma / sqrt(var + epsilon)
                conv_module.weight.data = conv_module.weight.data * (gamma / torch.sqrt(var + epsilon)).view(-1, 1, 1, 1)

                if ".bn" in name:
                    name_to_save = name.replace(".bn", "_conv")
                elif ".shortcut_bn" in name:  # Handling shortcut BatchNorm
                    name_to_save = name.replace(".shortcut_bn", "_shortcut")
                     # If there is a stride (downsampling), scale the weights and biases accordingly
                    # if conv_module.stride != (1, 1):  # Check if stride > 1 (downsampling)
                        # When stride > 1, we need to scale the gamma and beta by the stride factor
                        # scale_factor = torch.sqrt(torch.tensor(1.0 / (conv_module.stride[0] ** 2)))
                        # conv_module.weight.data *= scale_factor.view(-1, 1, 1, 1)
                        # gamma *= scale_factor
                        # beta *= scale_factor
                    # print(f"Processing shortcut BN: {name_to_save}")
                elif "bn1" in name:
                    name_to_save = name.replace("bn1", "layer0_conv1")

                # Save weights
                weights_filename = os.path.join(data_path, f'{name_to_save}_weight.csv')
                print(f'weight name: {weights_filename}')
                clean_data = conv_module.weight.data.numpy() if hasattr(conv_module.weight.data, 'numpy') else conv_module.weight.data
                with open(weights_filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(clean_data.flatten().reshape(1, -1))

                # Adjust Conv bias using BatchNorm parameters
                if conv_module.bias is not None:
                    conv_module.bias.data = (gamma * (conv_module.bias.data - mean) / torch.sqrt(var + epsilon)) + beta
                    
                    # Save bias
                    biases_filename = os.path.join(data_path, f'{name_to_save}_bias.csv')
                    print(f'bias name: {biases_filename}')
                    clean_bias_data = conv_module.bias.numpy() if hasattr(conv_module.bias, 'numpy') else conv_module.bias
                    with open(biases_filename, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerows(clean_bias_data.flatten().reshape(1, -1))
                else:
                    # If Conv2d doesn't have a bias, add one using BatchNorm parameters
                    biases_filename = os.path.join(data_path, f'{name_to_save}_bias.csv')
                    print(f'bias name: {biases_filename}')
                    conv_module.bias = nn.Parameter((gamma * (-mean) / torch.sqrt(var + epsilon)) + beta)
                    clean_bias_data = conv_module.bias.numpy() if hasattr(conv_module.bias, 'numpy') else conv_module.bias
                    with open(biases_filename, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerows(clean_bias_data.flatten().reshape(1, -1))
            # Reset conv_module after processing BatchNorm folding
            conv_module = None

        # Handle fully connected layer (fc)
        if name == "fc":
            fc_weights_filename = os.path.join(data_path, f'layer_fc_weight.csv')
            fc_bias_filename = os.path.join(data_path, f'layer_fc_bias.csv')
            print(f'fc layer weights name: {fc_weights_filename}')
            print(f'fc layer bias name: {fc_bias_filename}')
            clean_weights_data = module.weight.data.numpy() if hasattr(module.weight.data, 'numpy') else module.weight.data
            clean_bias_data = module.bias.data.numpy() if hasattr(module.bias.data, 'numpy') else module.bias.data
            with open(fc_weights_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(clean_weights_data.flatten().reshape(1, -1))
            
            with open(fc_bias_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(clean_bias_data.flatten().reshape(1, -1))

    print("BatchNorm folded into Conv layers and weights/biases saved to CSV.")
    return replace_bn_with_identity(model)

def replace_bn_with_identity(model):
    """
    Replaces all BatchNorm layers in the model with Identity layers.
    This is done after folding BatchNorm into Conv layers.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm3d):
            # Replace with Identity layer
            setattr(model, name, nn.Identity())
        else:
            # Recursively handle nested submodules
            replace_bn_with_identity(module)
    return model


data_path = 'path to save your model weights'

# load weights
channel_values = [16, 32, 64]
num_classes = 10

model = ResNet20(channel_values, num_classes)

model.load_state_dict(torch.load('path to load model'), strict=False)
model.eval()

# fake_data = torch.rand((1, 3, 32, 32))
# print(model(fake_data))
model = fold_batch_norm(model, data_path)
# print(model(fake_data))
