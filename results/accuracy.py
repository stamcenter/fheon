# ***********************************************************************************************************************
# @author: Nges Brian, Njungle
# 
# MIT License
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

# plaintext prediction
def compare_strings(fhe_predictions, dataset_labels, printMessage):
    # Find the minimum length to avoid index out of range issues
    min_length = min(len(fhe_predictions), len(dataset_labels))
    if min_length == 0:
        return 0
    # Count the number of matching labels at the same position
    match_count = sum(1 for i in range(min_length) if fhe_predictions[i] == dataset_labels[i])
    accuracy = (match_count/min_length) * 100
    print(f'{printMessage} Accuracy: {accuracy:.1f}% ({match_count}/{min_length} lines)')
    return match_count




def construct_text(file_path):
    try:
        with open(file_path, 'r') as file:
            values = []
            for line in file:
                parts = line.strip().split()
                for item in parts:
                    try:
                        values.append(int(item))
                    except ValueError:
                        values.append(item)
            return values
    except FileNotFoundError:
        return []




# Loading the content of the file
true_labels_file_path = './lenet5/truelabels.txt'
fhe_predictions_file_path = './lenet5/fhepredictions.txt'
pytorch_predictions_file_path = './lenet5/pytorchpredictions.txt'
mnist_truelabels = construct_text(true_labels_file_path)
fhe_predictions = construct_text(fhe_predictions_file_path)
pytorch_predictions = construct_text(pytorch_predictions_file_path)
matching_chars = compare_strings(pytorch_predictions, mnist_truelabels, 'PyTorch Lenet5:')
matching_chars = compare_strings(fhe_predictions, mnist_truelabels, 'FHE Lenet5:')

# Loading the different files
true_labels_file_path = './resnet20/truelabels.txt'
fhe_predictions_file_path = './resnet20/resnet20fhepredictions.txt'
pytorch_predictions_file_path = './resnet20/pytorchpredictions.txt'
cifar10_truelabels = construct_text(true_labels_file_path)
fhe_predictions = construct_text(fhe_predictions_file_path)
pytorch_predictions = construct_text(pytorch_predictions_file_path)
matching_chars = compare_strings(pytorch_predictions, cifar10_truelabels, 'PyTorch ResNet20:')
matching_chars = compare_strings(fhe_predictions, cifar10_truelabels, 'FHE ResNet20:')


# Loading the different files
true_labels_file_path = './resnet34/truelabels.txt'
fhe_predictions_file_path = './resnet34/resnet34fhepredictions.txt'
pytorch_predictions_file_path = './resnet34/pytorchpredictions.txt'
cifar100_truelabels = construct_text(true_labels_file_path)
fhe_predictions = construct_text(fhe_predictions_file_path)
pytorch_predictions = construct_text(pytorch_predictions_file_path)
matching_chars = compare_strings(pytorch_predictions, cifar100_truelabels, 'PyTorch ResNet34:')
matching_chars = compare_strings(fhe_predictions, cifar100_truelabels, 'FHE ResNet34:')


# Loading the different files
true_labels_file_path = './vgg11/truelabels.txt'
fhe_predictions_file_path = './vgg11/vgg11fhepredictions.txt'
pytorch_predictions_file_path = './vgg11/pytorchpredictions.txt'
cifar10_truelabels = construct_text(true_labels_file_path)
fhe_predictions = construct_text(fhe_predictions_file_path)
pytorch_predictions = construct_text(pytorch_predictions_file_path)
matching_chars = compare_strings(pytorch_predictions, cifar10_truelabels, 'PyTorch VGG11:')
matching_chars = compare_strings(fhe_predictions, cifar10_truelabels, 'FHE VGG11:')


# Loading the different files
true_labels_file_path = './vgg16/truelabels.txt'
fhe_predictions_file_path = './vgg16/vgg16fhepredictions.txt'
pytorch_predictions_file_path = './vgg16/pytorchpredictions.txt'
cifar10_truelabels = construct_text(true_labels_file_path)
fhe_predictions = construct_text(fhe_predictions_file_path)
pytorch_predictions = construct_text(pytorch_predictions_file_path)
matching_chars = compare_strings(pytorch_predictions, cifar10_truelabels, 'PyTorch VGG16:')
matching_chars = compare_strings(fhe_predictions, cifar10_truelabels, 'FHE VGG16:')