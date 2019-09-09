# Modular_Neural_Networks

This repository contains the code (in PyTorch) for "Modular Neural Networks for Low-Power ImageClassification in Embedded Devices" paper by Abhinav Goel, Sara Aghajanzadeh, Caleb Tung, Shuo-Han Chen, George K. Thiruvathukal, and Yung-Hsiang Lu.

## Introduction

The Modular Neural Network-Tree (MNN-Tree) is a novel hierarchial deep neural network (DNNs) architecture. It is designed to reduce redundant computation and memory accesses to reduce the energy consumption, memory requirement, and inference time of DNNs. We test the effectiveness of our model to obtain high-speed inference on embedded devices, like the Raspberry Pi 3 and Raspberry Pi Zero. The performance gains of using a hierarchical DNN comes at minimal loss to the classification accuracy.

In our hierarchical DNN, visually similar categories are grouped together using a novel similarity metric - called the Averaged Softmax Likelihood (ASL). This similarity metric has advantages over existing hierarchical classifiers that use semantic similarities, clustering, Gabor and HSV filters, etc.

## Usage

### Dependencies

- [Python3](https://www.python.org/downloads/)
- [PyTorch(1.1.0)](http://pytorch.org)

## Results

### Results on CIFAR

| Model | FLOPs | Params | CIFAR-10 | CIFAR-100 |
|---|---|---|---|---|
| CondenseNet-50 | 28.6M | 0.22M | 6.22 | - |
| CondenseNet-74 | 51.9M | 0.41M | 5.28 | - |
| CondenseNet-86 | 65.8M | 0.52M | 5.06 | 23.64 |
| CondenseNet-98 | 81.3M | 0.65M | 4.83 | - |
| CondenseNet-110 | 98.2M | 0.79M | 4.63 | - |
| CondenseNet-122 | 116.7M | 0.95M | 4.48 | - |
| CondenseNet-182* | 513M | 4.2M | 3.76 | 18.47 |

(* trained 600 epochs)

### Inference time on ARM platform

| Model | FLOPs | Top-1 | Time(s) |
|---|---|---|---|
| VGG-16 | 15,300M | 28.5 | 354 |
| ResNet-18 | 1,818M | 30.2 | 8.14 |
| 1.0 MobileNet-224 | 569M | 29.4 | 1.96 |
| CondenseNet-74 (C=G=4) | 529M | 26.2 | 1.89 |
| CondenseNet-74 (C=G=8) | 274M | 29.0 | 0.99 |

## Contact
goel39@purdue.edu
