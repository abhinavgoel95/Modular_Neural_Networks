# Modular_Neural_Networks

This repository contains the code (in PyTorch) for "Modular Neural Networks for Low-Power ImageClassification in Embedded Devices" paper by Abhinav Goel, Sara Aghajanzadeh, Caleb Tung, Shuo-Han Chen, George K. Thiruvathukal, and Yung-Hsiang Lu.

## Introduction

The Modular Neural Network-Tree (MNN-Tree) is a novel hierarchial deep neural network (DNNs) architecture. It is designed to reduce redundant computation and memory accesses to reduce the energy consumption, memory requirement, and inference time of DNNs. We test the effectiveness of our model to obtain high-speed inference on embedded devices, like the Raspberry Pi 3 and Raspberry Pi Zero. The performance gains of using a hierarchical DNN comes at minimal loss to the classification accuracy.

In our hierarchical DNN, visually similar categories are grouped together using a novel similarity metric - called the Averaged Softmax Likelihood (ASL). This similarity metric has advantages over existing hierarchical classifiers that use semantic similarities, clustering, Gabor and HSV filters, etc.

## Usage

### Dependencies

- [Python3](https://www.python.org/downloads/)
- [PyTorch(1.1.0)](http://pytorch.org)

Use ``` pip3 install -r requirements.txt ``` to install the dependencies for Python3.

## Results

### Results on CIFAR

| Model | Model Size (kB) | FLOPs | Error|
|---|---|---|---|---|
| VGG 16 | 78,410 | 313 M | 6.22 | 0.067 |
| VGG Pruned | 28,200 | 206 M | 5.28 | 0.066 |
| DenseNet-190 | 102,000 | 9,388 M | 5.06 | 0.070 |
| CondenseNet-160 | 43,000 | 1,084 M | 4.83 | 0.034 |
| WideResNet-28,10 | 140,000 | 25,748M | 4.63 | 0.040 |
| MNN-Tree | 806 | 28M | 0.079 |



## Contact
goel39@purdue.edu
