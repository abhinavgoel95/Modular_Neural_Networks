# Modular Neural Network Tree (MNN-Tree)

This repository contains the code (in PyTorch) for "Modular Neural Networks for Low-Power ImageClassification in Embedded Devices" paper by Abhinav Goel, Sara Aghajanzadeh, Caleb Tung, Shuo-Han Chen, George K. Thiruvathukal, and Yung-Hsiang Lu.

## Usage

### Dependencies

- [Python3](https://www.python.org/downloads/)
- [PyTorch(1.1.0)](http://pytorch.org)

Use ``` pip3 install -r ../../requirements.txt ``` to install the dependencies for Python3.

## Directory Structure

- Demo: Contains Jupyter Notebook file that performs inference on sample images.
- Models: Contains trained pytorch models of the modules of the MNN-Tree.
- Softmax Output: Contains the saved softmax output values; required by Averaged Softmax Likelihood.
- Test: Contains scripts to find the accuracy of the MNN-Tree.
- Train: Contains scripts to train the MNN-Tree's modules.

## Contact
goel39@purdue.edu
