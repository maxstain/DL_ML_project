# Image recognition using Convolutional Neural Networks (CNN)
## Introduction

This project is a simple implementation of a Convolutional Neural Network (CNN) for image recognition. The CNN is trained on the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 testing images. The classes are completely mutually exclusive. There are no overlaps between the classes.

The CNN is implemented using the Keras library with a TensorFlow backend. The model is trained using the Adam optimizer and the categorical cross-entropy loss function. The model is trained for 100 epochs with a batch size of 32. The model achieves an accuracy of 75.5% on the test set.

## Requirements

- Python 3.6
- TensorFlow 1.14.0
- Keras 2.2.4
- NumPy 1.16.4
- Matplotlib 3.1.0
- Jupyter 1.0.0
- scikit-learn 0.21.2
- h5py 2.9.0
- Pillow 6.0.0
- tqdm 4.32.1
- requests 2.22.0
- pandas 0.24.2
- opencv-python
- opencv-python-headless
- opencv-contrib-python
- opencv-contrib-python-headless
- opencv-python-headless
- opencv-contrib-python-headless
- opencv-python-headless

## Usage

To train the model, run the following command:

- bash
- python train.py