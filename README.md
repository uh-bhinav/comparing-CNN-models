Concrete Crack Detection with VGG16 and ResNet50

This project demonstrates the use of the VGG16 and ResNet50 pre-trained models for classifying images of concrete surfaces to identify cracks. By leveraging the strengths of these convolutional neural network (CNN) architectures, the project evaluates and compares their performance in detecting cracks in concrete images.

Table of Contents

Introduction
Project Structure
Data
Model Development
VGG16 Classifier
ResNet50 Classifier
Evaluation
Prediction
Results
Requirements
Installation
Introduction

The goal of this project is to build and evaluate two image classifiers using the VGG16 and ResNet50 architectures. We will assess the performance of each model and determine which performs better at distinguishing cracked concrete from non-cracked concrete images.

Project Structure

Download Data: Retrieve and unzip the data, which is divided into train, validation, and test sets.
Model Development:
Part 1: Build an image classifier using the VGG16 model.
Part 2: Evaluate the VGG16 and ResNet50 models on the test dataset.
Part 3: Use the trained models to predict and classify images as cracked or non-cracked concrete.
Data

The dataset can be downloaded using the following command:

!wget https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/concrete_data_week4.zip
!unzip concrete_data_week4.zip
The data is organized into three sets: train, valid, and test. Each set includes images labeled for cracked or non-cracked concrete.

Model Development

VGG16 Classifier
Import Libraries: Load essential libraries, including VGG16 and ImageDataGenerator.
Data Augmentation: Construct ImageDataGenerator instances for training and validation with the necessary preprocessing.
Build the Model: Initialize a Sequential model and add the VGG16 layers (with pre-trained weights) and a Dense layer for classification.
Compile and Train: Compile the model with adam optimizer and categorical_crossentropy loss. Train on the augmented dataset.
ResNet50 Classifier
Load Pre-trained Model: Load the saved ResNet50 model.
Evaluate: Evaluate its performance on the test set using evaluate_generator.
Evaluation

Both VGG16 and ResNet50 classifiers are evaluated on the test dataset using evaluate_generator, which computes loss and accuracy metrics. These metrics allow us to compare the models' performance on unseen data.

Prediction

The VGG16 model is used to predict the class of each image in the test set using the predict_generator function. The class predictions of the first five test images are reported to provide a quick insight into the model's classification output.

Results

The performance metrics for both models (test loss and test accuracy) are printed as follows:

VGG16 Performance
Test Loss: test_loss_vgg
Test Accuracy: test_accuracy_vgg
ResNet50 Performance
Test Loss: test_loss_resnet
Test Accuracy: test_accuracy_resnet

Requirements

TensorFlow (2.17.0)
Keras (2.15.0)
Matplotlib (3.9.2)
NumPy (1.26.4)
SciPy (1.14.1)
scikit-learn (1.5.2)
skillsnetwork
