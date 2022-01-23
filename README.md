# Convolution Neural Networks for predicting the alphabet of american sign language image dataset.

## Overview
### Implement Convolutional Neural Network that is especially good for reading and classifying images, to classify American Sign language image dataset. The model was able to learn how to correctly classify the training dataset and validation dataset with generalization to non-training dataset.

## Dataset
### The Sign Language MNIST is used here and follows the same CSV format with labels and pixel values in single rows. The American Sign Language letter database of hand gestures represent a multi-class problem with 24 classes of letters (excluding J and Z which require motion). Each training and test case represents a label (0-25) as a one-to-one map for each alphabetic letter A-Z (and no cases for 9=J or 25=Z because of gesture motions). The training data (27,455 cases) and test data (7172 cases) are approximately half the size of the standard MNIST but otherwise similar with a header row of label, pixel1,pixel2….pixel784 which represent a single 28x28 pixel image with grayscale values between 0-255.

## Implementation
### Preparing the data
* In order to teach our model to be more robust when looking at new data, we're going to programmatically increase the size and variance in our dataset, also called data augmentation.The increase in size gives the model more images to learn from while training. The increase in variance helps the model ignore unimportant features and select only the features that are truly important in classification, allowing it to generalize better.
*  Model Creation :  "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 28, 28, 75)        750       
_________________________________________________________________
batch_normalization (BatchNo (None, 28, 28, 75)        300       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 75)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 14, 50)        33800     
_________________________________________________________________
dropout (Dropout)            (None, 14, 14, 50)        0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 14, 14, 50)        200       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 50)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 7, 7, 25)          11275     
_________________________________________________________________
batch_normalization_2 (Batch (None, 7, 7, 25)          100       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 25)          0         
_________________________________________________________________
flatten (Flatten)            (None, 400)               0         
_________________________________________________________________
dense (Dense)                (None, 512)               205312    
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 24)                12312     
=================================================================
Total params: 264,049
Trainable params: 263,749
Non-trainable params: 300

* Data Augmentation: Before compiling the model, it's time to set up our data augmentation. Keras comes with an image augmentation class called ImageDataGenerator. It accepts a series of options for augmenting your data. We would want to flip images horizontally, but not vertically. When you have an idea, reveal the text below. Our dataset is pictures of hands signing the alphabet. If we want to use this model to classify hand images later, it's unlikely that those hands are going to be upside-down, but, they might be left-handed. This kind of domain-specific reasoning can help make good decisions for your own deep learning applications.
* Training with Augmentation: When using an image data generator with Keras, a model trains a bit differently: instead of just passing the x_train and y_train datasets into the model, we pass the generator in, calling the generator's flow method. This causes the images to get augmented live and in memory right before they are passed into the model for training. Generators can supply an indefinite amount of data, and when we use them to train our data, we need to explicitly set how long we want each epoch to run, or else the epoch will go on indefinitely, with the generator creating an indefinite number of augmented images to provide the model.
* Finally made predictions on real dataset.

## Results
