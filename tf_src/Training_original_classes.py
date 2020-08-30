from ConfusionMat_MultiClass import ConfusionMat_MultiClass
from readAndPreprocessImage import readAndPreprocessImage
from SeriesNet_newtask import SeriesNet_newtask
from DAGNet_newtask import DAGNet_newtask
from extract_features import extract_features
from utils import load_images, addImgArrToX, KFold_cross_validation_split, get_checkpointer
import os
from os import path
import sys
from math import floor

import random
from collections import OrderedDict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.applications import VGG16
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import load_model, save_model, Sequential
from keras.optimizers import SGD

import matplotlib.pyplot as plt
import pathlib
import cv2

# Pretrained Model Instance
vgg16 = VGG16(weights = 'imagenet', input_shape = (224, 224, 3), include_top = True)

# Paths to dataset
sars_path = '../dataset_A/SARS/'
covid_path = '../dataset_A/Covid_19/'
normal_path = '../dataset_A/normal/'


# Get features and labels for each piece of data
covid19_x, covid_y = load_images(covid_path, 224)
sars_x, sars_y = load_images(sars_path, 224)
normal_x, normal_y = load_images(normal_path, 224)

# Construct full dataset (x = all features, y = all labels)
x = np.concatenate((covid19_x, sars_x, normal_x))
y = np.concatenate((covid_y, sars_y, normal_y))

# There are 3 classes: Covid-19, SARS and Normal
numClasses = 3

# Training params
maxEpochs = 100
miniBatchSize = 64
learningRate = 0.0001
decayRate = 1e-3
momentum = 0.9
# Create the train set (features + labels) and the test set (features + labels)
# X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size = 0.7)
X_train, X_test, Y_train, Y_test = KFold_cross_validation_split(x, y, 2)

# Normalize the data to allow the net to train easily. 
# In images, colors are mapped with values from 0 to 255, so we'll divide by 255 to achieve values in the interval [0, 1]
X_train /= 255
X_test /= 255

# # For now, this isn't used, so it will be commented out.
# '''
# # In the original file, there is this variable assignment:
# # numObservations = numel(trainingimages.Files);
# # Where is "trainingimages.Files coming from?"
# # Replacing for now with len(X_train)
# numObservations = len(X_train)
# numIterationsPerEpoch = floor(numObservations / miniBatchSize)

# # NOTE: Might have to use a different approach to 
# # replicate this part of the code:
# # opts = trainingOptions('sgdm',...
# #                     'Initiallearnrate',0.0001,...
# #                     'Minibatchsize',miniBatchSize,...   
# #                     'maxEpoch',maxEpochs,...            
# #                     'L2Regularization',0.001,...        
# #                     'Shuffle','every-epoch','Momentum',0.9,...
# #                     'Plots','training-progress','LearnRateSchedule', 'piecewise', ...    
# #                     'LearnRateSchedule', 'piecewise', 'LearnRateDropFactor', 0.9,'LearnRateDropPeriod',3,...
# #                     'CheckpointPath' ,'C:\.........\New folder');
# '''



reduce_lr_on_plateau = ReduceLROnPlateau(monitor = "val_loss", factor = 0.9, patience = 3, verbose = 1)
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 6, verbose = 1, mode = 'min', restore_best_weights = True)

# This is where we train our net.
def trainSeriesNet(img_net, numClasses, filepath, model_name):
    if not os.path.exists(f"./checkpoint/weight_checkpoint.h5"):
        new_model_prompt = "T"
    else:
        new_model_prompt = input("Train the model / Load the existing model [T / L]")

    # Train mode condition
    if new_model_prompt == "T":

        # Instantiate model
        model = SeriesNet_newtask(img_net, numClasses, learningRate, momentum, decayRate)

        # Get checkpoint callback
        checkpointer = get_checkpointer(filepath)
        
        #model.build(input_shape = (
        #    None,
        #    input.shape[1],
        #    input.shape[2],
        #    input.shape[3]  
        #    )
        #)
        
        # Show summary of model
        
        
        for layer in model.layers[:len(model.layers) - 1]:
            layer.trainable = False

            
        model.summary()
        
        # Train model
        model.fit(
            x = X_train,
            y = Y_train,
            batch_size = 64,
            epochs = maxEpochs,
            validation_data = (X_test, Y_test),
            shuffle = True,
            verbose = 1,
            callbacks = [reduce_lr_on_plateau, checkpointer, early_stopping],
        )
        
        # Extract features using trained model
        covid_features = extract_features(covid_path, model, 224, './checkpoint/weight_checkpoint.h5')
        np.save("./features/covid_features.npy", covid_features)

        sars_features = extract_features(sars_path, model, 224, './checkpoint/weight_checkpoint.h5')
        np.save("./features/sars_features.npy", sars_features)

        normal_features = extract_features(normal_path, model, 224, './checkpoint/weight_checkpoint.h5')
        np.save("./features/normal_features.npy", normal_features)

    # Load mode condition
    else:

        # Instantiate model
        model_checkpoint = SeriesNet_newtask(img_net, numClasses)
        
        # Load model's weights
        model_checkpoint.load_weights(f"./checkpoint/{filepath}.h5")

        # Create model structure
        model = Sequential()
        
        # Add layers from checkpoint
        for layer in model_checkpoint.layers:
            model.add(layer)

        # Instantiate optimizer
        sgd = SGD(
            lr = learningRate,
            momentum = momentum,
            nesterov = False,
            decay = decayRate
        )

        # Compile model
        model.compile(
            optimizer = sgd,
            loss = 'categorical_crossentropy',
            metrics = ['accuracy']
        )
    
        # Load extracted features
        covid_features = np.load("./features/covid_features.npy")
        sars_features = np.load("./features/sars_features.npy")
        normal_features = np.load("./features/normal_features.npy")

    # Evaluate model
    score, acc = model.evaluate(
        x = X_test,
        y = Y_test,
        batch_size = 1
    )

    return covid_features, sars_features, normal_features, score, acc, model

# DAGNet isn't used for now, but we'll leave it here in case we go back to it.
def trainDAGNet(img_net, numClasses):
    model = DAGNet_newtask(img_net, numClasses)
    model.summary()
    model.fit(
        x = X_train,
        y = Y_train,
        batch_size = 1,
        epochs = 1,
        validation_data = (X_test, Y_test),
        shuffle = True
    )

# Train.
img_net = input('Input the ImageNet pre-trained network: ')
if img_net == 'SeriesNetwork' or img_net == 'SeriesNet':
    covid_features, sars_features, normal_features, score, acc, model = trainSeriesNet(
        img_net = vgg16, 
        numClasses = numClasses, 
        filepath = 'weight_checkpoint', 
        model_name = 'seriesNet'
    )

elif img_net == 'DAGNetwork' or img_net == 'DAGNet':
    trainDAGNet(vgg16, numClasses)
    
# Here we'll use a confusion matrix to fully assess the correctness of the output.
y_true = Y_test
y_pred = model.predict(X_test)

cmat = confusion_matrix(y_true.argmax(axis = 1), y_pred.argmax(axis = 1))

print(f"score = {score}\nacc = {acc}\ncmat = {cmat}\ny_true = {y_true}\ny_pred = {y_pred}\n\n")
acc, sn, sp = ConfusionMat_MultiClass(cmat, numClasses)

print(f"ACCURACY = {acc}\nSENSITIVITY = {sn}\nSPECIFICITY = {sp}")

# NOTE: ConfusionMat_MultiClass.py fully completed
# NOTE: readAndPreprocessImage.py fully completed
# NOTE: DAGNet.py completed (will have to modify it in time once we have a better idea of how to replicate it with the one in MATLAB)