import cv2
import os
import numpy as np

from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.model_selection import KFold
from keras.callbacks import ModelCheckpoint

# Here we'll do some preprocessing, using one-hot encoding for each category (Covid-19, SARS and Normal)
def load_images(dataset, im_size, decompose = True):
    x = []
    y = []

    # We go through each file
    for filename in os.listdir(dataset):
        # Preprocess the image:
        # 1. Read grayscale images
        # 2. Convert the grayscale channel to RGB channel
        # 3. Resize the image appropriately - Pretrained models usually use the 224x224 resolution
        # 4. Append the data to the feature array (x in this case) 
        gray_img = cv2.imread(os.path.join(dataset, filename))
        color_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB)
        img = cv2. resize(color_img, (im_size, im_size))
        img = img_to_array(img)
        img = np.expand_dims(img, axis = 0)
        x.append(img)
        print('Loading: {}'.format(filename))

        # Do one-hot encoding to create the labels and append them to the label array (y in this case)
        # If the images are being loaded for KMeans decomposition, we load images from the 3 classes
        if decompose:  
            
            if dataset.split('/')[-2] == "Covid_19":
                y.append([1, 0, 0])
            elif dataset.split('/')[-2] == "SARS":
                y.append([0, 1, 0])
            elif dataset.split('/')[-2] == "normal":
                y.append([0, 0, 1])
                
        # If the images are being loaded after decomposition, we load the images from the 6 classes        
        else:
            
            if dataset.split('/')[-2] == "COVID_19_1":
                y.append([1, 0, 0, 0, 0, 0])
            elif dataset.split('/')[-2] == "COVID_19_2":
                y.append([0, 1, 0, 0, 0, 0])
            elif dataset.split('/')[-2] == "SARS_1":
                y.append([0, 0, 1, 0, 0, 0])
            elif dataset.split('/')[-2] == "SARS_2":
                y.append([0, 0, 0, 1, 0, 0])
            elif dataset.split('/')[-2] == "norm_1":
                y.append([0, 0, 0, 0, 1, 0])
            elif dataset.split('/')[-2] == "norm_2":
                y.append([0, 0, 0, 0, 0, 1])
    # Reshape x to be a vertical stack of arrays
    # Parse y as a numpy array
    x = np.vstack(x)
    y = np.asarray(y)
    return x, y


def addImgArrToX(X, arr):
    for img in arr:
        X.append(img)

# We'll use the KFold method to split the validation data - that way we get more variety when we feed the validation set to the net
def KFold_cross_validation_split(x, y, k):
    kf = KFold(n_splits = k, shuffle = True)
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        Y_train, Y_test = y[train_index], y[test_index]

    return X_train, X_test, Y_train, Y_test

# Checkpoint callback function to be fed to Keras' training function.
def get_checkpointer(filepath):
    checkpointer = ModelCheckpoint(
        filepath = './checkpoint/{}.h5'.format(filepath),
        verbose = 1,
        save_best_only = True,
        save_weights_only = True
        )
    return checkpointer