import cv2
import numpy as np
import os
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.models import Sequential

# Here we extract the features from the trained model.
def extract_features(dataset, model_checkpoint, im_size, weights_checkpoint):
    # Our array where the raw features will go.
    train_features = []
    
    # Iterate through each file
    for filename in os.listdir(dataset):
        # Preprocess as usual.
        gray_img = cv2.imread(os.path.join(dataset, filename))
        color_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB)
        img = cv2. resize(color_img, (im_size, im_size))
        img = img_to_array(img)
        img = np.expand_dims(img, axis = 0)
        img = imagenet_utils.preprocess_input(img)
        train_features.append(img)

    # Reshape features array to be a vertical stack of features.
    x = np.vstack(train_features)
    
    print('Extracting features...')
    
    # Load model.
    model_checkpoint.load_weights(weights_checkpoint)
    layers = model_checkpoint.layers[:-1]
    model = Sequential()
    
    for layer in layers:
        model.add(layer)
        
    # Get predicted features based on actual features.
    features = model.predict(x, batch_size = 64)
    return features
