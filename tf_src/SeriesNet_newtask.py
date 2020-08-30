import keras
import tensorflow as tf

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.applications import VGG16
from keras.optimizers import SGD

# Here we have our model architecture.
# (Transfer learning)
def SeriesNet_newtask(img_net, numClasses):
    
    # Instantiate model type.
    model = Sequential()

    # Fetch input layer from pretrained model
    input_layer = img_net.layers[0:1]

    # Fetch conv layers from the pretrained model
    layersTransfer = img_net.layers[1:-2]

    # Fetch the last dense layer from the pretrained model
    lastFcLayer = img_net.layers[-2]

    # Use the last layers shapes to allow for compatibility.
    inpSize = lastFcLayer.input_shape[1]
    inpShape = lastFcLayer.input_shape
    
    # Initialize weights and biases.
    def W(shape, dtype = None):
        return tf.Variable(lambda: tf.random.normal(shape) * 0.0001)
    
    def b(shape, dtype = None):
        return tf.Variable(lambda: tf.random.normal(shape) * 0.0001 + 1)
    
#     W = tf.random.normal((inpSize, numClasses)) * 0.0001
#     b = tf.random.normal((numClasses,)) * 0.0001 + 1

    # Instantiate a dropout layer (regularization)
    dropoutLayer = Dropout(rate = 0.2)

    # Instantiate a dense layer (hidden layer)
    newFC = Dense(
        units = inpSize,
        input_shape = inpShape,
        activation = 'softmax',
        kernel_initializer = W,
        kernel_regularizer = keras.regularizers.l2(l = 1),
        bias_initializer = b
    )

    # Instantiate a dense layer (output layer)
    classificationLayer = Dense(
        units = numClasses,
        activation='softmax',
        kernel_initializer = W,
        kernel_regularizer = keras.regularizers.l2(l = 1),
        bias_initializer = b
    )

    # Construct model.
    model.add(input_layer[0])
    for layer in layersTransfer:
        model.add(layer)
    #model.add(dropoutLayer)
    #model.add(newFC)
    model.add(classificationLayer)
    
    # Instantiate optimizer
    sgd = SGD(
        lr = 0.0001,
        momentum = 0.9,
        nesterov = False,
        decay = 1e-4
    )

    # Compile model.
    model.compile(
        optimizer = sgd,
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )

    # Show model summary
    model.summary()
    
    return model

