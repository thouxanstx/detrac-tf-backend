import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.applications import VGG16
import keras.regularizers

def DAGNet_newtask(img_net, numClasses):
    model = Sequential()
    learnableLayer = img_net.get_layer('fc2')
    classLayer = img_net.get_layer('predictions')
    inpSize = learnableLayer.input_shape[1]
    inpShape = learnableLayer.input_shape

    W = np.random.randn(inpSize, numClasses) * 0.0001
    b = np.random.randn(numClasses) * 0.0001 + 1
    
    newLearnableLayer = Dense(
        units = numClasses,
        input_shape = inpShape,
        activation = 'relu',
        kernel_initializer = 'random_normal',
        kernel_regularizer = 'l2',
        bias_initializer= 'zeros', 
        bias_regularizer = keras.regularizers.l2(l = 1),
        weights = [W, b]
    )
    
    

    classificationLayer = Dense(
        units = numClasses,
        activation='softmax'
    )

    model.add(newLearnableLayer)
    model.add(classificationLayer)

    model.compile(
        optimizer = 'sgd',
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )

    return model