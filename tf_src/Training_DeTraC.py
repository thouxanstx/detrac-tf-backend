from SeriesNet_newtask import SeriesNet_newtask
from ConfusionMat_MultiClass import ConfusionMat_MultiClass
from utils import load_images, KFold_cross_validation_split, get_checkpointer
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import load_model, save_model, Sequential
from keras.optimizers import SGD
from keras.applications import VGG16
from sklearn.metrics import confusion_matrix
import numpy as np
import os

vgg16 = VGG16(weights = 'imagenet', input_shape = (224, 224, 3), include_top = True)


sars_1_path = '../dataset_B/SARS_1/'
sars_2_path = '../dataset_B/SARS_2/'

norm_1_path = '../dataset_B/norm_1/'
norm_2_path = '../dataset_B/norm_2/'

covid19_1_path = '../dataset_B/COVID_19_1/'
covid19_2_path = '../dataset_B/COVID_19_2/'

def load_training_data(covid19_1_path, covid19_2_path, sars_1_path, sars_2_path, norm_1_path, norm_2_path):
    covid19_1_x, covid19_1_y = load_images(covid19_1_path, 224, decompose = False)
    covid19_2_x, covid19_2_y = load_images(covid19_2_path, 224, decompose = False)

    norm_1_x, norm_1_y = load_images(norm_1_path, 224, decompose = False)
    norm_2_x, norm_2_y = load_images(norm_2_path, 224, decompose = False)

    sars_1_x, sars_1_y = load_images(sars_1_path, 224, decompose = False)
    sars_2_x, sars_2_y = load_images(sars_2_path, 224, decompose = False)

    x = np.concatenate((covid19_1_x, covid19_2_x, sars_1_x, sars_2_x, norm_1_x, norm_2_x))
    y = np.concatenate((covid19_1_y, covid19_2_y, sars_1_y, sars_2_y, norm_1_y, norm_2_y))
    
    return x, y




numClasses = 6

# Training params
maxEpochs = 100
miniBatchSize = 64
learningRate = 0.0001
momentum = 0.95
decayRate = 1e-4



reduce_lr_on_plateau = ReduceLROnPlateau(monitor = "val_loss", factor = 0.95, patience = 5, verbose = 1)
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 1, mode = 'min', restore_best_weights = True)

def trainDeTraC(img_net, numClasses, filepath, model_name):
    if not os.path.exists(f"./checkpoint/weight_checkpoint_detrac.h5"):
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
        
        for layer in model.layers:
            print(layer.trainable)
        

        #model.layers[-1].trainable = False
        
        # Show summary of model
        model.summary()
        
        x, y = load_training_data(covid19_1_path, covid19_2_path, sars_1_path, sars_2_path, norm_1_path, norm_2_path)
        X_train, X_test, Y_train, Y_test = KFold_cross_validation_split(x, y, 2)
        X_train /= 255
        X_test /= 255

        np.save('./features/detrac_x_train.npy', X_train)
        np.save('./features/detrac_y_train.npy', Y_train)
        np.save('./features/detrac_x_test.npy', X_test)
        np.save('./features/detrac_y_test.npy', Y_test)
        
        # Train model
        model.fit(
            x = X_train,
            y = Y_train,
            batch_size = 64,
            epochs = 100,
            validation_data = (X_test, Y_test),
            shuffle = True,
            verbose = 1,
            callbacks = [reduce_lr_on_plateau, checkpointer, early_stopping],
        )
        

    # Load mode condition
    else:
    
        # Instantiate model
        model_checkpoint = SeriesNet_newtask(img_net, numClasses, learningRate, momentum, decayRate)
        
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
    X_train = np.load('./features/detrac_x_train.npy')
    Y_train = np.load('./features/detrac_y_train.npy')
    X_test = np.load('./features/detrac_x_test.npy')
    Y_test = np.load('./features/detrac_y_test.npy')
    
    # Evaluate model
    score, acc = model.evaluate(
        x = X_test,
        y = Y_test,
        batch_size = 1
    )
    
    return score, acc, model, X_train, Y_train, X_test, Y_test
            
            
            
            


# Train.
img_net = input('Input the ImageNet pre-trained network: ')
if img_net == 'SeriesNetwork' or img_net == 'SeriesNet':
    score, acc, model, X_train, Y_train, X_test, Y_test = trainDeTraC(
        img_net = vgg16, 
        numClasses = numClasses, 
        filepath = 'weight_checkpoint_detrac', 
        model_name = 'SeriesNet'
    )
    y_true = Y_test
    y_pred = model.predict(X_test)


elif img_net == 'DAGNetwork' or img_net == 'DAGNet':
    trainDAGNet(vgg16, numClasses)
    
# Here we'll use a confusion matrix to fully assess the correctness of the output.

def compose_classes(cmat, block_size: tuple):
    sizes = list(tuple(np.array(cmat.shape) // block_size) + block_size)
    for i in range(len(sizes)):
        if (i + 1) == len(sizes) - 1:
            break
        if i % 2 != 0:
            temp = sizes[i]
            sizes[i] = sizes[i + 1]
            sizes[i + 1] = temp
            
    reshaped_matrix = cmat.reshape(sizes)
    composed = reshaped_matrix.sum(axis = (1, 3))
    return composed

cmat = confusion_matrix(y_true.argmax(axis = 1), y_pred.argmax(axis = 1))
    
k = 2
composition_classes = compose_classes(cmat, (k, k))




print(f"score = {score}\nacc = {acc}\ncmat = {cmat}\ny_true = {y_true}\ny_pred = {y_pred}\n\n")
acc, sn, sp = ConfusionMat_MultiClass(composition_classes, 3)

print(f"ACCURACY = {acc}\nSENSITIVITY = {sn}\nSPECIFICITY = {sp}")