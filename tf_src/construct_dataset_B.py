import numpy as np
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
import sys

import cv2

# We'll use KMeans clustering to label the extracted features (going from supervised learning to unsupervised learning)
def decompose(path_to_features, n_clusters, path_to_images, path_to_decomposed_images_1, path_to_decomposed_images_2):
    sys.stdout.write("\r" + f"Loading features from {path_to_features}")
    features = np.load(path_to_features)
    
    sys.stdout.write("\r" + f"Creating KMeans Clusters for {path_to_features.split('/')[2].split('.')[0]}")
    idx = KMeans(n_clusters = n_clusters, random_state=0).fit(features)
    idx = idx.predict(features)
    
    images = [filename for filename in os.listdir(path_to_images)] 
    
    for i in range(len(images)):
        filename = path_to_images + images[i]
        sys.stdout.write("\r" + f"Loading image: {filename}")
        I = plt.imread(filename)

        filename_1 = path_to_decomposed_images_1 + images[i]
        filename_2 = path_to_decomposed_images_2 + images[i]
        if (idx[i] == 1):  
            sys.stdout.write("\r" + f"Saving decomposed image: {filename_1}")
            plt.imsave(filename_1, I)
        else:
            sys.stdout.write("\r" + f"Saving decomposed image: {filename_2}")
            plt.imsave(filename_2, I)

    sys.stdout.flush()
            
# decomposition of normal class 
decompose(
    path_to_features = './features/normal_features.npy',
    n_clusters = 2,
    path_to_images = '../dataset_A/normal/',
    path_to_decomposed_images_1 = '../dataset_B/norm_1/',
    path_to_decomposed_images_2 = '../dataset_B/norm_2/'
)
       
# decomposition of covid 19 class
decompose(
    path_to_features = './features/covid_features.npy',
    n_clusters = 2,
    path_to_images = '../dataset_A/Covid_19/',
    path_to_decomposed_images_1 = '../dataset_B/COVID_19_1/',
    path_to_decomposed_images_2 = '../dataset_B/COVID_19_2/'
)

# decomposition of sars class 
decompose(
    path_to_features = './features/sars_features.npy',
    n_clusters = 2,
    path_to_images = '../dataset_A/SARS/',
    path_to_decomposed_images_1 = '../dataset_B/SARS_1/',
    path_to_decomposed_images_2 = '../dataset_B/SARS_2/'
)