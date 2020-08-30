import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# PCA = Principal Components Analysis
def pca_coeff_latent(matrix):
    ''' 
    param:
        N x M matrix (2D numpy array)

    returns:
        coefficient:
            M x M matrix
            Each column contains coefficients for one principal component

        score : 
            The principal component scores; that is, the representation 
            of the input matrix in the principal component space. Rows of output matrix 
            correspond to observations, columns to components.

        latent:
            Vector containing eigenvalues of the 
            covariance matrix of the input matrix
    '''

    matrix_with_subtracted_mean = (matrix - np.mean(matrix.T, axis=1)).T
    [latent, coeff] = np.linalg.eig(np.cov(matrix_with_subtracted_mean))
    score = np.dot(coeff.T, matrix_with_subtracted_mean)

    return coeff, score, latent

def pca_explained_variance(matrix):
    pca = PCA()
    pca.fit(matrix)
    matrix_transformed = pca.transform(matrix)

    explained_variance = pca.explained_variance_

    return explained_variance

# To be replaced with extracted features from
# the extract_features script file
normal_images_path = '../dataset_A/normal'
noimages_normal = len([filename for filename in os.listdir(normal_images_path)])
features_normal = np.random.randn(noimages_normal, 4096)

covid19_images_path = '../dataset_A/Covid_19'
noimages_covid19 = len([filename for filename in os.listdir(covid19_images_path)])
features_covid19 = np.random.randn(noimages_covid19, 4096)

sars_images_path = '../dataset_A/SARS'
noimages_sars = len([filename for filename in os.listdir(sars_images_path)])
features_sars = np.random.randn(noimages_sars, 4096)

X = features_normal
Y = features_covid19
Z = features_sars

coeff_1, score_1, latent_1 = pca_coeff_latent(X.T)
explained_1 = pca_explained_variance(X)

coeff_2, score_2, latent_2 = pca_coeff_latent(Y.T)
explained_2 = pca_explained_variance(Y)

coeff_3, score_3, latent_3 = pca_coeff_latent(Z.T) 
explained_3 = pca_explained_variance(Z)

# reduce the dimentionality of normal features space
sum_explained = 0
idx = 0
while sum_explained <= 95:
    idx += 1
    sum_explained += explained_1[idx]

X_reduce = np.array(score_1).T[0:idx].T

# reduce the dimentionality of COVID19 features space
sum_explained = 0
idx = 0
while sum_explained <= 95:
    idx += 1
    sum_explained += explained_2[idx]

Y_reduce = np.array(score_2).T[0:idx].T

# reduce the dimentionality of SARS features space
sum_explained = 0
idx = 0
while sum_explained <= 95:
    idx += 1
    sum_explained += explained_3[idx]

Z_reduce = np.array(score_3).T[0:idx].T

plt.plot(np.cumsum(explained_1))
plt.xlabel('Number of components (Explained_1')
plt.xlim([1, 100])
plt.ylabel('Variance Explained (%)')

plt.show()

plt.plot(np.cumsum(explained_2))
plt.xlabel('Number of components (Explained_2')
plt.xlim([1, 100])
plt.ylabel('Variance Explained (%)')

plt.show()

plt.plot(np.cumsum(explained_3))
plt.xlabel('Number of components (Explained_3')
plt.xlim([1, 100])
plt.ylabel('Variance Explained (%)')

plt.show()

plt.plot(idx, 95)

plt.show()