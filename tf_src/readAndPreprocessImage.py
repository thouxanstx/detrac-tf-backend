import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def imadjust(x, a, b, c, d, gamma=1):
    return (((x - a) / (b - a)) ** gamma) * (d - c) + c

def readAndPreprocessImage(filename):
    image = Image.open(filename)
    image = np.asarray(image)
    image = imadjust(image, image.min(), image.max(), 0, 1)

    image = cv2.resize(image, (224, 224))

    RGB = np.concatenate(
        (
            image[..., np.newaxis],
            image[..., np.newaxis],
            image[..., np.newaxis]
        ), 
        axis = -1
    )

    return RGB