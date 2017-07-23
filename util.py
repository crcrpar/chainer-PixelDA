import numpy as np

def gray2rgb(in_data):
    img, label = in_data
    img = np.tile(img, (3, 1, 1))
    return img, label
