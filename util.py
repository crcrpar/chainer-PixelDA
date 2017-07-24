import numpy as np
from opt import pixel_min
from opt import pixel_max


def gray2rgb(in_data):
    img, label = in_data
    img = np.tile(img, (3, 1, 1))
    return img, label


def scale(in_data):
    img, label = in_data
    assert (np.min(img) >= 0.0 and np.max(img) <= 1.0)
    # print("Assume each value in img is in [0,1)")
    # print("Rescale to [{:f},{:f})".format(pixel_min, pixel_max))
    img = pixel_min + (pixel_max - pixel_min) * img
    return img, label
