from __future__ import print_function

import os
import pickle as pkl

import numpy as np
from skimage import io
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data')

BST_PATH = 'BSR_bsds500.tgz'

rand = np.random.RandomState(42)

background_data = []
uncompressed_dir = 'BSR/BSDS500/data/images/train/'
for f in os.listdir(uncompressed_dir):
    bg_img = io.imread(os.path.join(uncompressed_dir, f))
    background_data.append(bg_img)


def compose_image(digit, background):
    """Difference-blend a digit and a random patch from a background image."""
    w, h, _ = background.shape
    dw, dh, _ = digit.shape
    x = np.random.randint(0, w - dw)
    y = np.random.randint(0, h - dh)

    bg = background[x:x + dw, y:y + dh]
    return np.abs(bg - digit).astype(np.uint8)


def mnist_to_img(x):
    """Binarize MNIST digit and convert to RGB."""
    x = (x > 0).astype(np.float32)
    d = x.reshape([28, 28, 1]) * 255
    return np.concatenate([d, d, d], 2)


def create_mnistm(X):
    """
    Give an array of MNIST digits, blend random background patches to
    build the MNIST-M dataset as described in
    http://jmlr.org/papers/volume17/15-239/15-239.pdf
    """
    X_ = np.zeros([X.shape[0], 28, 28, 3], np.uint8)
    for i in range(X.shape[0]):

        if i % 1000 == 0:
            print(i)

        bg_img = rand.choice(background_data)

        d = mnist_to_img(X[i])
        d = compose_image(d, bg_img)
        X_[i] = d

    return X_.transpose(0, 3, 1, 2)


train = create_mnistm(mnist.train.images)
test = create_mnistm(mnist.test.images)
valid = create_mnistm(mnist.validation.images)

# Save dataset as pickle
with open('mnistm_data.pkl', 'wb') as f:
    pkl.dump({
        'train': {'images': train, 'label': mnist.train.labels},
        'test': {'images': test, 'label': mnist.test.labels},
        'valid': {'images': valid, 'label': mnist.validation.labels}
    }, f, -1)
