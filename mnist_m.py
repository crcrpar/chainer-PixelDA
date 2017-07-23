import pickle

import numpy as np
from chainer.datasets import tuple_dataset


def get_mnist_m(dtype, withlabel=True, pklpath='mnistm_data.pkl', scale=1.0):
    assert (dtype in ['train', 'valid', 'test'])
    with open(pklpath, 'rb') as f:
        data = pickle.load(f)

    images = data[dtype]['images'].astype(np.float32)
    labels = data[dtype]['label'].astype(np.int32)

    images *= (scale / 255.0)
    if withlabel:
        return tuple_dataset.TupleDataset(images, labels)
    else:
        return images
