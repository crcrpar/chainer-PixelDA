import pickle

import numpy as np
from chainer.datasets import tuple_dataset


def get_mnist_m(split='train', withlabel=True,
                scale=1.0, dtype=np.float32, label_dtype=np.int32):
    """Dataset class for MNIST-M.
    Args:
        split ({'train', 'valid', 'test'}): Select a split of the dataset.
        withlabel (bool): If ``True``, dataset returns a tuple of an image and
            a label. Otherwise, the datasets only return an image.
        scale (float): Pixel value scale. If it is 1 (default), pixels are
            scaled to the interval ``[0, 1]``.
        dtype: Data type of resulting image arrays.
        label_dtype: Data type of the labels.
    """

    if split not in ['train', 'valid', 'test']:
        raise ValueError('please pick split from \'train\', \'test\'')

    with open('mnistm_data.pkl', 'rb') as f:
        data = pickle.load(f)
        images = data[split]['images'].astype(dtype)
        labels = data[split]['labels'].astype(label_dtype)

    images *= (scale / 255.0)
    if withlabel:
        return tuple_dataset.TupleDataset(images, labels)
    else:
        return images
