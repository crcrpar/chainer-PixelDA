import chainer
import chainer.functions as F
import chainer.links as L
import numpy
from chainer import cuda


class DigitClassifier(chainer.Chain):
    def __init__(self, n_class=10):
        super(DigitClassifier, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 32, ksize=5)
            self.conv2 = L.Convolution2D(None, 48, ksize=5)
            self.fc1 = L.Linear(100)
            self.fc2 = L.Linear(100)
            self.fc3 = L.Linear(n_class)

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=2, stride=2)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class GenResBlock(chainer.Chain):
    def __init__(self, out_channel, w):
        super(GenResBlock, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, out_channel, ksize=3, stride=1,
                                         pad=1, initialW=w)
            self.conv2 = L.Convolution2D(None, out_channel, ksize=3, stride=1,
                                         pad=1, initialW=w)
            self.bn1 = L.BatchNormalization(out_channel, use_gamma=False)
            self.bn2 = L.BatchNormalization(out_channel, use_gamma=False)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = x + self.bn2(self.conv2(h))
        return h


class Generator(chainer.Chain):
    def __init__(self, n_hidden=10, n_resblock=6, ch=64, wscale=0.02, res=28):
        super(Generator, self).__init__()
        self.ch = ch
        self.n_resblock = n_resblock
        self.n_hidden = n_hidden
        self.res = res
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.fc = L.Linear(None, self.res * self.res, initialW=w)
            self.conv1 = L.Convolution2D(None, ch, ksize=3, stride=1, pad=1,
                                         initialW=w)
            for i in range(1, self.n_resblock + 1):
                setattr(self, 'block{:d}'.format(i), GenResBlock(ch, w))
            self.conv2 = L.Convolution2D(None, 3, ksize=3, stride=1, pad=1,
                                         initialW=w)

    def make_hidden(self, batchsize):
        return numpy.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)) \
            .astype(numpy.float32)

    def __call__(self, x, z):
        n_batch = x.data.shape[0]
        h = F.concat(
            (x, F.reshape(self.fc(z), (n_batch, 1, self.res, self.res))),
            axis=1)
        h = F.relu(self.conv1(h))
        for i in range(1, self.n_resblock + 1):
            h = self['block{:d}'.format(i)](h)
        return F.tanh(self.conv2(h))


def add_noise(h, sigma=0.2):
    xp = cuda.get_array_module(h.data)
    if chainer.config.train:
        return h + sigma * xp.random.randn(*h.shape)
    else:
        return h


class DisBlock(chainer.Chain):
    def __init__(self, out_channel, initialW):
        super(DisBlock, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, out_channel, ksize=3, stride=2,
                                        pad=1, initialW=initialW)
            self.bn = L.BatchNormalization(out_channel, use_gamma=False)

    def __call__(self, x):
        # TODO check if the position of add_noise is OK
        return F.dropout(F.leaky_relu(add_noise(self.bn(self.conv(x)))), 0.9)


class Discriminator(chainer.Chain):
    def __init__(self, ch=512, wscale=0.02):
        super(Discriminator, self).__init__()
        # ch = 512 in mnist-m experiment
        self.ch = ch
        self.n_ch_list = [ch // 8, ch // 4, ch // 2, ch]
        w = chainer.initializers.Normal(wscale)
        with self.init_scope():
            self.conv = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1,
                                        initialW=w)
            self.bn = L.BatchNormalization(64, use_gamma=False)
            for i, n_ch in enumerate(self.n_ch_list):
                setattr(self, 'block{:d}'.format(i + 1), DisBlock(n_ch, w))
            self.fc = L.Linear(None, 1, initialW=w)

    def __call__(self, x):
        h = add_noise(x)
        h = F.dropout(F.leaky_relu(add_noise(self.bn(self.conv(h)))), 0.9)
        for i in range(1, len(self.n_ch_list) + 1):
            h = self['block{:d}'.format(i)](h)
        return self.fc(h)
