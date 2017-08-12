import chainer
import chainer.functions as F
import chainer.links as L
import numpy
from chainer import cuda


class Extractor(chainer.Chain):
    def __init__(self):
        super(Extractor, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, 32, ksize=5)

    def __call__(self, x):
        return F.max_pooling_2d(F.relu(self.conv(x)), ksize=2, stride=2)


class DigitClassifier(chainer.Chain):
    def __init__(self, n_class):
        super(DigitClassifier, self).__init__()
        self.use_source_extractor = False
        with self.init_scope():
            self.source_extractor = Extractor()
            self.generated_extractor = Extractor()
            self.conv = L.Convolution2D(None, 48, ksize=5)
            self.fc1 = L.Linear(100)
            self.fc2 = L.Linear(100)
            self.fc3 = L.Linear(n_class)

    def __call__(self, x):
        if self.use_source_extractor:
            h = self.source_extractor(x)
        else:
            h = self.generated_extractor(x)
        h = F.max_pooling_2d(F.relu(self.conv(h)), ksize=2, stride=2)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class GenResBlock(chainer.Chain):
    def __init__(self, n_out_ch, initialW, bn_eps):
        super(GenResBlock, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, n_out_ch, ksize=3, stride=1,
                                         pad=1, initialW=initialW)
            self.conv2 = L.Convolution2D(None, n_out_ch, ksize=3, stride=1,
                                         pad=1, initialW=initialW)
            self.bn1 = L.BatchNormalization(n_out_ch, eps=bn_eps)
            self.bn2 = L.BatchNormalization(n_out_ch, eps=bn_eps)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = x + self.bn2(self.conv2(h))
        return h


class Generator(chainer.Chain):
    def __init__(self, n_hidden, n_resblock, n_ch, wscale, res, bn_eps):
        super(Generator, self).__init__()
        self.n_resblock = n_resblock
        self.n_hidden = n_hidden
        self.res = res
        with self.init_scope():
            initialW = chainer.initializers.Normal(wscale)
            self.fc = L.Linear(None, self.res * self.res, initialW=initialW)
            self.conv1 = L.Convolution2D(None, n_ch, ksize=3, stride=1, pad=1,
                                         initialW=initialW)
            for i in range(1, self.n_resblock + 1):
                setattr(self, 'block{:d}'.format(i),
                        GenResBlock(n_ch, initialW, bn_eps))
            self.conv2 = L.Convolution2D(None, 3, ksize=3, stride=1, pad=1,
                                         initialW=initialW)

    def make_hidden(self, batchsize):
        return numpy.random.uniform(-1, 1, (batchsize, self.n_hidden)) \
            .astype(numpy.float32)

    def __call__(self, x, z):
        h = F.concat(
            (x, F.reshape(self.fc(z), (-1, 1, self.res, self.res))),
            axis=1)
        h = F.relu(self.conv1(h))
        for i in range(1, self.n_resblock + 1):
            h = self['block{:d}'.format(i)](h)
        return F.tanh(self.conv2(h))


def add_noise(h, sigma):
    xp = cuda.get_array_module(h.data)
    if chainer.config.train:
        return h + sigma * xp.random.randn(*h.shape)
    else:
        return h


class DisBlock(chainer.Chain):
    def __init__(self, n_out_ch, initialW, bn_eps, dr_prob, noise_sigma):
        super(DisBlock, self).__init__()
        self.dr_prob = dr_prob
        self.noise_sigma = noise_sigma
        with self.init_scope():
            self.conv = L.Convolution2D(None, n_out_ch, ksize=3, stride=2,
                                        pad=1, initialW=initialW)
            # TODO check if use_gamma=False is OK
            self.bn = L.BatchNormalization(n_out_ch, eps=bn_eps,
                                           use_gamma=False)

    def __call__(self, x):
        return add_noise(
            F.dropout(F.leaky_relu(self.bn(self.conv(x))), self.dr_prob),
            self.noise_sigma)


class Discriminator(chainer.Chain):
    def __init__(self, n_ch, wscale, bn_eps, dr_prob, noise_sigma):
        super(Discriminator, self).__init__()
        self.dr_prob = dr_prob
        self.noise_sigma = noise_sigma
        self.n_ch_list = [n_ch // 8, n_ch // 4, n_ch // 2, n_ch]
        initialW = chainer.initializers.Normal(wscale)
        with self.init_scope():
            self.conv = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1,
                                        initialW=initialW)
            for i, n_ch in enumerate(self.n_ch_list):
                setattr(self, 'block{:d}'.format(i + 1),
                        DisBlock(n_ch, initialW, bn_eps, self.dr_prob,
                                 self.noise_sigma))
            self.fc = L.Linear(None, 1, initialW=initialW)

    def __call__(self, x):
        h = add_noise(x, self.noise_sigma)  # No bn for the first input
        h = add_noise(F.dropout(F.leaky_relu(self.conv(h)), self.dr_prob),
                      self.noise_sigma)

        for i in range(1, len(self.n_ch_list) + 1):
            h = self['block{:d}'.format(i)](h)
        return self.fc(h)
