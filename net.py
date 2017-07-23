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
            self.conv1 = L.Convolution2D(None, out_channel, 3, 1, initialW=w)
            self.conv2 = L.Convolution2D(None, out_channel, 3, 1, initialW=w)
            self.bn1 = L.BatchNormalization(out_channel, use_gamma=False)
            self.bn2 = L.BatchNormalization(out_channel, use_gamma=False)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = x + self.bn1(self.conv1(h))
        return h


class Generator(chainer.Chain):
    def __init__(self, n_hidden=10, n_resblock=6, ch=64, wscale=0.02, res=28):
        super(Generator, self).__init__()
        self.ch = ch
        self.n_resblock = n_resblock
        self.n_hidden = n_hidden
        self.res = res
        w = chainer.initializers.Normal(wscale)
        with self.init_scope():
            # TODO ここの合流がマジで謎
            self.fc = L.Linear(None, self.res * self.res, initialW=w)
            self.conv1 = L.Convolution2D(None, ch, 3, 1, initialW=w)
            # TODO Improve such redundant initialization using for or something
            self.block1 = GenResBlock(ch, w)
            self.block2 = GenResBlock(ch, w)
            self.block3 = GenResBlock(ch, w)
            self.block4 = GenResBlock(ch, w)
            self.block5 = GenResBlock(ch, w)
            self.block6 = GenResBlock(ch, w)
            self.conv2 = L.Convolution2D(None, 3, 3, 1, initialW=w)

    def make_hidden(self, batchsize):
        return numpy.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)) \
            .astype(numpy.float32)

    def __call__(self, x, z):
        n_batch = x.data.shape[0]
        h = F.concat(
            (x, F.reshape(self.fc(z), (n_batch, 1, self.res, self.res))),
            axis=1)
        h = F.relu(self.conv1(h))
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
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
            self.conv = L.Convolution2D(None, out_channel, 3, 2,
                                        initialW=initialW)
            self.bn = L.BatchNormalization(out_channel, use_gamma=False)

    def __call__(self, x):
        # TODO check if the position of add_noise is OK
        return F.dropout(F.leaky_relu(add_noise(self.bn(self.conv(x)))), 0.9)


class Discriminator(chainer.Chain):
    def __init__(self, ch=512, wscale=0.02):
        super(Discriminator, self).__init__()
        # ch = 512 in mnist-m experiment
        self.ch = ch
        w = chainer.initializers.Normal(wscale)
        with self.init_scope():
            self.conv = L.Convolution2D(None, 64, 3, 1, initialW=w)
            self.bn = L.BatchNormalization(64, use_gamma=False)
            self.block1 = DisBlock(ch // 8, initialW=w)
            self.block2 = DisBlock(ch // 4, initialW=w)
            self.block3 = DisBlock(ch // 2, initialW=w)
            self.block4 = DisBlock(ch, initialW=w)
            self.fc = L.Linear(None, 1, initialW=w)

    def __call__(self, x):
        h = add_noise(x)
        h = F.dropout(F.leaky_relu(add_noise(self.bn(self.conv(h)))), 0.9)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        return self.fc(h)
