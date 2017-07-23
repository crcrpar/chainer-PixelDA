import chainer
import chainer.functions as F
import chainer.links as L


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
