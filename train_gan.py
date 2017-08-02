#!/usr/bin/env python3

import matplotlib

matplotlib.use('Agg')

import argparse

import chainer
import chainer.links as L

from chainer import training
from chainer.datasets import TransformDataset
from chainer.iterators import MultiprocessIterator
from chainer.training import extensions

from extension import out_generated_image
from mnist_m import get_mnist_m
from net import DigitClassifier
from net import Discriminator
from net import Generator
from opt import params
from updater import UPLDAGANUpdater
from util import gray2rgb
from util import scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--source', default='mnist')
    parser.add_argument('--target', default='mnist_m')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--n_processes', type=int, default=16,
                        help='Interval of displaying log to console')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    gen = Generator()
    dis = Discriminator()
    cls = L.Classifier(DigitClassifier())

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()
        cls.to_gpu()

    # Setup an optimizer
    def make_optimizer(model):
        optimizer = chainer.optimizers.Adam(alpha=params['base_lr'],
                                            beta1=params['beta1'])
        optimizer.setup(model)
        optimizer.add_hook(
            chainer.optimizer.WeightDecay(params['weight_decay']))
        return optimizer

    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)
    opt_cls = make_optimizer(cls)

    def load_dataset(name, dtype='train'):
        if name == 'mnist':
            train, _ = chainer.datasets.get_mnist(withlabel=True, ndim=3)
            dataset = TransformDataset(train, transform=gray2rgb)
            return TransformDataset(dataset, transform=scale)
        elif name == 'mnist_m':
            dataset = get_mnist_m(dtype, withlabel=True)
            return TransformDataset(dataset, transform=scale)
        else:
            raise NotImplementedError

    source = load_dataset(args.source)
    # from chainer.datasets import split_dataset
    # source, _ = split_dataset(source, split_at=1000)

    target_train = load_dataset(args.target, dtype='train')

    source_iter = MultiprocessIterator(
        source, args.batchsize, n_processes=args.n_processes)
    target_train_iter = MultiprocessIterator(
        target_train, args.batchsize, n_processes=args.n_processes)

    # Set up a trainer
    updater = UPLDAGANUpdater(
        models=(gen, dis, cls),
        iterator={'main': source_iter, 'target': target_train_iter},
        optimizer={
            'gen': opt_gen, 'dis': opt_dis, 'cls': opt_cls},
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (1, 'epoch')
    display_interval = (10, 'iteration')

    trainer.extend(
        extensions.ExponentialShift('alpha', params['alpha_decay_rate']),
        trigger=(params['alpha_decay_steps'], 'iteration'))
    trainer.extend(
        extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/loss', 'dis/loss', 'cls/loss',
        'validation/main/accuracy'
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['gen/loss', 'dis/loss', 'cls/loss'],
                                  'iteration', trigger=(100, 'iteration'),
                                  file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(['validation/main/accuracy'],
                                  'epoch', file_name='accuracy.png'))

    # Dump examples of generated images for every epoch
    trainer.extend(out_generated_image(source_iter, gen, args.gpu, args.out))

    # Evaluate the model with the test dataset for each epoch
    target_test = load_dataset(args.target, dtype='test')
    target_test_iter = MultiprocessIterator(
        target_test, args.batchsize, n_processes=args.n_processes,
        repeat=False, shuffle=False)
    trainer.extend(
        extensions.Evaluator(target_test_iter, cls, device=args.gpu))

    # Visualize computational graph for debug
    # trainer.extend(extensions.dump_graph('gen/loss', out_name='gen.dot'))
    # trainer.extend(extensions.dump_graph('dis/loss', out_name='dis.dot'))
    # trainer.extend(extensions.dump_graph('cls/loss', out_name='cls.dot'))

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
