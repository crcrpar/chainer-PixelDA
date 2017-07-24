#!/usr/bin/env python3

import argparse

import chainer
from chainer import training
from chainer.datasets import TransformDataset
from chainer.iterators import MultiprocessIterator
from chainer.training import extensions

from mnist_m import get_mnist_m
from net import DigitClassifier
from net import Discriminator
from net import Generator
from updater import UPLDAGANUpdater
from util import gray2rgb
from util import scale

from opt import params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1000,
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
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    parser.add_argument('--n_processes', type=int, default=8,
                        help='Interval of displaying log to console')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    gen = Generator()
    dis = Discriminator()
    cls = DigitClassifier()

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()
        cls.to_gpu()

    # Setup an optimizer
    def make_optimizer(model, **kwargs):
        optimizer = chainer.optimizers.Adam(**kwargs)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))
        return optimizer

    opt_gen = make_optimizer(gen, alpha=params['base_lr'], beta1=0.5)
    opt_dis = make_optimizer(dis, alpha=params['base_lr'], beta1=0.5)
    opt_cls = make_optimizer(cls, alpha=params['base_lr'], beta1=0.5)

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
    target = load_dataset(args.target, dtype='train')
    # test = load_dataset(args.target, dtype='valid', withlabel=True)
    # test = load_dataset(args.target, dtype='test', withlabel=True)

    source_iter = MultiprocessIterator(
        source, args.batchsize, n_processes=args.n_processes)
    target_iter = MultiprocessIterator(
        target, args.batchsize, n_processes=args.n_processes)

    # Set up a trainer
    updater = UPLDAGANUpdater(
        models=(gen, dis, cls),
        iterator={'main': source_iter, 'target': target_iter},
        optimizer={
            'gen': opt_gen, 'dis': opt_dis, 'cls': opt_cls},
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/loss', 'dis/loss', 'cls/loss',
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    # trainer.extend(
    #     out_generated_image(
    #         gen, dis,
    #         10, 10, args.seed, args.out),
    #     trigger=snapshot_interval)

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
