import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os

import chainer
from chainer import Variable
from chainer.dataset import convert


def out_generated_image(iterator, generator, device, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        batch = iterator.next()
        source, _ = convert.concat_examples(batch, device)
        source = chainer.Variable(source)
        xp = chainer.cuda.get_array_module(source.data)
        n_images = source.shape[0]

        noise = Variable(xp.asarray(generator.make_hidden(n_images)))
        with chainer.no_backprop_mode():
            result = generator(source, noise)

        # to cpu
        result = chainer.cuda.to_cpu(result.data)
        result = (result + 1.0) / 2.0

        n_column = 8
        n_row = n_images // n_column
        fig = plt.figure(figsize=(n_column, n_row))
        gs = gridspec.GridSpec(n_row, n_column, wspace=0.1, hspace=0.1)

        for i in range(n_images):
            r_index = i // n_column
            c_index = i % n_column
            ax = fig.add_subplot(gs[r_index, c_index])
            ax.imshow(result[i].transpose(1, 2, 0), interpolation='none')
            ax.set_xticks([])
            ax.set_yticks([])

        gs.tight_layout(fig)
        preview_dir = '{:s}/preview'.format(dst)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        plt.savefig(os.path.join(preview_dir, 'epoch{:d}.png'.format(
            trainer.updater.epoch)))

    return make_image
