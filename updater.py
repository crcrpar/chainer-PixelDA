#!/usr/bin/env python3

import chainer
import chainer.functions as F
from chainer import Variable

from opt import params


class UPLDAGANUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis, self.cls = kwargs.pop('models')
        super(UPLDAGANUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')
        cls_optimizer = self.get_optimizer('cls')

        source_batch = self.get_iterator('main').next()
        target_batch = self.get_iterator('target').next()
        batchsize = len(target_batch)

        x_real, _ = self.converter(target_batch, self.device)
        x_real = Variable(x_real)
        xp = chainer.cuda.get_array_module(x_real.data)
        y_real = self.dis(x_real)

        source_image, source_label = [Variable(x) for x in
                                      self.converter(source_batch,
                                                     self.device)]
        noise = Variable(xp.asarray(self.gen.make_hidden(batchsize)))
        x_fake = self.gen(source_image, noise)
        y_fake = self.dis(x_fake)

        # sigmoid_cross_entropy(x,1) = softplus(-x)
        # sigmoid_cross_entropy(x,0) = softplus(x)
        loss_dis = F.sum(F.softplus(-y_real)) / batchsize
        loss_dis += F.sum(F.softplus(y_fake)) / batchsize
        loss_dis *= params['dis_loss']

        loss_gen = F.sum(F.softplus(-y_fake)) / batchsize
        loss_gen *= params['gen_loss']

        self.cls.predictor.use_source_extractor = True
        loss_cls = self.cls(source_image, source_label)
        self.cls.predictor.use_source_extractor = False
        loss_cls += self.cls(x_fake, source_label)

        loss_cls *= params['task_loss']

        """
            In the original paper, 
            During the first step, we update the discriminator and 
            task-specific parameters θD, θT, while keeping 
            the generator parameters θG fixed.
            During the second step we fix θD, θT and update θG.
            However, we swap it because of the effeciency
        """

        self.gen.cleargrads()
        loss_gen.backward()
        gen_optimizer.update()
        chainer.report({'loss': loss_gen}, self.gen)

        self.cls.cleargrads()
        loss_cls.backward()
        cls_optimizer.update()
        x_fake.unchain_backward()
        chainer.report({'loss': loss_cls}, self.cls)

        self.dis.cleargrads()
        loss_dis.backward()
        dis_optimizer.update()
        chainer.report({'loss': loss_dis}, self.dis)
