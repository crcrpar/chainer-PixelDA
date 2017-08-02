# Currently only supports the case of MNIST -> MNIST-M
params = {'base_lr': 1e-3, 'dis_loss': 0.13,
          'gen_loss': 0.011, 'task_loss': 0.01, 'beta1': 0.5,
          'weight_decay': 1e-5, 'dropout_prob': 0.1, 'bn_eps': 1e-3,
          'alpha_decay_steps': 20000, 'alpha_decay_rate': 0.95}

pixel_min = -1.0
pixel_max = 1.0
