# Currently only supports the case of MNIST -> MNIST-M
# ch = 512 in mnist-m experiment
pixel_min = -1.0
pixel_max = 1.0

params = {
    'res_image': 28,
    'n_class': 10,
    'bn_eps': 1e-3,
    'gaussian_wscale': 0.02,
    'optimize': {'base_lr': 1e-3, 'beta1': 0.5, 'weight_decay': 1e-5,
                 'alpha_decay_steps': 20000, 'alpha_decay_rate': 0.95},
    'loss': {'gen': 0.011, 'dis': 0.13, 'task': 0.01},
    'gen': {'n_ch': 64, 'n_hidden': 10, 'n_resblock': 6},
    'dis': {'n_ch': 512, 'dropout_prob': 0.1, 'noise_sigma': 0.2}
}
