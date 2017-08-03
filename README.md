# chainer-PixelDA

This is an unofficial chainer re-implementation of a paper, Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks [Bousmalis+, CVPR2017].

## Requirements
- Python 3.5+
- Chainer 2.0+
- Numpy
- Matplotlib

## Usage

### Training source-only model (training on MNIST, test on MNIST-M)
```
python train.py source_only --gpu gpuno --out directory_out
```

### Training target-only model (training on MNIST-M, test on MNIST-M)
```
python train.py target_only --gpu gpuno --out directory_out
```

### Training PixelDA model (training on MNIST-M, test on MNIST-M)
```
python train_gan.py --gpu gpuno --out directory_out
```

![generated](pixelda_result.png)

![loss](pixelda_loss.png)
![accuracy](pixelda_accuracy.png)

## Performance on MNIST -> MNIST-M

Note that this is not reproduced perfectly.

| Method | Original [1] | Ours |
|:-:|:-:|:-:|
| Source-only | 63.6 % |  60.4 % (20 epoch)|
| Target-only | 96.4 % |  95.9 % (20 epoch)|
| PixelDA | 98.2 %  |  97.6 % (100 epoch) |

## References
- [1]: K. Bousmalis, et al. "Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks.", in CVPR, 2017.
- [2]: [Original implementation](https://github.com/tensorflow/models/tree/master/domain_adaptation) 