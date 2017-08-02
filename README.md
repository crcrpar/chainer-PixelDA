# chainer-bousmaliscvpr2017

This is chainer re-implementation of a paper, Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks [Bousmalis+, CVPR2017].


## Requirements
- Chainer 2.0+
- Numpy

<!--
## Preprocess

1. Download BSDS dataset for background ``` $ wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz```
2. Decompress the dataset ``` $ tar -zxvf BSR_bsds500.tgz```
3. Create MNIST-M dataset ``` $ python create_mnist_m.py ```

## Usage
```
from mnist_m import get_mnist_m  
train = get_mnist_m(split='train', withlabel=False)  #  Get train subset, images only, scaled to [0, 1)
test = get_mnist_m(split='test', withlabel=True, scale=255.0)  #  Get test subset, images + labels, scaled to [0, 255)
```
For more details. please refer to mnist_m.py

## Examples
``` $ python demo.py ```

![example](example.jpg)
 -->

## References
- [1]: K. Bousmalis, et al. "Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks.", in CVPR, 2017.
