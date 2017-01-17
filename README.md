# PixelVAE

Code for the models in [PixelVAE: A Latent Variable Model for Natural Images](https://arxiv.org/abs/1611.05013)

## Binarized MNIST

To train:

```
python models/mnist_pixelvae_train.py -L 12 -fs 5 -algo cond_z_bias -dpx 16 -ldim 16
```

To evaluate, take the weights of the model with best validation score from the above training procedure and then run

```
python models/mnist_pixelvae_evaluate.py -L 12 -fs 5 -algo cond_z_bias -dpx 16 -ldim 16 -w path/to/weights.pkl
```

## Real-valued MNIST, LSUN Bedrooms, 64x64 ImageNet

To train, evaluate, and generate samples:

```
python pixelvae.py
```

By default, this runs on real-valued MNIST. You can specify different datasets or model settings within `pixelvae.py`.
