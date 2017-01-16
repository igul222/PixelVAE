import tflib.mnist

import numpy as np

def discretize(x):
    return (x*(256-1e-8)).astype('int32')

def binarized_generator(generator, include_targets=False, n_labelled=None):
    def get_epoch():
        for data in generator():
            if n_labelled is not None:
                images, targets, labelled = data
            else:
                images, targets = data
            images = images.reshape((-1, 1, 28, 28))
            images = discretize(images)
            if include_targets:
                if n_labelled is not None:
                    yield (images, targets, labelled)
                else:
                    yield (images, targets)
            else:
                yield (images,)
    return get_epoch

def load(batch_size, test_batch_size, include_targets=False, n_labelled=None):
    train_gen, dev_gen, test_gen = tflib.mnist.load(batch_size, test_batch_size, n_labelled)
    return (
        binarized_generator(train_gen, include_targets=include_targets, n_labelled=n_labelled),
        binarized_generator(dev_gen, include_targets=include_targets, n_labelled=n_labelled),
        binarized_generator(test_gen, include_targets=include_targets, n_labelled=n_labelled)
    )