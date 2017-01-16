from fuel.datasets import BinarizedMNIST
import numpy as np

from fuel.datasets import BinarizedMNIST
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
# from fuel.transformers.image import RandomFixedSizeCrop

def _make_stream(stream, bs):
    def new_stream():
        result = np.empty((bs, 1, 28, 28), dtype = 'float32')
        for (imb,) in stream.get_epoch_iterator():
            for i, img in enumerate(imb):
                result[i] = img
            yield (result,)
    return new_stream

def load(batch_size, test_batch_size):
    tr_data = BinarizedMNIST(which_sets=('train',))
    val_data = BinarizedMNIST(which_sets=('valid',))
    test_data = BinarizedMNIST(which_sets=('test',))

    ntrain = tr_data.num_examples
    nval = val_data.num_examples
    ntest = test_data.num_examples

    tr_scheme = ShuffledScheme(examples=ntrain, batch_size=batch_size)
    tr_stream = DataStream(tr_data, iteration_scheme=tr_scheme)

    te_scheme = SequentialScheme(examples=ntest, batch_size=test_batch_size)
    te_stream = DataStream(test_data, iteration_scheme=te_scheme)

    val_scheme = SequentialScheme(examples=nval, batch_size=batch_size)
    val_stream = DataStream(val_data, iteration_scheme=val_scheme)

    return _make_stream(tr_stream, batch_size), \
           _make_stream(val_stream, batch_size), \
           _make_stream(te_stream, test_batch_size)