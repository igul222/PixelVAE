import sys
sys.modules['theano'] = None

import numpy as np

from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
# from fuel.transformers.image import RandomFixedSizeCrop

PATH = '/home/ishaan/data/lsun_bedrooms_2727000_64px.hdf5'

from scipy.misc import imsave
def color_grid_vis(X, nh, nw, save_path):
    # from github.com/Newmu
    X = X.transpose(0,2,3,1)
    h, w = X[0].shape[:2]
    img = np.zeros((h*nh, w*nw, 3))
    for n, x in enumerate(X):
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w, :] = x
    imsave(save_path, img)


def _make_stream(stream, bs, downsample):
    def new_stream():
        if downsample:
            result = np.empty((bs, 32, 32, 3), dtype='int32')
        else:
            result = np.empty((bs, 64, 64, 3), dtype='int32')
        for (imb,) in stream.get_epoch_iterator():
            for i, img in enumerate(imb):
                if downsample:
                    a = img[:64:2, :64:2, :]
                    b = img[:64:2, 1:64:2, :]
                    c = img[1:64:2, :64:2, :]
                    d = img[1:64:2, 1:64:2, :]
                    result[i] = a
                    result[i] += b
                    result[i] += c
                    result[i] += d
                    result[i] /= 4                    
                    # print (a+b+c+d).dtype
                    # raise Exception()
                    # result[i] =  (a+b+c+d)/4
                else:
                    result[i] =  img[:64, :64, :]                
            # print "warning overfit mode"
            # color_grid_vis(result.transpose(0,3,1,2)[:,:3,:,:], 2, 2, 'reals.png')
            # while True:
            yield (result.transpose(0,3,1,2),)
            # yield (result.transpose(0,3,1,2)[:,:3,:,:],)
    return new_stream

def load(batch_size=128, downsample=True):
    tr_data = H5PYDataset(PATH, which_sets=('train',))
    te_data = H5PYDataset(PATH, which_sets=('valid',))

    ntrain = tr_data.num_examples
    # ntest = te_data.num_examples
    nval = te_data.num_examples

    # print "ntrain {}, nval {}".format(ntrain, nval)

    tr_scheme = ShuffledScheme(examples=ntrain, batch_size=batch_size)
    tr_stream = DataStream(tr_data, iteration_scheme=tr_scheme)

    # te_scheme = SequentialScheme(examples=ntest, batch_size=batch_size)
    # te_stream = DataStream(te_data, iteration_scheme=te_scheme)

    val_scheme = SequentialScheme(examples=nval, batch_size=batch_size)
    val_stream = DataStream(tr_data, iteration_scheme=val_scheme)

    return _make_stream(tr_stream, batch_size, downsample), _make_stream(val_stream, batch_size, downsample)
    # return (
    #     (lambda: tr_stream.get_epoch_iterator()),
    #     (lambda: val_stream.get_epoch_iterator()),
    #     # (lambda: te_stream.get_epoch_iterator())
    # )
