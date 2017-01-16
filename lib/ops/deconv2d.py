import lib
import lib.debug

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           host_from_gpu,
                                           gpu_contiguous, HostFromGpu,
                                           gpu_alloc_empty)
from theano.sandbox.cuda.dnn import (GpuDnnConvDesc, 
                                        GpuDnnConv, 
                                        GpuDnnConvGradI, 
                                        dnn_conv, 
                                        dnn_pool)

_default_weightnorm = False
def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

def _deconv2d(X, w, subsample=(1, 1), border_mode=(0, 0), conv_mode='conv'):
    """ 
    from Alec (https://github.com/Newmu/dcgan_code/blob/master/lib/ops.py)
    sets up dummy convolutional forward pass and uses its grad as deconv
    currently only tested/working with same padding
    """
    img = gpu_contiguous(X)
    kerns = gpu_contiguous(w)

    out = gpu_alloc_empty(
        img.shape[0], 
        kerns.shape[1], 
        img.shape[2]*subsample[0], 
        img.shape[3]*subsample[1]
    )

    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
                          conv_mode=conv_mode)

    desc = desc(
        out.shape,
        kerns.shape
    )

    d_img = GpuDnnConvGradI()(kerns, img, out, desc)

    return d_img


def Deconv2D(
    name, 
    input_dim, 
    output_dim, 
    filter_size, 
    inputs, 
    he_init=True,
    weightnorm=None,
    ):
    """
    inputs: tensor of shape (batch size, num channels, height, width)
    returns: tensor of shape (batch size, num channels, 2*height, 2*width)
    """
    def uniform(stdev, size):
        return np.random.uniform(
            low=-stdev * np.sqrt(3),
            high=stdev * np.sqrt(3),
            size=size
        ).astype(theano.config.floatX)

    filters_stdev = np.sqrt(1./(input_dim * filter_size**2))
    filters_stdev *= 2. # Because of the stride
    if he_init:
        filters_stdev *= np.sqrt(2.)

    filter_values = uniform(
        filters_stdev,
        (input_dim, output_dim, filter_size, filter_size)
    )

    filters = lib.param(
        name+'.Filters',
        filter_values
    )

    if weightnorm==None:
        weightnorm = _default_weightnorm
    if weightnorm:
        norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0,2,3)))
        norms = lib.param(
            name + '.g',
            norm_values
        )
        filters = filters * (norms / T.sqrt(T.sum(T.sqr(filters), axis=(0,2,3)))).dimshuffle('x',0,'x','x')

    biases = lib.param(
        name+'.Biases',
        np.zeros(output_dim, dtype=theano.config.floatX)
    )

    pad = (filter_size-1)/2
    result = _deconv2d(
        inputs, 
        filters, 
        subsample=(2,2),
        border_mode=(pad,pad),
    )
    result = result + biases[None, :, None, None]
    # result = lib.debug.print_stats(name, result)
    return result