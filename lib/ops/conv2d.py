import lib
import lib.debug

import numpy as np
import theano
import theano.tensor as T

_default_weightnorm = False
def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

def Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=True, mask_type=None, mode = 'half', stride=1, weightnorm=None, biases=True):
    """
    inputs: tensor of shape (batch size, num channels, height, width)
    mask_type: one of None, 'a', 'b', 'hstack_a', 'hstack', 'vstack'

    returns: tensor of shape (batch size, num channels, height, width)
    """
    if mask_type is not None:
        mask_type, mask_n_channels = mask_type
        assert(mode == "half")

    if isinstance(filter_size, int): 
        filter_size = (filter_size, filter_size)

    #else it is assumed to be a tuple

    def uniform(stdev, size):
        return np.random.uniform(
            low=-stdev * np.sqrt(3),
            high=stdev * np.sqrt(3),
            size=size
        ).astype(theano.config.floatX)

    fan_in = input_dim * filter_size[0]*filter_size[1]
    fan_out = output_dim * filter_size[0]*filter_size[1]
    # TODO: shouldn't fan_out be divided by stride


    if mask_type is not None: # only approximately correct
        fan_in /= 2.
        fan_out /= 2.

    if he_init:
        filters_stdev = np.sqrt(4./(fan_in+fan_out))
    else: # Normalized init (Glorot & Bengio)
        filters_stdev = np.sqrt(2./(fan_in+fan_out))

    filter_values = uniform(
        filters_stdev,
        (output_dim, input_dim, filter_size[0], filter_size[1])
    )

    filters = lib.param(name+'.Filters', filter_values)

    if weightnorm==None:
        weightnorm = _default_weightnorm
    if weightnorm:
        norm_values = np.linalg.norm(filter_values.reshape((filter_values.shape[0], -1)), axis=1)
        norms = lib.param(
            name + '.g',
            norm_values
        )
        filters = filters * (norms / filters.reshape((filters.shape[0],-1)).norm(2, axis=1)).dimshuffle(0,'x','x','x')

    if mask_type is not None:
        mask = np.ones(
            (output_dim, input_dim, filter_size[0], filter_size[1]), 
            dtype=theano.config.floatX
        )
        center_row = filter_size[0] // 2

        center_col = filter_size[1]//2

        # Mask out future locations
        # filter shape is (out_channels, in_channels, height, width)
        if center_row == 0:
            mask[:, :, :, center_col+1:] = 0.
        elif center_col == 0:
            mask[:, :, center_row+1:, :] = 0.
        else:
            mask[:, :, center_row+1:, :] = 0.
            mask[:, :, center_row, center_col+1:] = 0.

        # Mask out future channels
        for i in xrange(mask_n_channels):
            for j in xrange(mask_n_channels):
                if ((mask_type=='a' or mask_type == 'hstack_a') and i >= j) or (mask_type=='b' and i > j):
                    mask[
                        j::mask_n_channels,
                        i::mask_n_channels,
                        center_row,
                        center_col
                    ] = 0.

        if mask_type == 'vstack':
            assert(center_col > 0 and center_row > 0)
            mask[:, :, center_row, :] = 1.

        # print mask[0,0,:,:]


        filters = filters * mask

    if biases:
        _biases = lib.param(
            name+'.Biases',
            np.zeros(output_dim, dtype=theano.config.floatX)
        )

    result = T.nnet.conv2d(
        inputs, 
        filters, 
        border_mode=mode,
        filter_flip=False,
        subsample=(stride,stride)
    )

    if biases:
        result = result + _biases[None, :, None, None]
    # result = lib.debug.print_stats(name, result)
    return result