"""
PixelVAE: A Latent Variable Model for Natural Images
Ishaan Gulrajani, Kundan Kumar, Faruk Ahmed, Adrien Ali Taiga, Francesco Visin, David Vazquez, Aaron Courville
"""

import os, sys
sys.path.append(os.getcwd())

N_GPUS = 1

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(tf=True, n_gpus=N_GPUS)
except ImportError:
    pass

import tflib as lib
import tflib.train_loop_2
import tflib.ops.kl_unit_gaussian
import tflib.ops.kl_gaussian_gaussian
import tflib.ops.conv2d
import tflib.ops.linear
import tflib.ops.batchnorm
import tflib.ops.embedding

import tflib.lsun_bedrooms
import tflib.mnist_256
import tflib.small_imagenet

import numpy as np
import tensorflow as tf
import scipy.misc
from scipy.misc import imsave

import time
import functools

DATASET = 'mnist_256' # mnist_256, lsun_32, lsun_64, imagenet_64
SETTINGS = 'mnist_256' # mnist_256, 32px_small, 32px_big, 64px_small, 64px_big

if SETTINGS == 'mnist_256':
    # two_level uses Enc1/Dec1 for the bottom level, Enc2/Dec2 for the top level
    # one_level uses EncFull/DecFull for the bottom (and only) level
    MODE = 'one_level'

    # Whether to treat pixel inputs to the model as real-valued (as in the 
    # original PixelCNN) or discrete (gets better likelihoods).
    EMBED_INPUTS = True

    # Turn on/off the bottom-level PixelCNN in Dec1/DecFull
    PIXEL_LEVEL_PIXCNN = True
    HIGHER_LEVEL_PIXCNN = True

    DIM_EMBED    = 16
    DIM_PIX_1    = 32
    DIM_1        = 16
    DIM_2        = 32
    DIM_3        = 32
    DIM_4        = 64
    LATENT_DIM_2 = 128

    ALPHA1_ITERS = 5000
    ALPHA2_ITERS = 5000
    KL_PENALTY = 1.0
    BETA_ITERS = 1000

    # In Dec2, we break each spatial location into N blocks (analogous to channels
    # in the original PixelCNN) and model each spatial location autoregressively
    # as P(x)=P(x0)*P(x1|x0)*P(x2|x0,x1)... In my experiments values of N > 1
    # actually hurt performance. Unsure why; might be a bug.
    PIX_2_N_BLOCKS = 1

    TIMES = {
        'test_every': 2*500,
        'stop_after': 500*500,
        'callback_every': 10*500
    }

    LR = 1e-3

    LR_DECAY_AFTER = TIMES['stop_after']
    LR_DECAY_FACTOR = 1.

    BATCH_SIZE = 100
    N_CHANNELS = 1
    HEIGHT = 28
    WIDTH = 28

    # These aren't actually (typically) used for one-level models but some parts
    # of the code still depend on them being defined.
    LATENT_DIM_1 = 64
    LATENTS1_HEIGHT = 7
    LATENTS1_WIDTH = 7

elif SETTINGS == '32px_small':
    MODE = 'two_level'

    EMBED_INPUTS = True

    PIXEL_LEVEL_PIXCNN = True
    HIGHER_LEVEL_PIXCNN = True

    DIM_EMBED    = 16
    DIM_PIX_1    = 128
    DIM_1        = 64
    DIM_2        = 128
    DIM_3        = 256
    LATENT_DIM_1 = 64
    DIM_PIX_2    = 512
    DIM_4        = 512
    LATENT_DIM_2 = 512

    ALPHA1_ITERS = 2000
    ALPHA2_ITERS = 5000
    KL_PENALTY = 1.00
    BETA_ITERS = 1000

    PIX_2_N_BLOCKS = 1

    TIMES = {
        'test_every': 1000,
        'stop_after': 200000,
        'callback_every': 20000
    }

    LR = 1e-3

    LR_DECAY_AFTER = 180000
    LR_DECAY_FACTOR = 1e-1

    BATCH_SIZE = 64
    N_CHANNELS = 3
    HEIGHT = 32
    WIDTH = 32

    LATENTS1_HEIGHT = 8
    LATENTS1_WIDTH = 8

elif SETTINGS == '32px_big':

    MODE = 'two_level'

    EMBED_INPUTS = False

    PIXEL_LEVEL_PIXCNN = True
    HIGHER_LEVEL_PIXCNN = True

    DIM_EMBED    = 16
    DIM_PIX_1    = 256
    DIM_1        = 128
    DIM_2        = 256
    DIM_3        = 512
    LATENT_DIM_1 = 128
    DIM_PIX_2    = 512
    DIM_4        = 512
    LATENT_DIM_2 = 512

    ALPHA1_ITERS = 2000
    ALPHA2_ITERS = 5000
    KL_PENALTY = 1.00
    BETA_ITERS = 1000

    PIX_2_N_BLOCKS = 1

    TIMES = {
        'test_every': 1000,
        'stop_after': 300000,
        'callback_every': 20000
    }

    VANILLA = False
    LR = 1e-3

    LR_DECAY_AFTER = 300000
    LR_DECAY_FACTOR = 1e-1

    BATCH_SIZE = 64
    N_CHANNELS = 3
    HEIGHT = 32
    WIDTH = 32
    LATENTS1_HEIGHT = 8
    LATENTS1_WIDTH = 8

elif SETTINGS == '64px_small':
    MODE = 'two_level'

    EMBED_INPUTS = True

    PIXEL_LEVEL_PIXCNN = True
    HIGHER_LEVEL_PIXCNN = True

    DIM_EMBED    = 16
    DIM_PIX_1    = 128
    DIM_0        = 64
    DIM_1        = 64
    DIM_2        = 128
    LATENT_DIM_1 = 64
    DIM_PIX_2    = 256
    DIM_3        = 256
    DIM_4        = 512
    LATENT_DIM_2 = 512

    PIX_2_N_BLOCKS = 1

    TIMES = {
        'test_every': 10000,
        'stop_after': 200000,
        'callback_every': 50000
    }

    VANILLA = False
    LR = 1e-3

    LR_DECAY_AFTER = 180000
    LR_DECAY_FACTOR = .1

    ALPHA1_ITERS = 2000
    ALPHA2_ITERS = 10000
    KL_PENALTY = 1.0
    BETA_ITERS = 1000

    BATCH_SIZE = 64
    N_CHANNELS = 3
    HEIGHT = 64
    WIDTH = 64
    LATENTS1_WIDTH = 16
    LATENTS1_HEIGHT = 16

elif SETTINGS == '64px_big':
    MODE = 'two_level'

    EMBED_INPUTS = True

    PIXEL_LEVEL_PIXCNN = True
    HIGHER_LEVEL_PIXCNN = True

    DIM_EMBED    = 16
    DIM_PIX_1    = 384
    DIM_0        = 192
    DIM_1        = 256
    DIM_2        = 512
    LATENT_DIM_1 = 64
    DIM_PIX_2    = 512
    DIM_3        = 512
    DIM_4        = 512
    LATENT_DIM_2 = 512

    PIX_2_N_BLOCKS = 1

    TIMES = {
        'test_every': 10000,
        'stop_after': 400000,
        'callback_every': 50000
    }

    VANILLA = False
    LR = 1e-3

    LR_DECAY_AFTER = 180000
    LR_DECAY_FACTOR = .5

    ALPHA1_ITERS = 1000
    ALPHA2_ITERS = 10000
    KL_PENALTY = 1.00
    BETA_ITERS = 500

    BATCH_SIZE = 48
    N_CHANNELS = 3
    HEIGHT = 64
    WIDTH = 64
    LATENTS1_WIDTH = 16
    LATENTS1_HEIGHT = 16

if DATASET == 'mnist_256':
    train_data, dev_data, test_data = lib.mnist_256.load(BATCH_SIZE, BATCH_SIZE)
elif DATASET == 'lsun_32':
    train_data, dev_data = lib.lsun_bedrooms.load(BATCH_SIZE, downsample=True)
elif DATASET == 'lsun_64':
    train_data, dev_data = lib.lsun_bedrooms.load(BATCH_SIZE, downsample=False)
elif DATASET == 'imagenet_64':
    train_data, dev_data = lib.small_imagenet.load(BATCH_SIZE)

lib.print_model_settings(locals().copy())

DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]

lib.ops.conv2d.enable_default_weightnorm()
lib.ops.linear.enable_default_weightnorm()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    bn_is_training = tf.placeholder(tf.bool, shape=None, name='bn_is_training')
    bn_stats_iter = tf.placeholder(tf.int32, shape=None, name='bn_stats_iter')
    total_iters = tf.placeholder(tf.int32, shape=None, name='total_iters')
    all_images = tf.placeholder(tf.int32, shape=[None, N_CHANNELS, HEIGHT, WIDTH], name='all_images')
    all_latents1 = tf.placeholder(tf.float32, shape=[None, LATENT_DIM_1, LATENTS1_HEIGHT, LATENTS1_WIDTH], name='all_latents1')

    split_images = tf.split(0, len(DEVICES), all_images)
    split_latents1 = tf.split(0, len(DEVICES), all_latents1)

    tower_cost = []
    tower_outputs1_sample = []

    for device_index, (device, images, latents1_sample) in enumerate(zip(DEVICES, split_images, split_latents1)):
        with tf.device(device):

            def nonlinearity(x):
                return tf.nn.elu(x)

            def pixcnn_gated_nonlinearity(a, b):
                return tf.sigmoid(a) * tf.tanh(b)

            def SubpixelConv2D(*args, **kwargs):
                kwargs['output_dim'] = 4*kwargs['output_dim']
                output = lib.ops.conv2d.Conv2D(*args, **kwargs)
                output = tf.transpose(output, [0,2,3,1])
                output = tf.depth_to_space(output, 2)
                output = tf.transpose(output, [0,3,1,2])
                return output

            def ResidualBlock(name, input_dim, output_dim, inputs, filter_size, mask_type=None, resample=None, he_init=True):
                """
                resample: None, 'down', or 'up'
                """
                if mask_type != None and resample != None:
                    raise Exception('Unsupported configuration')

                if resample=='down':
                    conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D, stride=2)
                    conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
                    conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim, stride=2)
                elif resample=='up':
                    conv_shortcut = SubpixelConv2D
                    conv_1        = functools.partial(SubpixelConv2D, input_dim=input_dim, output_dim=output_dim)
                    conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
                elif resample==None:
                    conv_shortcut = lib.ops.conv2d.Conv2D
                    conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
                    conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
                else:
                    raise Exception('invalid resample value')

                if output_dim==input_dim and resample==None:
                    shortcut = inputs # Identity skip-connection
                else:
                    shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, mask_type=mask_type, he_init=False, biases=True, inputs=inputs)

                output = inputs
                if mask_type == None:
                    output = nonlinearity(output)
                    output = conv_1(name+'.Conv1', filter_size=filter_size, mask_type=mask_type, inputs=output, he_init=he_init, weightnorm=False)
                    output = nonlinearity(output)
                    output = conv_2(name+'.Conv2', filter_size=filter_size, mask_type=mask_type, inputs=output, he_init=he_init, weightnorm=False, biases=False)
                    if device_index == 0:
                        output = lib.ops.batchnorm.Batchnorm(name+'.BN', [0,2,3], output, bn_is_training, bn_stats_iter)
                    else:
                        output = lib.ops.batchnorm.Batchnorm(name+'.BN', [0,2,3], output, bn_is_training, bn_stats_iter, update_moving_stats=False)
                else:
                    output = nonlinearity(output)
                    output_a = conv_1(name+'.Conv1A', filter_size=filter_size, mask_type=mask_type, inputs=output, he_init=he_init)
                    output_b = conv_1(name+'.Conv1B', filter_size=filter_size, mask_type=mask_type, inputs=output, he_init=he_init)
                    output = pixcnn_gated_nonlinearity(output_a, output_b)
                    output = conv_2(name+'.Conv2', filter_size=filter_size, mask_type=mask_type, inputs=output, he_init=he_init)

                return shortcut + output

            def Enc1(images):
                output = images

                if WIDTH == 64:
                    if EMBED_INPUTS:
                        output = lib.ops.conv2d.Conv2D('Enc1.Input', input_dim=N_CHANNELS*DIM_EMBED, output_dim=DIM_0, filter_size=1, inputs=output, he_init=False)
                        output = ResidualBlock('Enc1.InputRes0', input_dim=DIM_0, output_dim=DIM_0, filter_size=3, resample=None, inputs=output)
                        output = ResidualBlock('Enc1.InputRes', input_dim=DIM_0, output_dim=DIM_1, filter_size=3, resample='down', inputs=output)
                    else:
                        output = lib.ops.conv2d.Conv2D('Enc1.Input', input_dim=N_CHANNELS, output_dim=DIM_1, filter_size=1, inputs=output, he_init=False)
                        output = ResidualBlock('Enc1.InputRes', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, resample='down', inputs=output)
                else:
                    if EMBED_INPUTS:
                        output = lib.ops.conv2d.Conv2D('Enc1.Input', input_dim=N_CHANNELS*DIM_EMBED, output_dim=DIM_1, filter_size=1, inputs=output, he_init=False)
                    else:
                        output = lib.ops.conv2d.Conv2D('Enc1.Input', input_dim=N_CHANNELS, output_dim=DIM_1, filter_size=1, inputs=output, he_init=False)


                output = ResidualBlock('Enc1.Res1Pre', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, resample=None, inputs=output)
                output = ResidualBlock('Enc1.Res1Pre2', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, resample=None, inputs=output)
                output = ResidualBlock('Enc1.Res1', input_dim=DIM_1, output_dim=DIM_2, filter_size=3, resample='down', inputs=output)
                if LATENTS1_WIDTH == 16:
                    output = ResidualBlock('Enc1.Res4Pre', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('Enc1.Res4', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('Enc1.Res4Post', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, inputs=output)
                    mu_and_sigma = lib.ops.conv2d.Conv2D('Enc1.Out', input_dim=DIM_2, output_dim=2*LATENT_DIM_1, filter_size=1, inputs=output, he_init=False)
                else:
                    output = ResidualBlock('Enc1.Res2Pre', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('Enc1.Res2Pre2', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('Enc1.Res2', input_dim=DIM_2, output_dim=DIM_3, filter_size=3, resample='down', inputs=output)
                    output = ResidualBlock('Enc1.Res3Pre', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('Enc1.Res3Pre2', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('Enc1.Res3Pre3', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, inputs=output)
                    mu_and_sigma = lib.ops.conv2d.Conv2D('Enc1.Out', input_dim=DIM_3, output_dim=2*LATENT_DIM_1, filter_size=1, inputs=output, he_init=False)

                return mu_and_sigma, output

            def Dec1(latents, images):
                output = tf.clip_by_value(latents, -50., 50.)

                if LATENTS1_WIDTH == 16:
                    output = lib.ops.conv2d.Conv2D('Dec1.Input', input_dim=LATENT_DIM_1, output_dim=DIM_2, filter_size=1, inputs=output, he_init=False)
                    output = ResidualBlock('Dec1.Res1A', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('Dec1.Res1B', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('Dec1.Res1C', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, inputs=output)
                else:
                    output = lib.ops.conv2d.Conv2D('Dec1.Input', input_dim=LATENT_DIM_1, output_dim=DIM_3, filter_size=1, inputs=output, he_init=False)
                    output = ResidualBlock('Dec1.Res1', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('Dec1.Res1Post', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('Dec1.Res1Post2', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('Dec1.Res2', input_dim=DIM_3, output_dim=DIM_2, filter_size=3, resample='up', inputs=output)
                    output = ResidualBlock('Dec1.Res2Post', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, inputs=output)
                    output = ResidualBlock('Dec1.Res2Post2', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, inputs=output)

                output = ResidualBlock('Dec1.Res3', input_dim=DIM_2, output_dim=DIM_1, filter_size=3, resample='up', inputs=output)
                output = ResidualBlock('Dec1.Res3Post', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, resample=None, inputs=output)
                output = ResidualBlock('Dec1.Res3Post2', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, resample=None, inputs=output)

                if WIDTH == 64:
                    output = ResidualBlock('Dec1.Res4', input_dim=DIM_1, output_dim=DIM_0, filter_size=3, resample='up', inputs=output)
                    output = ResidualBlock('Dec1.Res4Post', input_dim=DIM_0, output_dim=DIM_0, filter_size=3, resample=None, inputs=output)

                if PIXEL_LEVEL_PIXCNN:

                    if WIDTH == 64:
                        if EMBED_INPUTS:
                            masked_images = lib.ops.conv2d.Conv2D('Dec1.Pix1', input_dim=N_CHANNELS*DIM_EMBED, output_dim=DIM_0, filter_size=5, inputs=images, mask_type=('a', N_CHANNELS), he_init=False)
                        else:
                            masked_images = lib.ops.conv2d.Conv2D('Dec1.Pix1', input_dim=N_CHANNELS, output_dim=DIM_0, filter_size=5, inputs=images, mask_type=('a', N_CHANNELS), he_init=False)
                    else:
                        if EMBED_INPUTS:
                            masked_images = lib.ops.conv2d.Conv2D('Dec1.Pix1', input_dim=N_CHANNELS*DIM_EMBED, output_dim=DIM_1, filter_size=5, inputs=images, mask_type=('a', N_CHANNELS), he_init=False)
                        else:
                            masked_images = lib.ops.conv2d.Conv2D('Dec1.Pix1', input_dim=N_CHANNELS, output_dim=DIM_1, filter_size=5, inputs=images, mask_type=('a', N_CHANNELS), he_init=False)

                    # Make the variance of output and masked_images (roughly) match
                    output /= 2

                    # Warning! Because of the masked convolutions it's very important that masked_images comes first in this concat
                    output = tf.concat(1, [masked_images, output])

                    if WIDTH == 64:
                        output = ResidualBlock('Dec1.Pix2Res', input_dim=2*DIM_0, output_dim=DIM_PIX_1, filter_size=3, mask_type=('b', N_CHANNELS), inputs=output)
                        output = ResidualBlock('Dec1.Pix3Res', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, mask_type=('b', N_CHANNELS), inputs=output)
                        output = ResidualBlock('Dec1.Pix4Res', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, mask_type=('b', N_CHANNELS), inputs=output)
                    else:
                        output = ResidualBlock('Dec1.Pix2Res', input_dim=2*DIM_1, output_dim=DIM_PIX_1, filter_size=3, mask_type=('b', N_CHANNELS), inputs=output)
                        output = ResidualBlock('Dec1.Pix3Res', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, mask_type=('b', N_CHANNELS), inputs=output)

                    output = lib.ops.conv2d.Conv2D('Dec1.Out', input_dim=DIM_PIX_1, output_dim=256*N_CHANNELS, filter_size=1, mask_type=('b', N_CHANNELS), he_init=False, inputs=output)

                else:

                    if WIDTH == 64:
                        output = lib.ops.conv2d.Conv2D('Dec1.Out', input_dim=DIM_0, output_dim=256*N_CHANNELS, filter_size=1, he_init=False, inputs=output)
                    else:
                        output = lib.ops.conv2d.Conv2D('Dec1.Out', input_dim=DIM_1, output_dim=256*N_CHANNELS, filter_size=1, he_init=False, inputs=output)

                return tf.transpose(
                    tf.reshape(output, [-1, 256, N_CHANNELS, HEIGHT, WIDTH]),
                    [0,2,3,4,1]
                )

            def Enc2(h1):
                output = h1

                if LATENTS1_WIDTH == 16:
                    output = ResidualBlock('Enc2.Res0', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, he_init=True, inputs=output)
                    output = ResidualBlock('Enc2.Res1Pre', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, he_init=True, inputs=output)
                    output = ResidualBlock('Enc2.Res1Pre2', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, he_init=True, inputs=output)
                    output = ResidualBlock('Enc2.Res1', input_dim=DIM_2, output_dim=DIM_3, filter_size=3, resample='down', he_init=True, inputs=output)

                output = ResidualBlock('Enc2.Res2Pre', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, he_init=True, inputs=output)
                output = ResidualBlock('Enc2.Res2Pre2', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, he_init=True, inputs=output)
                output = ResidualBlock('Enc2.Res2Pre3', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, he_init=True, inputs=output)
                output = ResidualBlock('Enc2.Res1A', input_dim=DIM_3, output_dim=DIM_4, filter_size=3, resample='down', he_init=True, inputs=output)
                output = ResidualBlock('Enc2.Res2PreA', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, resample=None, he_init=True, inputs=output)
                output = ResidualBlock('Enc2.Res2', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, resample=None, he_init=True, inputs=output)
                output = ResidualBlock('Enc2.Res2Post', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, resample=None, he_init=True, inputs=output)

                output = tf.reshape(output, [-1, 4*4*DIM_4])
                output = lib.ops.linear.Linear('Enc2.Output', input_dim=4*4*DIM_4, output_dim=2*LATENT_DIM_2, inputs=output)

                return output

            def Dec2(latents, targets):
                output = tf.clip_by_value(latents, -50., 50.)
                output = lib.ops.linear.Linear('Dec2.Input', input_dim=LATENT_DIM_2, output_dim=4*4*DIM_4, inputs=output)

                output = tf.reshape(output, [-1, DIM_4, 4, 4])

                output = ResidualBlock('Dec2.Res1Pre', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, resample=None, he_init=True, inputs=output)
                output = ResidualBlock('Dec2.Res1', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, resample=None, he_init=True, inputs=output)
                output = ResidualBlock('Dec2.Res1Post', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, resample=None, he_init=True, inputs=output)
                output = ResidualBlock('Dec2.Res3', input_dim=DIM_4, output_dim=DIM_3, filter_size=3, resample='up', he_init=True, inputs=output)
                output = ResidualBlock('Dec2.Res3Post', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, he_init=True, inputs=output)
                output = ResidualBlock('Dec2.Res3Post2', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, he_init=True, inputs=output)
                output = ResidualBlock('Dec2.Res3Post3', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, he_init=True, inputs=output)

                if LATENTS1_WIDTH == 16:
                    output = ResidualBlock('Dec2.Res3Post5', input_dim=DIM_3, output_dim=DIM_2, filter_size=3, resample='up', he_init=True, inputs=output)
                    output = ResidualBlock('Dec2.Res3Post6', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, he_init=True, inputs=output)
                    output = ResidualBlock('Dec2.Res3Post7', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, he_init=True, inputs=output)
                    output = ResidualBlock('Dec2.Res3Post8', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, he_init=True, inputs=output)

                if HIGHER_LEVEL_PIXCNN:

                    if LATENTS1_WIDTH == 16:
                        masked_targets = lib.ops.conv2d.Conv2D('Dec2.Pix1', input_dim=LATENT_DIM_1, output_dim=DIM_2, filter_size=5, mask_type=('a', PIX_2_N_BLOCKS), he_init=False, inputs=targets)
                    else:
                        masked_targets = lib.ops.conv2d.Conv2D('Dec2.Pix1', input_dim=LATENT_DIM_1, output_dim=DIM_3, filter_size=5, mask_type=('a', PIX_2_N_BLOCKS), he_init=False, inputs=targets)

                    # Make the variance of output and masked_targets roughly match
                    output /= 2

                    output = tf.concat(1, [masked_targets, output])

                    if LATENTS1_WIDTH == 16:
                        output = ResidualBlock('Dec2.Pix2Res', input_dim=2*DIM_2, output_dim=DIM_PIX_2, filter_size=3, mask_type=('b', PIX_2_N_BLOCKS), he_init=True, inputs=output)
                    else:
                        output = ResidualBlock('Dec2.Pix2Res', input_dim=2*DIM_3, output_dim=DIM_PIX_2, filter_size=3, mask_type=('b', PIX_2_N_BLOCKS), he_init=True, inputs=output)
                    output = ResidualBlock('Dec2.Pix3Res', input_dim=DIM_PIX_2, output_dim=DIM_PIX_2, filter_size=3, mask_type=('b', PIX_2_N_BLOCKS), he_init=True, inputs=output)
                    output = ResidualBlock('Dec2.Pix4Res', input_dim=DIM_PIX_2, output_dim=DIM_PIX_2, filter_size=1, mask_type=('b', PIX_2_N_BLOCKS), he_init=True, inputs=output)

                    output = lib.ops.conv2d.Conv2D('Dec2.Out', input_dim=DIM_PIX_2, output_dim=2*LATENT_DIM_1, filter_size=1, mask_type=('b', PIX_2_N_BLOCKS), he_init=False, inputs=output)

                else:

                    if LATENTS1_WIDTH == 16:
                        output = lib.ops.conv2d.Conv2D('Dec2.Out', input_dim=DIM_2, output_dim=2*LATENT_DIM_1, filter_size=1, mask_type=('b', PIX_2_N_BLOCKS), he_init=False, inputs=output)
                    else:
                        output = lib.ops.conv2d.Conv2D('Dec2.Out', input_dim=DIM_3, output_dim=2*LATENT_DIM_1, filter_size=1, mask_type=('b', PIX_2_N_BLOCKS), he_init=False, inputs=output)

                return output

            # Really only for MNIST. Will require modification for other datasets.
            def EncFull(images):
                output = images

                if EMBED_INPUTS:
                    output = lib.ops.conv2d.Conv2D('EncFull.Input', input_dim=N_CHANNELS*DIM_EMBED, output_dim=DIM_1, filter_size=1, inputs=output, he_init=False)
                else:
                    output = lib.ops.conv2d.Conv2D('EncFull.Input', input_dim=N_CHANNELS, output_dim=DIM_1, filter_size=1, inputs=output, he_init=False)

                output = ResidualBlock('EncFull.Res1', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, resample=None, inputs=output)
                output = ResidualBlock('EncFull.Res2', input_dim=DIM_1, output_dim=DIM_2, filter_size=3, resample='down', inputs=output)
                output = ResidualBlock('EncFull.Res3', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, inputs=output)
                output = ResidualBlock('EncFull.Res4', input_dim=DIM_2, output_dim=DIM_3, filter_size=3, resample='down', inputs=output)
                output = ResidualBlock('EncFull.Res5', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, inputs=output)
                output = ResidualBlock('EncFull.Res6', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, inputs=output)

                output = tf.reduce_mean(output, reduction_indices=[2,3])
                output = lib.ops.linear.Linear('EncFull.Output', input_dim=DIM_3, output_dim=2*LATENT_DIM_2, initialization='glorot', inputs=output)

                return output

            # Really only for MNIST. Will require modification for other datasets.
            def DecFull(latents, images):
                output = tf.clip_by_value(latents, -50., 50.)

                output = lib.ops.linear.Linear('DecFull.Input', input_dim=LATENT_DIM_2, output_dim=DIM_3, initialization='glorot', inputs=output)
                output = tf.reshape(tf.tile(tf.reshape(output, [-1, DIM_3, 1]), [1, 1, 49]), [-1, DIM_3, 7, 7])

                output = ResidualBlock('DecFull.Res2', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, he_init=True, inputs=output)
                output = ResidualBlock('DecFull.Res3', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, he_init=True, inputs=output)
                output = ResidualBlock('DecFull.Res4', input_dim=DIM_3, output_dim=DIM_2, filter_size=3, resample='up', he_init=True, inputs=output)
                output = ResidualBlock('DecFull.Res5', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, he_init=True, inputs=output)
                output = ResidualBlock('DecFull.Res6', input_dim=DIM_2, output_dim=DIM_1, filter_size=3, resample='up', he_init=True, inputs=output)
                output = ResidualBlock('DecFull.Res7', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, resample=None, he_init=True, inputs=output)

                if PIXEL_LEVEL_PIXCNN:

                    if EMBED_INPUTS:
                        masked_images = lib.ops.conv2d.Conv2D('DecFull.Pix1', input_dim=N_CHANNELS*DIM_EMBED, output_dim=DIM_1, filter_size=5, inputs=images, mask_type=('a', N_CHANNELS), he_init=False)
                    else:
                        masked_images = lib.ops.conv2d.Conv2D('DecFull.Pix1', input_dim=N_CHANNELS, output_dim=DIM_1, filter_size=5, inputs=images, mask_type=('a', N_CHANNELS), he_init=False)

                    # Warning! Because of the masked convolutions it's very important that masked_images comes first in this concat

                    output = tf.concat(1, [masked_images, output])

                    # output = ResidualBlock('DecFull.Pix2Res', input_dim=2*DIM_1,   output_dim=DIM_PIX_1, filter_size=1, mask_type=('b', N_CHANNELS),          inputs=output)

                    output = ResidualBlock('DecFull.Pix2Res', input_dim=2*DIM_1,   output_dim=DIM_PIX_1, filter_size=3, mask_type=('b', N_CHANNELS),          inputs=output)
                    output = ResidualBlock('DecFull.Pix3Res', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, mask_type=('b', N_CHANNELS), inputs=output)
                    output = ResidualBlock('DecFull.Pix4Res', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, mask_type=('b', N_CHANNELS), inputs=output)
                    output = ResidualBlock('DecFull.Pix5Res', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, mask_type=('b', N_CHANNELS), inputs=output)

                    output = lib.ops.conv2d.Conv2D('Dec1.Out', input_dim=DIM_PIX_1, output_dim=256*N_CHANNELS, filter_size=1, mask_type=('b', N_CHANNELS), he_init=False, inputs=output)

                else:

                    output = lib.ops.conv2d.Conv2D('Dec1.Out', input_dim=DIM_1, output_dim=256*N_CHANNELS, filter_size=1, he_init=False, inputs=output)

                return tf.transpose(
                    tf.reshape(output, [-1, 256, N_CHANNELS, HEIGHT, WIDTH]),
                    [0,2,3,4,1]
                )

            def split(mu_and_logsig):
                mu, logsig = tf.split(1, 2, mu_and_logsig)
                sig = 0.5 * (tf.nn.softsign(logsig)+1)
                logsig = tf.log(sig)
                return mu, logsig, sig
         
            def clamp_logsig_and_sig(logsig, sig):
                # Early during training (see BETA_ITERS), stop sigma from going too low
                floor = 1. - tf.minimum(1., tf.cast(total_iters, 'float32') / BETA_ITERS)
                log_floor = tf.log(floor)
                return tf.maximum(logsig, log_floor), tf.maximum(sig, floor)


            scaled_images = (tf.cast(images, 'float32') - 128.) / 64.
            if EMBED_INPUTS:
                embedded_images = lib.ops.embedding.Embedding('Embedding', 256, DIM_EMBED, images)
                embedded_images = tf.transpose(embedded_images, [0,4,1,2,3])
                embedded_images = tf.reshape(embedded_images, [-1, DIM_EMBED*N_CHANNELS, HEIGHT, WIDTH])

            if MODE == 'one_level':

                # Layer 1

                if EMBED_INPUTS:
                    mu_and_logsig1 = EncFull(embedded_images)
                else:
                    mu_and_logsig1 = EncFull(scaled_images)
                mu1, logsig1, sig1 = split(mu_and_logsig1)

                eps = tf.random_normal(tf.shape(mu1))
                latents1 = mu1 + (eps * sig1)

                if EMBED_INPUTS:
                    outputs1 = DecFull(latents1, embedded_images)
                else:
                    outputs1 = DecFull(latents1, scaled_images)

                reconst_cost = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        tf.reshape(outputs1, [-1, 256]),
                        tf.reshape(images, [-1])
                    )
                )

                # Assembly

                # An alpha of exactly 0 can sometimes cause inf/nan values, so we're
                # careful to avoid it.
                alpha = tf.minimum(1., tf.cast(total_iters+1, 'float32') / ALPHA1_ITERS) * KL_PENALTY

                kl_cost_1 = tf.reduce_mean(
                    lib.ops.kl_unit_gaussian.kl_unit_gaussian(
                        mu1, 
                        logsig1,
                        sig1
                    )
                )

                kl_cost_1 *= float(LATENT_DIM_2) / (N_CHANNELS * WIDTH * HEIGHT)

                cost = reconst_cost + (alpha * kl_cost_1)

            elif MODE == 'two_level':
                # Layer 1

                if EMBED_INPUTS:
                    mu_and_logsig1, h1 = Enc1(embedded_images)
                else:
                    mu_and_logsig1, h1 = Enc1(scaled_images)
                mu1, logsig1, sig1 = split(mu_and_logsig1)

                if mu1.get_shape().as_list()[2] != LATENTS1_HEIGHT:
                    raise Exception("LATENTS1_HEIGHT doesn't match mu1 shape!")
                if mu1.get_shape().as_list()[3] != LATENTS1_WIDTH:
                    raise Exception("LATENTS1_WIDTH doesn't match mu1 shape!")

                eps = tf.random_normal(tf.shape(mu1))
                latents1 = mu1 + (eps * sig1)

                if EMBED_INPUTS:
                    outputs1 = Dec1(latents1, embedded_images)
                    outputs1_sample = Dec1(latents1_sample, embedded_images)
                else:
                    outputs1 = Dec1(latents1, scaled_images)
                    outputs1_sample = Dec1(latents1_sample, scaled_images)

                reconst_cost = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        tf.reshape(outputs1, [-1, 256]),
                        tf.reshape(images, [-1])
                    )
                )

                # Layer 2

                mu_and_logsig2 = Enc2(h1)
                mu2, logsig2, sig2 = split(mu_and_logsig2)

                eps = tf.random_normal(tf.shape(mu2))
                latents2 = mu2 + (eps * sig2)

                outputs2 = Dec2(latents2, latents1)

                mu1_prior, logsig1_prior, sig1_prior = split(outputs2)
                logsig1_prior, sig1_prior = clamp_logsig_and_sig(logsig1_prior, sig1_prior)
                mu1_prior = 2. * tf.nn.softsign(mu1_prior / 2.)

                # Assembly

                # An alpha of exactly 0 can sometimes cause inf/nan values, so we're
                # careful to avoid it.
                alpha1 = tf.minimum(1., tf.cast(total_iters+1, 'float32') / ALPHA1_ITERS) * KL_PENALTY
                alpha2 = tf.minimum(1., tf.cast(total_iters+1, 'float32') / ALPHA2_ITERS) * alpha1# * KL_PENALTY

                kl_cost_1 = tf.reduce_mean(
                    lib.ops.kl_gaussian_gaussian.kl_gaussian_gaussian(
                        mu1, 
                        logsig1,
                        sig1,
                        mu1_prior,
                        logsig1_prior,
                        sig1_prior
                    )
                )

                kl_cost_2 = tf.reduce_mean(
                    lib.ops.kl_unit_gaussian.kl_unit_gaussian(
                        mu2, 
                        logsig2,
                        sig2
                    )
                )

                kl_cost_1 *= float(LATENT_DIM_1 * LATENTS1_WIDTH * LATENTS1_HEIGHT) / (N_CHANNELS * WIDTH * HEIGHT)
                kl_cost_2 *= float(LATENT_DIM_2) / (N_CHANNELS * WIDTH * HEIGHT)

                cost = reconst_cost + (alpha1 * kl_cost_1) + (alpha2 * kl_cost_2)

            tower_cost.append(cost)
            if MODE == 'two_level':
                tower_outputs1_sample.append(outputs1_sample)

    full_cost = tf.reduce_mean(
        tf.concat(0, [tf.expand_dims(x, 0) for x in tower_cost]), 0
    )

    if MODE == 'two_level':
        full_outputs1_sample = tf.concat(0, tower_outputs1_sample)

    # Sampling

    if MODE == 'one_level':

        ch_sym = tf.placeholder(tf.int32, shape=None)
        y_sym = tf.placeholder(tf.int32, shape=None)
        x_sym = tf.placeholder(tf.int32, shape=None)
        logits = tf.reshape(tf.slice(outputs1, tf.pack([0, ch_sym, y_sym, x_sym, 0]), tf.pack([-1, 1, 1, 1, -1])), [-1, 256])
        dec1_fn_out = tf.multinomial(logits, 1)[:, 0]
        def dec1_fn(_latents, _targets, _ch, _y, _x):
            return session.run(dec1_fn_out, feed_dict={latents1: _latents, images: _targets, ch_sym: _ch, y_sym: _y, x_sym: _x, total_iters: 99999, bn_is_training: False, bn_stats_iter:0})

        def enc_fn(_images):
            return session.run(latents1, feed_dict={images: _images, total_iters: 99999, bn_is_training: False, bn_stats_iter:0})

        sample_fn_latents1 = np.random.normal(size=(8, LATENT_DIM_2)).astype('float32')

        def generate_and_save_samples(tag):
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

            latents1_copied = np.zeros((64, LATENT_DIM_2), dtype='float32')
            for i in xrange(8):
                latents1_copied[i::8] = sample_fn_latents1

            samples = np.zeros(
                (64, N_CHANNELS, HEIGHT, WIDTH), 
                dtype='int32'
            )

            print "Generating samples"
            for y in xrange(HEIGHT):
                for x in xrange(WIDTH):
                    for ch in xrange(N_CHANNELS):
                        next_sample = dec1_fn(latents1_copied, samples, ch, y, x)
                        samples[:,ch,y,x] = next_sample

            print "Saving samples"
            color_grid_vis(
                samples, 
                8, 
                8, 
                'samples_{}.png'.format(tag)
            )


    elif MODE == 'two_level':

        def dec2_fn(_latents, _targets):
            return session.run([mu1_prior, logsig1_prior], feed_dict={latents2: _latents, latents1: _targets, total_iters: 99999, bn_is_training: False, bn_stats_iter: 0})

        ch_sym = tf.placeholder(tf.int32, shape=None)
        y_sym = tf.placeholder(tf.int32, shape=None)
        x_sym = tf.placeholder(tf.int32, shape=None)
        logits_sym = tf.reshape(tf.slice(full_outputs1_sample, tf.pack([0, ch_sym, y_sym, x_sym, 0]), tf.pack([-1, 1, 1, 1, -1])), [-1, 256])

        def dec1_logits_fn(_latents, _targets, _ch, _y, _x):
            return session.run(logits_sym,
                               feed_dict={all_latents1: _latents,
                                          all_images: _targets,
                                          ch_sym: _ch,
                                          y_sym: _y,
                                          x_sym: _x,
                                          total_iters: 99999,
                                          bn_is_training: False, 
                                          bn_stats_iter: 0})

        N_SAMPLES = BATCH_SIZE
        if N_SAMPLES % N_GPUS != 0:
            raise Exception("N_SAMPLES must be divisible by N_GPUS")
        HOLD_Z2_CONSTANT = False
        HOLD_EPSILON_1_CONSTANT = False
        HOLD_EPSILON_PIXELS_CONSTANT = False

        # Draw z2 from N(0,I)
        z2 = np.random.normal(size=(N_SAMPLES, LATENT_DIM_2)).astype('float32')
        if HOLD_Z2_CONSTANT:
          z2[:] = z2[0][None]

        # Draw epsilon_1 from N(0,I)
        epsilon_1 = np.random.normal(size=(N_SAMPLES, LATENT_DIM_1, LATENTS1_HEIGHT, LATENTS1_WIDTH)).astype('float32')
        if HOLD_EPSILON_1_CONSTANT:
          epsilon_1[:] = epsilon_1[0][None]

        # Draw epsilon_pixels from U[0,1]
        epsilon_pixels = np.random.uniform(size=(N_SAMPLES, N_CHANNELS, HEIGHT, WIDTH))
        if HOLD_EPSILON_PIXELS_CONSTANT:
          epsilon_pixels[:] = epsilon_pixels[0][None]


        def generate_and_save_samples(tag):
            # Draw z1 autoregressively using z2 and epsilon1
            print "Generating z1"
            z1 = np.zeros((N_SAMPLES, LATENT_DIM_1, LATENTS1_HEIGHT, LATENTS1_WIDTH), dtype='float32')
            for y in xrange(LATENTS1_HEIGHT):
              for x in xrange(LATENTS1_WIDTH):
                z1_prior_mu, z1_prior_logsig = dec2_fn(z2, z1)
                z1[:,:,y,x] = z1_prior_mu[:,:,y,x] + np.exp(z1_prior_logsig[:,:,y,x]) * epsilon_1[:,:,y,x]

            # Draw pixels (the images) autoregressively using z1 and epsilon_x
            print "Generating pixels"
            pixels = np.zeros((N_SAMPLES, N_CHANNELS, HEIGHT, WIDTH)).astype('int32')
            for y in xrange(HEIGHT):
                for x in xrange(WIDTH):
                    for ch in xrange(N_CHANNELS):
                        # start_time = time.time()
                        logits = dec1_logits_fn(z1, pixels, ch, y, x)
                        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                        probs = probs / np.sum(probs, axis=-1, keepdims=True)
                        cdf = np.cumsum(probs, axis=-1)
                        pixels[:,ch,y,x] = np.argmax(cdf >= epsilon_pixels[:,ch,y,x,None], axis=-1)
                        # print time.time() - start_time

            # Save them
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

            print "Saving"
            rows = int(np.sqrt(N_SAMPLES))
            while N_SAMPLES % rows != 0:
                rows -= 1
            color_grid_vis(
                pixels, rows, N_SAMPLES/rows, 
                'samples_{}.png'.format(tag)
            )

    # Train!

    if MODE == 'one_level':
        prints=[
            ('alpha', alpha), 
            ('reconst', reconst_cost), 
            ('kl1', kl_cost_1)
        ]
    elif MODE == 'two_level':
        prints=[
            ('alpha1', alpha1),
            ('alpha2', alpha2),
            ('reconst', reconst_cost), 
            ('kl1', kl_cost_1),
            ('kl2', kl_cost_2),
        ]

    decayed_lr = tf.train.exponential_decay(
        LR,
        total_iters,
        LR_DECAY_AFTER,
        LR_DECAY_FACTOR,
        staircase=True
    )

    lib.train_loop_2.train_loop(
        session=session,
        inputs=[total_iters, all_images],
        inject_iteration=True,
        bn_vars=(bn_is_training, bn_stats_iter),
        cost=full_cost,
        stop_after=TIMES['stop_after'],
        prints=prints,
        optimizer=tf.train.AdamOptimizer(decayed_lr),
        train_data=train_data,
        test_data=dev_data,
        callback=generate_and_save_samples,
        callback_every=TIMES['callback_every'],
        test_every=TIMES['test_every'],
        save_checkpoints=True
    )