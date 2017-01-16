"""
VAE + Pixel CNN
Ishaan Gulrajani
"""


"""
Modified by Kundan Kumar

Usage: THEANO_FLAGS='mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=.95' python models/mnist_pixelvae_evaluate.py -L 12 -fs 5 -algo cond_z_bias -dpx 16 -ldim 16 -w path/to/weights.pkl
"""

import os, sys
sys.path.append(os.getcwd())

import time

import argparse

import lib
import lib.train_loop
import lib.mnist_binarized
import lib.ops.kl_unit_gaussian
import lib.ops.conv2d
import lib.ops.deconv2d
import lib.ops.linear

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import scipy.misc
import lasagne
import pickle

import functools


parser = argparse.ArgumentParser(description='Generating images pixel by pixel')
parser.add_argument('-L','--num_pixel_cnn_layer', required=True, type=int, help='Number of layers to use in pixelCNN')
parser.add_argument('-algo', '--decoder_algorithm', required = True, help="One of 'cond_z_bias', 'upsample_z_no_conv', 'upsample_z_conv', 'upsample_z_conv_tied' 'vae_only'" )
parser.add_argument('-enc', '--encoder', required = False, default='simple', help="Encoder: 'complecated' or 'simple' " )
parser.add_argument('-dpx', '--dim_pix', required = False, default=32, type = int )
parser.add_argument('-fs', '--filter_size', required = False, default=5, type = int )
parser.add_argument('-ldim', '--latent_dim', required = False, default=64, type = int )
parser.add_argument('-ait', '--alpha_iters', required = False, default=10000, type = int )
parser.add_argument('-w', '--pre_trained_weights', required = True)


args = parser.parse_args()


assert args.decoder_algorithm in ['cond_z_bias', 'upsample_z_conv']

print args



lib.ops.conv2d.enable_default_weightnorm()
lib.ops.linear.enable_default_weightnorm()

OUT_DIR = '/Tmp/kumarkun/mnist_pixel_final' + "/num_layers_new2_" + str(args.num_pixel_cnn_layer) + args.decoder_algorithm + "_"+args.encoder

if not os.path.isdir(OUT_DIR):
   os.makedirs(OUT_DIR)
   print "Created directory {}".format(OUT_DIR)

def floatX(num):
    if theano.config.floatX == 'float32':
        return np.float32(num)
    else:
        raise Exception("{} type not supported".format(theano.config.floatX))


T.nnet.elu = lambda x: T.switch(x >= floatX(0.), x, T.exp(x) - floatX(1.))

DIM_1 = 32
DIM_2 = 32
DIM_3 = 64
DIM_4 = 64
DIM_PIX = args.dim_pix
PIXEL_CNN_FILTER_SIZE = args.filter_size
PIXEL_CNN_LAYERS = args.num_pixel_cnn_layer

LATENT_DIM = args.latent_dim
ALPHA_ITERS = args.alpha_iters
VANILLA = False
LR = 1e-3

BATCH_SIZE = 100
N_CHANNELS = 1
HEIGHT = 28
WIDTH = 28

TEST_BATCH_SIZE = 100
TIMES = ('iters', 500, 500*400, 500, 400*500, 2*ALPHA_ITERS)

lib.print_model_settings(locals().copy())

theano_srng = RandomStreams(seed=234)

def PixCNNGate(x):
    a = x[:,::2]
    b = x[:,1::2]
    return T.tanh(a) * T.nnet.sigmoid(b)

def PixCNN_condGate(x, z, dim,  activation= 'tanh', name = ""):
    a = x[:,::2]
    b = x[:,1::2]

    Z_to_tanh = lib.ops.linear.Linear(name+".tanh", input_dim=LATENT_DIM, output_dim=dim, inputs=z)
    Z_to_sigmoid = lib.ops.linear.Linear(name+".sigmoid", input_dim=LATENT_DIM, output_dim=dim, inputs=z)

    a = a + Z_to_tanh[:,:, None, None]
    b = b + Z_to_sigmoid[:,:,None, None]

    if activation == 'tanh':
        return T.tanh(a) * T.nnet.sigmoid(b)
    else:
        return T.nnet.elu(a) * T.nnet.sigmoid(b)

def next_stacks(X_v, X_h, inp_dim, name,
                global_conditioning = None,
                filter_size = 3,
                hstack = 'hstack',
                residual = True
            ):
    zero_pad = T.zeros((X_v.shape[0], X_v.shape[1], 1, X_v.shape[3]))

    X_v_padded = T.concatenate([zero_pad, X_v], axis = 2)

    X_v_next = lib.ops.conv2d.Conv2D(
            name + ".vstack",
            input_dim=inp_dim,
            output_dim=2*DIM_PIX,
            filter_size=filter_size,
            inputs=X_v_padded,
            mask_type=('vstack', N_CHANNELS)
        )

    X_v_next_gated = PixCNNGate(X_v_next)

    X_v2h = lib.ops.conv2d.Conv2D(
            name + ".v2h",
            input_dim=2*DIM_PIX,
            output_dim=2*DIM_PIX,
            filter_size=(1,1),
            inputs=X_v_next[:,:,:-1,:]
        )

    X_h_next = lib.ops.conv2d.Conv2D(
            name + '.hstack',
            input_dim= inp_dim,
            output_dim= 2*DIM_PIX,
            filter_size= (1,filter_size),
            inputs= X_h,
            mask_type=(hstack, N_CHANNELS)
        )

    X_h_next = PixCNNGate(X_h_next + X_v2h)

    X_h_next = lib.ops.conv2d.Conv2D(
            name + '.h2h',
            input_dim=DIM_PIX,
            output_dim=DIM_PIX,
            filter_size=(1,1),
            inputs= X_h_next
            )

    if residual == True:
        X_h_next = X_h_next + X_h

    return X_v_next_gated[:, :, 1:, :], X_h_next

def next_stacks_gated(X_v, X_h, inp_dim, name, global_conditioning = None,
                                             filter_size = 3, hstack = 'hstack', residual = True):
    zero_pad = T.zeros((X_v.shape[0], X_v.shape[1], 1, X_v.shape[3]))

    X_v_padded = T.concatenate([zero_pad, X_v], axis = 2)

    X_v_next = lib.ops.conv2d.Conv2D(
            name + ".vstack",
            input_dim=inp_dim,
            output_dim=2*DIM_PIX,
            filter_size=filter_size,
            inputs=X_v_padded,
            mask_type=('vstack', N_CHANNELS)
        )
    X_v_next_gated = PixCNN_condGate(X_v_next, global_conditioning, DIM_PIX,
                                     name = name + ".vstack.conditional")

    X_v2h = lib.ops.conv2d.Conv2D(
            name + ".v2h",
            input_dim=2*DIM_PIX,
            output_dim=2*DIM_PIX,
            filter_size=(1,1),
            inputs=X_v_next[:,:,:-1,:]
        )


    X_h_next = lib.ops.conv2d.Conv2D(
            name + '.hstack',
            input_dim= inp_dim,
            output_dim= 2*DIM_PIX,
            filter_size= (1,filter_size),
            inputs= X_h,
            mask_type=(hstack, N_CHANNELS)
        )

    X_h_next = PixCNN_condGate(X_h_next + X_v2h, global_conditioning, DIM_PIX, name = name + ".hstack.conditional")

    X_h_next = lib.ops.conv2d.Conv2D(
            name + '.h2h',
            input_dim=DIM_PIX,
            output_dim=DIM_PIX,
            filter_size=(1,1),
            inputs= X_h_next
            )

    if residual:
        X_h_next = X_h_next + X_h

    return X_v_next_gated[:, :, 1:, :], X_h_next



def Encoder(inputs):

    output = inputs

    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.1', input_dim=N_CHANNELS, output_dim=DIM_1, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.2', input_dim=DIM_1,      output_dim=DIM_2, filter_size=3, inputs=output, stride=2))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.3', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.4', input_dim=DIM_2, output_dim=DIM_3, filter_size=3, inputs=output, stride=2))

    # Pad from 7x7 to 8x8
    padded = T.zeros((output.shape[0], output.shape[1], 8, 8), dtype='float32')
    output = T.inc_subtensor(padded[:,:,:7,:7], output)

    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.5', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.6', input_dim=DIM_3, output_dim=DIM_4, filter_size=3, inputs=output, stride=2))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.7', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.8', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))

    output = output.reshape((output.shape[0], -1))
    output = lib.ops.linear.Linear('Enc.Out', input_dim=4*4*DIM_4, output_dim=2*LATENT_DIM, inputs=output)
    return output[:, ::2], output[:, 1::2]


def Decoder_no_blind(latents, images):
    output = latents

    output = lib.ops.linear.Linear('Dec.Inp', input_dim=LATENT_DIM, output_dim=4*4*DIM_4, inputs=output)
    output = T.nnet.relu(output.reshape((output.shape[0], DIM_4, 4, 4)))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec.1', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec.2', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.3', input_dim=DIM_4, output_dim=DIM_3, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.4', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, inputs=output))

    # Cut from 8x8 to 7x7
    output = output[:,:,:7,:7]

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.5', input_dim=DIM_3, output_dim=DIM_2, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.6', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, inputs=output))

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.7', input_dim=DIM_2, output_dim=DIM_1, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.8', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, inputs=output))

    skip_outputs = []

    images_with_latent = T.concatenate([images, output], axis=1)

    X_v, X_h = next_stacks(images_with_latent, images_with_latent, N_CHANNELS + DIM_1, "Dec.PixInput", filter_size = 7, hstack = "hstack_a", residual = False)

    for i in xrange(PIXEL_CNN_LAYERS):
        X_v, X_h = next_stacks(X_v, X_h,  DIM_PIX, "Dec.Pix"+str(i+1), filter_size = PIXEL_CNN_FILTER_SIZE)


    output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_PIX, output_dim=2*DIM_1, filter_size=1, inputs=X_h))

    output = PixCNNGate(lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))

    output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_1, output_dim=N_CHANNELS, filter_size=1, inputs=output, he_init=False)

    return output


def Decoder_no_blind_conditioned_on_z(latents, images):
    output = latents

    X_v, X_h = next_stacks_gated(
                images, images, N_CHANNELS, "Dec.PixInput",
                global_conditioning = latents, filter_size = 7,
                hstack = "hstack_a", residual = False
                )

    for i in xrange(PIXEL_CNN_LAYERS):
        X_v, X_h = next_stacks_gated(X_v, X_h, DIM_PIX, "Dec.Pix"+str(i+1), global_conditioning = latents, filter_size = PIXEL_CNN_FILTER_SIZE)


    output = lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_PIX, output_dim=2*DIM_PIX, filter_size=1, inputs=X_h)
    output = PixCNN_condGate(output, latents, DIM_PIX, name='Dec.PixOut1.cond' )
    output = lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_PIX, output_dim=2*DIM_PIX, filter_size=1, inputs=output)
    output = PixCNN_condGate(output, latents, DIM_PIX, name='Dec.PixOut2.cond' )

    output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX, output_dim=N_CHANNELS, filter_size=1, inputs=output, he_init=False)

    return output

def binarize(images):
    """
    Stochastically binarize values in [0, 1] by treating them as p-values of
    a Bernoulli distribution.
    """
    return (
        np.random.uniform(size=images.shape) < images
    ).astype(theano.config.floatX)



if args.decoder_algorithm == 'cond_z_bias':
    decode_algo = Decoder_no_blind_conditioned_on_z
elif args.decoder_algorithm == 'upsample_z_conv':
    decode_algo = Decoder_no_blind
else:
    assert False, "you should never be here!!"


encoder = Encoder

total_iters = T.iscalar('total_iters')
images = T.tensor4('images') # shape: (batch size, n channels, height, width)

mu, log_sigma = encoder(images)

if VANILLA:
    latents = mu
else:
    eps = T.cast(theano_srng.normal(mu.shape), theano.config.floatX)
    latents = mu + (eps * T.exp(log_sigma))

# Theano bug: NaNs unless I pass 2D tensors to binary_crossentropy
reconst_cost = T.nnet.binary_crossentropy(
    T.nnet.sigmoid(
        decode_algo(latents, images).reshape((-1, N_CHANNELS*HEIGHT*WIDTH))
    ),
    images.reshape((-1, N_CHANNELS*HEIGHT*WIDTH))
).sum(axis=1)

reg_cost = lib.ops.kl_unit_gaussian.kl_unit_gaussian(
    mu,
    log_sigma
).sum(axis=1)

alpha = T.minimum(
    1,
    T.cast(total_iters, theano.config.floatX) / lib.floatX(ALPHA_ITERS)
)

if VANILLA:
    cost = reconst_cost
else:
    cost = reconst_cost + (alpha * reg_cost)

sample_fn_latents = T.matrix('sample_fn_latents')
sample_fn = theano.function(
    [sample_fn_latents, images],
    T.nnet.sigmoid(decode_algo(sample_fn_latents, images)),
    on_unused_input='warn'
)

eval_fn = theano.function(
    [images, total_iters],
    cost.mean()
)

train_data, dev_data, test_data = lib.mnist_binarized.load(
    BATCH_SIZE,
    TEST_BATCH_SIZE
)


#############################################
##############Importance Sampling###########
log2pi = T.constant(np.log(2*np.pi).astype(theano.config.floatX))

k_ = 10

def log_mean_exp(x, axis=1):
    m = T.max(x,  keepdims=True)
    return m + T.log(T.sum(T.exp(x - m), keepdims=True)) - T.log(k_)

def log_lik(samples, mean, log_sigma):
    return -log2pi*T.cast(samples.shape[1], 'float32') / 2 -  \
        T.sum(T.sqr((samples-mean)/T.exp(log_sigma)) + 2*log_sigma, axis=1) / 2

vae_bound = reconst_cost + reg_cost
log_lik_latent_prior = log_lik(latents, 0., 0.)
log_lik_latent_posterior = log_lik(latents, mu, log_sigma)
loglikelihood_normal =  log_lik_latent_prior - reconst_cost - log_lik_latent_posterior

loglikelihood = -log_mean_exp(loglikelihood_normal)
lik_fn = theano.function(
    [images],
    [loglikelihood, vae_bound, reconst_cost, reg_cost, log_lik_latent_prior, log_lik_latent_posterior, loglikelihood_normal]
)



def compute_importance_weighted_likelihood():
    i = 0
    total_lik = []
    total_lik_bound = []
    for (images,) in test_data():
        for im in images:
            batch_ = np.tile(im, [k_, 1, 1, 1])
            res = lik_fn(batch_)
            total_lik_bound.append(res[1].mean())

            total_lik.append(res[0])
            i += 1

    print "Importance weighted likelihood", np.mean(total_lik)
    print "normal likelihood", np.mean(total_lik_bound)

print("Loading parameters...")

lib.load_params(args.pre_trained_weights)

print("Computing Log-likelihood..")
compute_importance_weighted_likelihood()


