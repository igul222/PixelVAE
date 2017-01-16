import tensorflow as tf

def kl_unit_gaussian(mu, log_sigma, sigma):
    """
    KL divergence from a unit Gaussian prior
    based on yaost, via Alec
    """
    with tf.name_scope('kl_unit_gaussian') as scope:
        return -0.5 * (1 + 2 * log_sigma - mu**2 - sigma**2)