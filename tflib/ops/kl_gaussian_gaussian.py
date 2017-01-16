import tensorflow as tf

# def kl_gaussian_gaussian(mu1, sig1, mu2, sig2):
#     """
#     (adapted from https://github.com/jych/cle)
#     mu1, sig1 = posterior mu and *log* sigma
#     mu2, sig2 = prior mu and *log* sigma
#     """
#     return 0.5 * (2*sig2 - 2*sig1 + (tf.exp(2*sig1) + (mu1 - mu2)**2) / tf.exp(2*sig2) - 1)

def kl_gaussian_gaussian(mu1, logsig1, sig1, mu2, logsig2, sig2):
    """
    (adapted from https://github.com/jych/cle)
    mu1, logsig1, sig2 = posterior mu and *log* sigma
    mu2, logsig2, sig2 = prior mu and *log* sigma
    """
    with tf.name_scope('kl_gaussian_gaussian') as scope:
        return 0.5 * (2*logsig2 - 2*logsig1 + (sig1**2 + (mu1 - mu2)**2) / sig2**2 - 1)