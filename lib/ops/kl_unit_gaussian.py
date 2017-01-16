import theano.tensor as T

def kl_unit_gaussian(mu, log_sigma):
    """
    KL divergence from a unit Gaussian prior
    mean across axis 0 (minibatch), sum across all other axes
    based on yaost, via Alec
    """
    return -0.5 * (1 + 2 * log_sigma - mu**2 - T.exp(2 * log_sigma))