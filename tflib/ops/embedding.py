import tflib as lib

import numpy as np
import tensorflow as tf

def Embedding(name, vocab_size, dim, indices):
    embeddings = lib.param(
        name+'.EmbeddingMatrix',
        np.random.normal(size=(vocab_size, dim)).astype('float32')
    )
    return tf.gather(embeddings, indices)