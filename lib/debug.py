import numpy as np
import theano
from theano import gof

class DebugOp(gof.Op):
    def __init__(self, name, fn):
        super(DebugOp, self).__init__()
        self._name = name
        self._fn = fn

    def make_node(self, x):
        return gof.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        self._fn(self._name, inputs[0])
        output_storage[0][0] = np.copy(inputs[0])

    def grad(self, inputs, output_gradients):
        return [DebugOp(self._name+'.grad', self._fn)(output_gradients[0])]

def print_shape(name, x):
    def fn(_name, _x):
        print "{} shape: {}".format(_name, _x.shape)
    return DebugOp(name, fn)(x)

def print_stats(name, x):
    return x
    def fn(_name, _x):
        mean = np.mean(_x)
        std = np.std(_x)
        percentiles = np.percentile(_x, [0,25,50,75,100])
        # percentiles = "skipping"
        print "{}\tmean:{}\tstd:{}\tpercentiles:{}\t".format(_name, mean, std, percentiles)
    return DebugOp(name, fn)(x)