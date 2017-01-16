import numpy
import theano
import theano.tensor as T

import cPickle as pickle
import math
import time
import locale

locale.setlocale(locale.LC_ALL, '')

_params = {}
def param(name, *args, **kwargs):
    """
    A wrapper for `theano.shared` which enables parameter sharing in models.
    
    Creates and returns theano shared variables similarly to `theano.shared`, 
    except if you try to create a param with the same name as a 
    previously-created one, `param(...)` will just return the old one instead of 
    making a new one.

    This constructor also adds a `param` attribute to the shared variables it 
    creates, so that you can easily search a graph for all params.
    """

    if name not in _params:
        kwargs['name'] = name
        param = theano.shared(*args, **kwargs)
        param.param = True
        _params[name] = param
    return _params[name]

def delete_params_with_name(name):
    to_delete = [p_name for p_name in _params if name in p_name]
    for p_name in to_delete:
        del _params[p_name]

def delete_all_params():
    to_delete = [p_name for p_name in _params]
    for p_name in to_delete:
        del _params[p_name]

def save_params(path):
    param_vals = {}
    for name, param in _params.iteritems():
        param_vals[name] = param.get_value()
        # print name

    with open(path, 'wb') as f:
        pickle.dump(param_vals, f)

def load_params(path):
    with open(path, 'rb') as f:
        param_vals = pickle.load(f)

    for name, val in param_vals.iteritems():
        _params[name].set_value(val)
        # print name

def search(node, critereon):
    """
    Traverse the Theano graph starting at `node` and return a list of all nodes
    which match the `critereon` function. When optimizing a cost function, you 
    can use this to get a list of all of the trainable params in the graph, like
    so:

    `lib.search(cost, lambda x: hasattr(x, "param"))`
    """

    def _search(node, critereon, visited):
        if node in visited:
            return []
        visited.add(node)

        results = []
        if isinstance(node, T.Apply):
            for inp in node.inputs:
                results += _search(inp, critereon, visited)
        else: # Variable node
            if critereon(node):
                results.append(node)
            if node.owner is not None:
                results += _search(node.owner, critereon, visited)
        return results

    return _search(node, critereon, set())

def floatX(x):
    """
    Convert `x` to the numpy type specified in `theano.config.floatX`.
    """
    if theano.config.floatX == 'float16':
        return numpy.float16(x)
    elif theano.config.floatX == 'float32':
        return numpy.float32(x)
    else: # Theano's default float type is float64
        print "Warning: lib.floatX using float64"
        return numpy.float64(x)

def print_params_info(params):
    """Print information about the parameters in the given param set."""

    params = sorted(params, key=lambda p: p.name)
    values = [p.get_value(borrow=True) for p in params]
    shapes = [p.shape for p in values]
    print "Params for cost:"
    for param, value, shape in zip(params, values, shapes):
        print "\t{0} ({1})".format(
            param.name,
            ",".join([str(x) for x in shape])
        )

    total_param_count = 0
    for shape in shapes:
        param_count = 1
        for dim in shape:
            param_count *= dim
        total_param_count += param_count
    print "Total parameter count: {0}".format(
        locale.format("%d", total_param_count, grouping=True)
    )

def print_model_settings(locals_):
    print "Model settings:"
    all_vars = [(k,v) for (k,v) in locals_.items() if (k.isupper() and k!='T')]
    all_vars = sorted(all_vars, key=lambda x: x[0])
    for var_name, var_value in all_vars:
        print "\t{}: {}".format(var_name, var_value)