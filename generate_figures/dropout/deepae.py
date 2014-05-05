import numpy as np
import os
import sys
import time

import theano
import theano.tensor as T
from theano.ifelse import ifelse
import theano.printing
import theano.tensor.shared_randomstreams

from logistic_sgd import LogisticRegression
from load_data import load_mnist

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out,
                 activation, W=None, b=None,
                 use_bias=False, sparse_init=15):

        self.input = input
        self.activation = activation

        if W is None:
            if sparse_init is not None:
                W_values = np.zeros((n_in, n_out), dtype=theano.config.floatX)
                for i in xrange(n_out):
                    for j in xrange(sparse_init):
                        idx = rng.randint(0, n_in)
                        # don't worry about getting exactly sparse_init nonzeroese
                        #while W_values[idx, i] != 0.:
                        #    idx = rng.randint(0, n_in)
                        W_values[idx, i] = rng.randn()/np.sqrt(sparse_init) * 1.2
            else:
                W_values = 1. * np.asarray(rng.standard_normal(
                    size=(n_in, n_out)), dtype=theano.config.floatX) / np.float(n_in)**0.5
            W = theano.shared(value=W_values, name='W')
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')
        self.W = W
        self.b = b

        if use_bias:
            lin_output = T.dot(input, self.W) + self.b
        else:
            lin_output = T.dot(input, self.W)

        self.output = (lin_output if activation is None else activation(lin_output))
        # parameters of the model
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]


class DeepAE(object):
    """
    Deep Autoencoder.
    """
    def __init__(self,
            rng,
            input,
            layer_sizes,
            use_bias=True, objective='crossent'):
        # Activation functions are all sigmoid except
        # "coding" layer that is linear.
        # TODO(ben): Make sure layer after coding layer is sigmoid.
        layer_acts = [T.nnet.sigmoid for l in layer_sizes[1:-1]]
        layer_acts = layer_acts + [None, T.nnet.sigmoid] + layer_acts
        # use for untied weights
        #layer_acts = layer_acts + [None, None] + layer_acts
        layer_sizes = layer_sizes + layer_sizes[:-1][::-1]
        print 'Layer sizes:',layer_sizes
        print 'Layer acts:',layer_acts

        # Set up all the hidden layers
        weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
        self.layers = []
        next_layer_input = input
        idx = 0
        for n_in, n_out in weight_matrix_sizes:
            print (n_in,n_out), layer_acts[idx]
            #if idx == 1:
            #    W_in = next_layer.W.T
            #else:
            #    W_in = None
            next_layer = HiddenLayer(rng=rng,
                    input=next_layer_input,
                    activation=layer_acts[idx],
                    n_in=n_in, n_out=n_out,
                    use_bias=use_bias)
            self.layers.append(next_layer)
            next_layer_input = next_layer.output
            idx += 1
        xpred = next_layer_input
        # Compute cost function, making sure to sum across data examples
        # so that we can properly average across minibatches.
        if objective == 'crossent':
            self.cost = (-input * T.log(xpred) - (1 - input) * T.log(1 - xpred)).sum(axis=1).sum()
        else:
            self.cost = ((input-xpred)**2).sum(axis=1).sum()
        # Grab all the parameters together.
        #XXX
        self.params = [ param for layer in self.layers for param in layer.params ]
        #self.params = [self.layers[0].W]

def convert_variable(x):
    if x.ndim == 1:
        return T.vector(x.name, dtype=x.dtype)
    else:
        return T.matrix(x.name, dtype=x.dtype)

def build_f_df(layer_sizes, dropout=False, **kwargs):
    print '... building the model'
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    # Make sure initialization is repeatable
    rng = np.random.RandomState(1234)
    # construct the MLP class
    dae = DeepAE(rng=rng, input=x,
           layer_sizes=layer_sizes, **kwargs)
    # Build the expresson for the cost function.
    cost = dae.cost
    gparams = []
    for param in dae.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)
    symbolic_params = [convert_variable(param) for param in dae.params]
    givens = dict(zip(dae.params, symbolic_params))
    f_df = theano.function(inputs=symbolic_params + [x], outputs=[cost] + gparams, givens=givens)
    return f_df, dae
