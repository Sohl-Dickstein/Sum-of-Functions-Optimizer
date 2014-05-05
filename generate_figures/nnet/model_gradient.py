import theano
import theano.tensor as T
import numpy as np
from collections import OrderedDict
from itertools import izip
from string import Template

from pylearn2.utils import serial, safe_zip
from pylearn2.utils.data_specs import DataSpecsMapping
from pylearn2.config.yaml_parse import load_path, load
from theano.sandbox.cuda import CudaNdarray

def _tonp(x):
    if type(x) not in [CudaNdarray, np.array, np.ndarray]:
        x = x.eval()
    return np.array(x)

def load_model(filename, batch_size=100):
    out=Template(open(filename, 'r').read()).substitute({'batch_size':batch_size})
    stuff = load(out)
    model = stuff['model']
    #model.batch_size = batch_size
    #model.set_batch_size(batch_size)
    cost = stuff['algorithm'].cost
    if cost is None:
        cost = model.get_default_cost()
    mg = ModelGradient(model, cost,batch_size=batch_size)
    return mg


class ModelGradient:
    def __init__(self, model, cost=None,  batch_size=100):
        self.model = model
        self.model.set_batch_size(batch_size)
        self.model._test_batch_size = batch_size
        print 'it should really be ', batch_size
        self.cost = cost
        self.batch_size = batch_size
        self.setup()

    def setup(self):
        self.X = T.matrix('X')
        self.Y = T.matrix('Y')

        # Taken from pylearn2/training_algorithms/sgd.py


        data_specs = self.cost.get_data_specs(self.model)
        mapping = DataSpecsMapping(data_specs)
        space_tuple = mapping.flatten(data_specs[0], return_tuple=True)
        source_tuple = mapping.flatten(data_specs[1], return_tuple=True)

        # Build a flat tuple of Theano Variables, one for each space.
        # We want that so that if the same space/source is specified
        # more than once in data_specs, only one Theano Variable
        # is generated for it, and the corresponding value is passed
        # only once to the compiled Theano function.
        theano_args = []
        for space, source in safe_zip(space_tuple, source_tuple):
            name = '%s[%s]' % (self.__class__.__name__, source)
            arg = space.make_theano_batch(name=name, batch_size = self.batch_size)
            theano_args.append(arg)
        print 'BATCH SIZE=',self.batch_size
        theano_args = tuple(theano_args)

        # Methods of `self.cost` need args to be passed in a format compatible
        # with data_specs
        nested_args = mapping.nest(theano_args)
        print self.cost
        fixed_var_descr = self.cost.get_fixed_var_descr(self.model, nested_args)
        print self.cost
        self.on_load_batch = fixed_var_descr.on_load_batch
        params = list(self.model.get_params())
        self.X = nested_args[0]
        self.Y = nested_args[1]
        init_grads, updates = self.cost.get_gradients(self.model, nested_args)

        params = self.model.get_params()
        # We need to replace parameters with purely symbolic variables in case some are shared
        # Create gradient and cost functions
        self.params = params
        symbolic_params = [self._convert_variable(param) for param in params]
        givens = dict(zip(params, symbolic_params))
        costfn = self.model.cost_from_X((self.X, self.Y))
        gradfns = [init_grads[param] for param in params]
        #self.symbolic_params = symbolic_params
        #self._loss = theano.function(symbolic_para[self.X, self.Y], self.model.cost_from_X((self.X, self.Y)))#, givens=givens)
        #1/0
        print 'Compiling function...'
        self.theano_f_df = theano.function(inputs=symbolic_params + [self.X, self.Y], outputs=[costfn] + gradfns, givens=givens)
        print 'done'
     #   self.grads = theano.function(symbolic_params + [self.X, self.Y], [init_grads[param] for param in params], givens=givens)
     #   self._loss = theano.function(symbolic_params + [self.X, self.Y], self.model.cost(self.X, self.Y), givens=givens)
        # Maps params -> their derivative

    def f_df(self, theta, args):
        X = args[0]
        y = args[1]
        X = np.transpose(X,(1,2,3,0))
        y = y
        nsamples = X.shape[-1]
        nbatches = nsamples / self.batch_size
        # lets hope it's actually divisible
        #XXX(ben): fix this
        cost = 0.0
        thetas = None
        idxs = np.array_split(np.arange(nsamples), nbatches)
        for idx in idxs:
            theano_args = theta + [X[...,idx], y[idx,...]]
            results = self.theano_f_df(*theano_args)
            results = [_tonp(result) for result in results]
            if thetas is None:
                thetas = results[1:]
            else:
                thetas = [np.array(t) + np.array(result) for t,result in zip(thetas,results[1:])] 
            cost += results[0]/nbatches

        thetas = [np.array(theta)/nbatches for theta in thetas]

#        if X.shape[1] != self.batch_size:


#        print X.shape, y.shape
#        theano_args = theta + [X,y]
#        results = self.theano_f_df(*theano_args)
#        return results[0], results[1:]
        return cost, thetas

    def _convert_variable(self, x):
        return T.TensorType(x.dtype, x.broadcastable)(x.name) #'int32', broadcastable=())('myvar')


    
if __name__ == '__main__':
    # Load train obj
    f = np.load('test.npz')
    model = f['model'].item()
    cost = f['cost'].item()
    m = ModelGradient(model, cost)
    p = model.weights.shape.eval()[0]
    X = np.random.randn(100,p).astype(np.float32)
    theta = m.get_params()
    m.f_df(theta, X)
    1/0

    # Load cost function
    W = np.random.randn(10,10)
    W = theano.shared(W)

    cost = (W**2).sum()
    grad = T.grad(cost, W)

    params = [W]
    1/0
