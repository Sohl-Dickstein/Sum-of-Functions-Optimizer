import numpy as np
from theano.sandbox.cuda import CudaNdarraySharedVariable as CudaNdarray

#XXX: Shamefully stolen from pyautodiff
# https://github.com/jaberg/pyautodiff/blob/master/autodiff/fmin_scipy.py

def _tonp(x):
    try:
        x = x.eval()
    except:
        pass
    #if not np.isscalar(x) and type(x) not in [CudaNdarray, np.array, np.ndarray]:
        #x = x.eval()
    return np.array(x)
        #return np.array(x)
    #elif type(x) in [np.array, np.ndarray]:
    #    return x
    #else:
    #    return np.array(x.eval())

def vector_from_args(raw_args):
    args = [_tonp(a) for a in raw_args]
    args_sizes = [w.size for w in args]
    x_size = sum(args_sizes)
    x = np.empty(x_size, dtype=args[0].dtype) # has to be float64 for fmin_l_bfgs_b
    i = 0
    for w in args:
        x[i: i + w.size] = w.flatten()
        i += w.size
    return x

def args_from_vector(x, orig_args):
    #if type(orig_args[0]) != np.ndarray:
    #    orig_args = [a.eval() for a in orig_args]
    # unpack x_opt -> args-like structure `args_opt`
    rval = []
    i = 0
    for w in orig_args:
        size = _tonp(w.size)
        rval.append(x[i: i + size].reshape(_tonp(w.shape)).astype(w.dtype))
        i += size
    return rval
