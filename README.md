Sum of Functions Optimizer (SFO)
================================

This code implements the optimization algorithm, and reproduces the figures, contained in the paper<br>
> Jascha Sohl-Dickstein, Ben Poole, and Surya Ganguli<br>
> An adaptive low dimensional quasi-Newton sum of functions optimizer<br>
> arXiv preprint arXiv:1311.2115 (2013)<br>
> http://arxiv.org/abs/1311.2115

## Use SFO
To use SFO, you should first import SFO,  
`from sfo import SFO`  
then initialize it,    
`optimizer = SFO(f_df, theta_init, subfunction_references)`    
then call the optimizer, specifying the number of optimization passes to perform,    
`theta = optimizer.optimize(num_passes=1)`.

The three required parameters for initialization are:    
- *f_df* - Returns the function value and gradient for a single subfunction
            call.  Should have the form
                `[f, dfdtheta] = f_df(theta, subfunction_references[idx])`,
            where *idx* is the index of a single subfunction.    
- *theta_init* - The initial parameters to be used for optimization.  *theta_init* can
            be either a NumPy array, an array of NumPy arrays, or a dictionary
            of NumPy arrays.  The gradient returned by *f_df* should have the
            same form as *theta_init*.    
- *subfunction_references* - A list containing an identifying element for
            each subfunction.  The elements in this list could be, eg, numpy
            matrices containing minibatches, or indices identifying the
            subfunction, or filenames from which target data should be read.

More detailed documentation, and additional options, can be found in **sfo.py**.  If too much time is spent inside SFO, relative to inside the objective function, then reduce the number of subfunctions by increasing the minibatch size or merging subfunctions.  Simple example code training an autoencoder is included at the end of this readme.  Email jascha@stanford.edu with any remaining questions.

## Reproduce figures from the paper
To reproduce the figures from the paper, run **figure\_cartoon.py**, **figure\_overhead.py**, or **figure\_convergence.py**.  **figure\_overhead.py** and **figure\_convergence.py** both require a subdirectory **figure_data/** which contains training data, and is too large to commit to this GitHub repository.  This will be available for download shortly -- URL to follow.


## Example code

The following code example trains an autoencoder using SFO.

```python
from numpy import *
from numpy.random import randn
from sfo import SFO

# define an objective function and gradient
def f_df(theta, v):
    """
    Calculate reconstruction error and gradient for an autoencoder with sigmoid
    nonlinearity.
    v contains the training data, and will be different for each subfunction.
    """
    h = 1./(1. + exp(-(dot(theta['W'], v) + theta['b_h'])))
    v_hat = dot(theta['W'].T, h) + theta['b_v']
    f = sum((v_hat - v)**2) / v.shape[1]
    dv_hat = 2.*(v_hat - v) / v.shape[1]
    db_v = sum(dv_hat, axis=1).reshape((-1,1))
    dW = dot(h, dv_hat.T)
    dh = dot(theta['W'], dv_hat)
    db_h = sum(dh*h*(1.-h), axis=1).reshape((-1,1))
    dW += dot(dh*h*(1.-h), v.T)
    dfdtheta = {'W':dW, 'b_h':db_h, 'b_v':db_v}
    return f, dfdtheta

# set model and training data parameters
M = 20 # number visible units
J = 10 # number hidden units
D = 100000 # full data batch size
N = int(sqrt(D)/10.) # number minibatches
# generate random training data
v = randn(M,D)

# create the array of subfunction specific arguments
sub_refs = []
for i in range(N):
    # extract a single minibatch of training data.
    sub_refs.append(v[:,i::N])

# initialize parameters
theta_init = {'W':randn(J,M), 'b_h':randn(J,1), 'b_v':randn(M,1)}
# initialize the optimizer
optimizer = SFO(f_df, theta_init, sub_refs)
# run the optimizer for 1 pass through the data
theta = optimizer.optimize(num_passes=1)
# continue running the optimizer for another 50 passes through the data
theta = optimizer.optimize(num_passes=50)
# test the gradient of f_df
optimizer.check_grad()
```
