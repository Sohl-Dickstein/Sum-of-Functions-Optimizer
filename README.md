Sum of Functions Optimizer (SFO)
================================

SFO is a function optimizer for the case where the target function breaks into a sum over minibatches, or a sum over contributing functions.  It combines the benefits of both quasi-Newton and stochastic gradient descent techniques, and will likely converge faster and to a better function value than either.  It does not require tuning of hyperparameters.  It is described in more detail in the paper:
> Jascha Sohl-Dickstein, Ben Poole, and Surya Ganguli<br>
> Fast large-scale optimization by unifying stochastic gradient and quasi-Newton methods<br>
> International Conference on Machine Learning (2014)<br>
> arXiv preprint arXiv:1311.2115 (2013)<br>
> http://arxiv.org/abs/1311.2115

This repository provides easy to use Python and MATLAB implementations of SFO, as well as functions to exactly reproduce the figures in the paper.<br>

## Use SFO

Simple example code which trains an autoencoder is in **sfo_demo.py** and **sfo_demo.m**, and is reproduced at the end of this README.

### Python package

To use SFO, you should first import SFO,  
`from sfo import SFO`  
then initialize it,    
`optimizer = SFO(f_df, theta_init, subfunction_references)`    
then call the optimizer, specifying the number of optimization passes to perform,    
`theta = optimizer.optimize(num_passes=1)`.

The three required initialization parameters are:    
- *f_df* - Returns the function value and gradient for a single subfunction
            call.  Should have the form
                `f, dfdtheta = f_df(theta, subfunction_references[idx])`,
            where *idx* is the index of a single subfunction.    
- *theta_init* - The initial parameters to be used for optimization.  *theta_init* can
            be either a NumPy array, an array of NumPy arrays, a dictionary of NumPy
            arrays, or a nested combination thereof.  The gradient returned by *f_df*
            should have the same form as *theta_init*.    
- *subfunction_references* - A list containing an identifying element for
            each subfunction.  The elements in this list could be, eg, numpy
            matrices containing minibatches, or indices identifying the
            subfunction, or filenames from which target data should be read.
            **If each subfunction corresponds to a minibatch, then the number of
            subfunctions should be approximately \[number subfunctions\] = sqrt(\[dataset size\])/10**.

More detailed documentation, and additional options, can be found in **sfo.py**.

### MATLAB package

To use SFO you must first initialize the optimizer,    
`optimizer = sfo(@f_df, theta_init, subfunction_references, [varargin]);`    
then call the optimizer, specifying the number of optimization passes to perform,    
`theta = optimizer.optimize(20);`.

The initialization parameters are:    
- *f_df* - Returns the function value and gradient for a single subfunction
            call.  Should have the form
                `[f, dfdtheta] = f_df(theta, subfunction_references{idx}, varargin{:})`,
            where *idx* is the index of a single subfunction.    
- *theta_init* - The initial parameters to be used for optimization.  *theta_init* can
            be either a vector, a matrix, or a cell array with a vector or
            matrix in every cell.  The gradient returned by *f_df* should have the
            same form as *theta_init*.    
- *subfunction_references* - A cell array containing an identifying element
            for each subfunction.  The elements in this list could be, eg,
            matrices containing minibatches, or indices identifying the
            subfunction, or filenames from which target data should be read.
            **If each subfunction corresponds to a minibatch, then the number of
            subfunctions should be approximately \[number subfunctions\] = sqrt(\[dataset size\])/10**.
- *[varargin]* - Any additional parameters, which will be passed through to *f_df* each time
            it is called.

Slightly more documentation can be found in **sfo.m**.

### Special situations

Email jaschasd@google.com with questions if you don't find your answer here.

#### Reducing overhead

If too much time is spent inside SFO relative to inside the objective function, then reduce the number of subfunctions by increasing the minibatch size or merging subfunctions.

#### Replacing minibatches / using SFO in the infinite data limit

In the Python version of the code, a subfunction or minibatch can be replaced by calling the function `replace_subfunction`.  See the documentation in ***sfo.py*** for more details.  Note that replacing a subfunction without calling `replace_subfunction` will cause the optimizer to fail, since SFO relies on subfunctions returning consistent gradients.

#### Using with dropout

Stochastic gradients will break SFO, because it uses the change in the minibatch/subfunction gradient to estimate the Hessian matrix.  The benefits of noise regularization can be achieved without making the gradients stochastic by using frozen noise.  That is, in the case of dropout, assign a random dropout mask to each datapoint.  Every time that datapoint is evaluated however, use the same dropout mask.  This makes the gradients consistent across multiple evaluations of the minibatch.

## Reproduce figures from paper
To reproduce the figures from the paper, run **generate\_figures/figure\_\*.py**.  Several of the figures rely on a subdirectory **figure_data/** with training data.  This can be downloaded from https://www.dropbox.com/sh/h9z4djlgl2tagmu/GlVAJyErf8 .

## Example code

The following code blocks train an autoencoder using SFO in Python and MATLAB respectively.  Identical code is in **sfo_demo.py** and **sfo_demo.m**.

### Python example code

```python
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn
from sfo import SFO

# define an objective function and gradient
def f_df(theta, v):
    """
    Calculate reconstruction error and gradient for an autoencoder with sigmoid
    nonlinearity.
    v contains the training data, and will be different for each subfunction.
    """
    h = 1./(1. + np.exp(-(np.dot(theta['W'], v) + theta['b_h'])))
    v_hat = np.dot(theta['W'].T, h) + theta['b_v']
    f = np.sum((v_hat - v)**2) / v.shape[1]
    dv_hat = 2.*(v_hat - v) / v.shape[1]
    db_v = np.sum(dv_hat, axis=1).reshape((-1,1))
    dW = np.dot(h, dv_hat.T)
    dh = np.dot(theta['W'], dv_hat)
    db_h = np.sum(dh*h*(1.-h), axis=1).reshape((-1,1))
    dW += np.dot(dh*h*(1.-h), v.T)
    dfdtheta = {'W':dW, 'b_h':db_h, 'b_v':db_v}
    return f, dfdtheta

# set model and training data parameters
M = 20 # number visible units
J = 10 # number hidden units
D = 100000 # full data batch size
N = int(np.sqrt(D)/10.) # number minibatches
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
# continue running the optimizer for another 20 passes through the data
theta = optimizer.optimize(num_passes=20)

# plot the convergence trace
plt.plot(np.array(optimizer.hist_f_flat))
plt.xlabel('Iteration')
plt.ylabel('Minibatch Function Value')
plt.title('Convergence Trace')

# test the gradient of f_df
optimizer.check_grad()
```

### MATLAB example code

```MATLAB
% set model and training data parameters
M = 20; % number visible units
J = 10; % number hidden units
D = 100000; % full data batch size
N = floor(sqrt(D)/10.); % number minibatches
% generate random training data
v = randn(M,D);

% create the cell array of subfunction specific arguments
sub_refs = cell(N,1);
for i = 1:N
    % extract a single minibatch of training data.
    sub_refs{i} = v(:,i:N:end);
end

% initialize parameters
% Parameters can be stored as a vector, a matrix, or a cell array with a
% vector or matrix in each cell.  Here the parameters are 
% {[weight matrix], [hidden bias], [visible bias]}.
theta_init = {randn(J,M), randn(J,1), randn(M,1)};
% initialize the optimizer
optimizer = sfo(@f_df_autoencoder, theta_init, sub_refs);
% run the optimizer for half a pass through the data
theta = optimizer.optimize(0.5);
% run the optimizer for another 20 passes through the data, continuing from 
% the theta value where the prior call to optimize() ended
theta = optimizer.optimize(20);

% plot the convergence trace
plot(optimizer.hist_f_flat);
xlabel('Iteration');
ylabel('Minibatch Function Value');
title('Convergence Trace');

% test the gradient of f_df
optimizer.check_grad();
```

The subfunction/minibatch objective function and gradient for the MATLAB code is defined as follows,
```MATLAB
function [f, dfdtheta] = f_df_autoencoder(theta, v)
    % [f, dfdtheta] = f_df_autoencoder(theta, v)
    %     Calculate L2 reconstruction error and gradient for an autoencoder
    %     with sigmoid nonlinearity.
    %     Parameters:
    %         theta - A cell array containing
    %              {[weight matrix], [hidden bias], [visible bias]}.
    %         v - A [# visible, # datapoints] matrix containing training data.
    %              v will be different for each subfunction.
    %     Returns:
    %         f - The L2 reconstruction error for data v and parameters theta.
    %         df - A cell array containing the gradient of f with each of the
    %              parameters in theta.

    W = theta{1};
    b_h = theta{2};
    b_v = theta{3};
    
    h = 1./(1 + exp(-bsxfun(@plus, W * v, b_h)));
    v_hat = bsxfun(@plus, W' * h, b_v);
    f = sum(sum((v_hat - v).^2)) / size(v, 2);
    dv_hat = 2*(v_hat - v) / size(v, 2);
    db_v = sum(dv_hat, 2);
    dW = h * dv_hat';
    dh = W * dv_hat;
    db_h = sum(dh.*h.*(1-h), 2);
    dW = dW + dh.*h.*(1-h) * v';
    % give the gradients the same order as the parameters
    dfdtheta = {dW, db_h, db_v};
end
```
