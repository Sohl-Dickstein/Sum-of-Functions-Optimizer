"""
Train an autoencoder using SFO.

Demonstrates usage of the Sum of Functions Optimizer (SFO) Python
package.  See sfo.py and
https://github.com/Sohl-Dickstein/Sum-of-Functions-Optimizer
for additional documentation.

Author: Jascha Sohl-Dickstein (2014)
This software is made available under the Creative Commons
Attribution-Noncommercial License.
( http://creativecommons.org/licenses/by-nc/3.0/ )
"""

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
# # uncomment the following line to test the gradient of f_df
# optimizer.check_grad()
# run the optimizer for 1 pass through the data
theta = optimizer.optimize(num_passes=1)
# continue running the optimizer for another 20 passes through the data
theta = optimizer.optimize(num_passes=20)

# plot the convergence trace
plt.plot(np.array(optimizer.hist_f_flat))
plt.xlabel('Iteration')
plt.ylabel('Minibatch Function Value')
plt.title('Convergence Trace')
plt.show()
