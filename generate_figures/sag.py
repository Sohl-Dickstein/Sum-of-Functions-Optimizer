"""
Author: Jascha Sohl-Dickstein (2014)
This software is made available under the Creative Commons
Attribution-Noncommercial License.
( http://creativecommons.org/licenses/by-nc/3.0/ )

This is an implementation of the Stochastic Average Gradient (SAG) algorithm: 
  Le Roux, Nicolas, Mark Schmidt, and Francis Bach.
  "A Stochastic Gradient Method with an Exponential Convergence _Rate for Finite Training Sets."
  Advances in Neural Information Processing Systems 25. 2012.
  http://books.nips.cc/papers/files/nips25/NIPS2012_1246.pdf
"""

from numpy import *
import scipy.linalg

class SAG(object):

    def __init__(self, f_df, theta, subfunction_references, args=(), kwargs={}, L=1., L_freq=0):
        """
        L is the Lipschitz constant.  Smaller corresponds to faster (but
        noisier) learning.  If L_freq is greater than 0, then every L_freq
        steps L will be adjusted as described in the SAG paper.
        """
        self.L = L
        self.L_freq = L_freq # how often to perform the extra function evaluations in order to test L

        self.N = len(subfunction_references)
        self.sub_ref = subfunction_references
        self.M = theta.shape[0]
        self.f_df = f_df
        self.args = args
        self.kwargs = kwargs

        self.num_steps = 0
        self.theta = theta.copy().reshape((-1,1))

        self.f = ones((self.N))*nan
        self.df = zeros((self.M,self.N))

        self.grad_sum = sum( self.df, axis=1 ).reshape((-1,1))

    def optimize(self, num_passes = 10, num_steps = None):
        if num_steps==None:
            num_steps = num_passes*self.N
        for i in range(num_steps):
            if not self.optimization_step():
                break
        return self.theta

    def optimization_step(self):
        # choose a subfunction at random
        ind = int(floor(self.N*random.rand()))

        # calculate the objective function and gradient for the subfunction
        fl, dfl = self.f_df(self.theta, (self.sub_ref[ind], ), *self.args, **self.kwargs)
        dfl = dfl.reshape(-1,1)
        # store them
        self.f[ind] = fl
        self.grad_sum += (dfl - self.df[:,[ind]]) # TODO this may slowly accumulate errors. occasionally do the full sum?
        self.df[:,ind] = dfl.flat

        # only adjust the learning rate with frequency L_freq, to reduce computational load
        if self.L_freq > 0 and mod(self.num_steps, self.L_freq) == 0:
            # adapt the learning rate
            self.L *= 2**(-1. / self.N)
            theta_shift = self.theta - dfl / self.L
            # evaluate the objective function at theta_shift
            fl_shift, _ = self.f_df(theta_shift, (self.sub_ref[ind], ), *self.args, **self.kwargs)
            # test whether the change in objective satisfies Lip. inequality, otherwise increase constant
            if fl_shift - fl > -sum((dfl)**2) / (2.*self.L):
                self.L *= 2.

        self.num_steps += 1

        # take a gradient descent step
        div = min([self.num_steps, self.N])
        delta_theta = -self.grad_sum / self.L / div
        self.theta += delta_theta

        if not isfinite(fl):
            print("Non-finite subfunction.  Ending run.")
            return False
        return True
