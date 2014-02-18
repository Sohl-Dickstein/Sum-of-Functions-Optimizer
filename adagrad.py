"""
Author: Ben Poole, Jascha Sohl-Dickstein (2013)
This software is made available under the Creative Commons
Attribution-Noncommercial License.
( http://creativecommons.org/licenses/by-nc/3.0/ )

This is an implementation of the ADAGrad algorithm: 
    Duchi, J., Hazan, E., & Singer, Y. (2010).
    Adaptive subgradient methods for online learning and stochastic optimization.
    Journal of Machine Learning Research
    http://www.eecs.berkeley.edu/Pubs/TechRpts/2010/EECS-2010-24.pdf
"""

from numpy import *
import numpy as np

class ADAGrad(object):

    def __init__(self, f_df, theta, subfunction_references, reps=1, learning_rate=0.1, args=(), kwargs={}):

        self.reps = reps
        self.learning_rate = learning_rate

        self.N = len(subfunction_references)
        self.sub_ref = subfunction_references
        self.f_df = f_df
        self.args = args
        self.kwargs = kwargs

        self.num_steps = 0
        self.theta = theta.copy().reshape((-1,1))
        self.grad_history = np.zeros_like(self.theta)
        self.M = self.theta.shape[0]

        self.f = ones((self.N))*np.nan


    def optimize(self, num_passes = 10, num_steps = None):
        if num_steps==None:
            num_steps = num_passes*self.N
        for i in range(num_steps):
            if not self.optimization_step():
                break
        #print 'L ', self.L
        return self.theta

    def optimization_step(self):
        idx = np.random.randint(self.N)
        gradii = np.zeros_like(self.theta)
        lossii = 0.
        for i in range(self.reps):
            lossi, gradi = self.f_df(self.theta, (self.sub_ref[idx], ), *self.args, **self.kwargs)
            lossii += lossi / self.reps
            gradii += gradi.reshape(gradii.shape) / self.reps

        self.num_steps += 1
        learning_rates = self.learning_rate / (np.sqrt(1./self.num_steps + self.grad_history))
        learning_rates[np.isinf(learning_rates)] = self.learning_rate
        self.theta -= learning_rates * gradii
        self.grad_history += gradii**2
        self.f[idx] = lossii

        if not np.isfinite(lossii):
            print("Non-finite subfunction.  Ending run.")
            return False
        return True

