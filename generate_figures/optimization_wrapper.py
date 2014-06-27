"""
Trains the model using a variety of optimization algorithms.
This class also wraps the objective and gradient of the model,
so that it can store a history of the objective during
optimization.

This is slower than directly calling the optimizers, because
it periodically evaluates (and stores) the FULL objective rather
than always evaluating only a single subfunction per update step.

Designed to be used by figure_comparison*.py.

Author: Jascha Sohl-Dickstein (2014)
Web: http://redwood.berkeley.edu/jascha
This software is made available under the Creative Commons
Attribution-Noncommercial License.
( http://creativecommons.org/licenses/by-nc/3.0/ )
"""

# allow SFO to be imported despite being in the parent directoy
import sys
sys.path.append("..")
sys.path.append(".")

from sfo import SFO
from sag import SAG
from adagrad import ADAGrad
from collections import defaultdict
from itertools import product
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

# numpy < 1.7 does not have np.random.choice
def my_random_choice(n, k, replace):
    perm = np.random.permutation(n)
    return perm[:k]
if hasattr(np.random, 'choice'):
    random_choice = np.random.choice
else:
    random_choice = my_random_choice


class train:
    """
    Trains the model using a variety of optimization algorithms.
    This class also wraps the objective and gradient of the model,
    so that it can evaluate and store the full objective for each
    step in the optimization.

    This is WAY SLOWER than just calling the optimizers, because
    it evaluates the FULL objective and gradient instead of a single
    subfunction several times per pass.

    Designed to be used by figure_convergence.py.
    """

    def __init__(self, model, calculate_full_objective=True, num_projection_dims=5, full_objective_per_pass=4):
        """
        Trains the model using a variety of optimization algorithms.
        This class also wraps the objective and gradient of the model,
        so that it can evaluate and store the full objective for each
        step in the optimization.

        This is WAY SLOWER than just calling the optimizers, because
        it evaluates the FULL objective and gradient instead of a single
        subfunction several times per pass.

        Designed to be used by figure_convergence.py.
        """

        self.model = model
        self.history = {'f':defaultdict(list), 'x_projection':defaultdict(list), 'events':defaultdict(list), 'x':defaultdict(list)}

        # we use SFO to flatten/unflatten parameters for the other optimizers
        self.x_map = SFO(self.model.f_df, self.model.theta_init, self.model.subfunction_references)
        self.xinit_flat = self.x_map.theta_original_to_flat(self.model.theta_init)
        self.calculate_full_objective = calculate_full_objective

        M = self.xinit_flat.shape[0]
        self.x_projection_matrix = np.random.randn(num_projection_dims, M)/np.sqrt(M)

        self.num_subfunctions = len(self.model.subfunction_references)
        self.full_objective_period = int(self.num_subfunctions/full_objective_per_pass)


    def f_df_wrapper(self, *args, **kwargs):
        """
        This (slightly hacky) function stands between the optimizer and the objective function.
        It evaluates the objective on the full function every full_objective_function times a 
        subfunction is evaluated, and stores the history of the full objective function value.
        """

        ## call the true subfunction objective function, passing through all parameters
        f, df = self.model.f_df(*args, **kwargs)

        if len(self.history['f'][self.learner_name]) == 0:
            # this is the first time step for this learner
            self.last_f = np.inf
            self.last_idx = -1
            self.nsteps_this_learner = 0

        self.nsteps_this_learner += 1
        # only record the step every once every self.full_objective_period steps
        if np.mod(self.nsteps_this_learner, self.full_objective_period) != 1 and self.full_objective_period > 1:
            return f, df

        # the full objective function on all subfunctions
        if self.calculate_full_objective:
            new_f = 0.
            for ref in self.model.full_objective_references:
                new_f += self.model.f_df(args[0], ref)[0]
        else:
            new_f = f

        events = dict() # holds anything special about this step
        # a unique identifier for the current subfunction
        new_idx = id(args[1])
        if 'SFO' in self.learner_name:
            events = dict(self.optimizer.events)
        # append the full objective value, projections, etc to the history
        self.history['f'][self.learner_name].append(new_f)
        x_proj = np.dot(self.x_projection_matrix, self.x_map.theta_original_to_flat(args[0])).ravel()
        self.history['x_projection'][self.learner_name].append(x_proj)
        self.history['events'][self.learner_name].append(events)
        self.history['x'][self.learner_name] = args[0]
        print("full f %g"%(new_f))
        # store the prior values
        self.last_f = new_f
        self.last_idx = new_idx

        return f, df


    def f_df_wrapper_flattened(self, x_flat, subfunction_references, *args, **kwargs):
        """
        Calculate the subfunction objective and gradient.
        Takes a 1d parameter vector, and returns a 1d gradient, even
        if the parameters for f_df are a list or a dictionary.
        x_flat should be the flattened version of the parameters.
        """

        x = self.x_map.theta_flat_to_original(x_flat)
        f = 0.
        df = 0.
        for sr in subfunction_references:
            fl, dfl = self.f_df_wrapper(x, sr, *args, **kwargs)
            dfl_flat = self.x_map.theta_original_to_flat(dfl)
            f += fl
            df += dfl_flat
        return f, df.ravel()


    def SGD(self, num_passes=20):
        """ Train model using SGD with various learning rates """

        # get the number of minibatches
        N = len(self.model.subfunction_references)
        # step through all the hyperparameters.  eta is step length.
        for eta in 10**np.linspace(-5,2,8):
            # label this convergence trace using the optimizer name and hyperparameter
            self.learner_name = "SGD %.4f"%eta
            print("\n\n" + self.learner_name)

            # initialize the parameters
            x = self.xinit_flat.copy()
            ## perform stochastic gradient descent
            for _ in range(num_passes*N): # number of minibatch evaluations
                # choose a minibatch at random
                idx = np.random.randint(N)
                sr = self.model.subfunction_references[idx]
                # evaluate the objective and gradient for that minibatch
                fl, dfl = self.f_df_wrapper_flattened(x.reshape((-1,1)), (sr,))
                # update the parameters
                x -= dfl.reshape(x.shape) * eta
                # if the objective has diverged, skip the rest of the run for this hyperparameter
                if not np.isfinite(fl):
                    print("Non-finite subfunction.")
                    break


    def LBFGS(self, num_passes=20):
        """ Train model using LBFGS """

        self.learner_name = "LBFGS"
        print("\n\n" + self.learner_name)
        _, _, _ = fmin_l_bfgs_b(
            self.f_df_wrapper_flattened,
            self.xinit_flat.copy(), 
            disp=1,
            args=(self.model.subfunction_references, ),
            maxfun=num_passes)


    def SFO(self, num_passes=20, learner_name='SFO', **kwargs):
        """ Train model using SFO."""
        self.learner_name = learner_name
        print("\n\n" + self.learner_name)

        self.optimizer = SFO(self.f_df_wrapper, self.model.theta_init, self.model.subfunction_references, **kwargs)
        # # check the gradients
        # self.optimizer.check_grad()
        x = self.optimizer.optimize(num_passes=num_passes)


    def SFO_variations(self, num_passes=20):
        """
        Train model using several variations on the standard SFO algorithm.
        """

        np.random.seed(0) # make experiments repeatable
        self.learner_name = 'SFO standard'
        print("\n\n" + self.learner_name)
        self.optimizer = SFO(self.f_df_wrapper, self.model.theta_init, self.model.subfunction_references)
        x = self.optimizer.optimize(num_passes=num_passes)

        np.random.seed(0) # make experiments repeatable
        self.learner_name = 'SFO all active'
        print("\n\n" + self.learner_name)
        self.optimizer = SFO(self.f_df_wrapper, self.model.theta_init, self.model.subfunction_references,
            init_subf=len(self.model.subfunction_references))
        x = self.optimizer.optimize(num_passes=num_passes)

        np.random.seed(0) # make experiments repeatable
        self.learner_name = 'SFO rank 1'
        print("\n\n" + self.learner_name)
        self.optimizer = SFO(self.f_df_wrapper, self.model.theta_init, self.model.subfunction_references,
            hessian_algorithm='rank1')
        x = self.optimizer.optimize(num_passes=num_passes)

        self.learner_name = 'SFO random'
        print("\n\n" + self.learner_name)
        self.optimizer = SFO(self.f_df_wrapper, self.model.theta_init, self.model.subfunction_references,
            subfunction_selection='random'
            )
        x = self.optimizer.optimize(num_passes=num_passes)

        self.learner_name = 'SFO cyclic'
        print("\n\n" + self.learner_name)
        self.optimizer = SFO(self.f_df_wrapper, self.model.theta_init, self.model.subfunction_references,
            subfunction_selection='cyclic'
            )
        x = self.optimizer.optimize(num_passes=num_passes)


    def SAG(self, num_passes=20):
        """ Train model using SAG with line search, for various initial Lipschitz """

        # larger L is easier, so start large
        for L in 10**(-np.linspace(-3, 3, 7)):
            self.learner_name = "SAG %.4f"%L
            #learner_name = "SAG (diverges)"
            print("\n\n" + self.learner_name)
            self.optimizer = SAG(self.f_df_wrapper_flattened, self.xinit_flat.copy(), self.model.subfunction_references, L=L)
            x = self.optimizer.optimize(num_passes=num_passes)
            print(np.mean(self.optimizer.f), "average value at last evaluation")


    def LBFGS_minibatch(self, num_passes=20, data_fraction=0.1, num_steps=10):
        """ Perform LBFGS on minibatches of size data_fraction of the full datastep, and with num_steps LBFGS steps per minibatch."""

        self.learner_name = "LBFGS minibatch"


        x = self.xinit_flat.copy()
        for epoch in range(num_passes):
            idx = random_choice(len(self.model.subfunction_references),
                int(data_fraction*len(self.model.subfunction_references)),
                replace=False)
            sr = []
            for ii in idx:
                sr.append(self.model.subfunction_references[ii])
            x, _, _ = fmin_l_bfgs_b(
                self.f_df_wrapper_flattened,
                x, 
                args=(sr, ),
                disp=1,
                maxfun=num_steps)

    
    def SGD_momentum(self, num_passes=20):
        """ Train model using SGD with various learning rates and momentums"""

        learning_rates = 10**np.linspace(-5,2,8)
        momentums = np.array([0.5, 0.9, 0.95, 0.99])
        params = product(learning_rates, momentums)
        N = len(self.model.subfunction_references)
        for eta, momentum in params:
            self.learner_name = "SGD_momentum eta=%.5f, mu=%.2f" % (eta, momentum)
            print("\n\n" + self.learner_name)
            f = np.ones((N))*np.nan
            x = self.xinit_flat.copy()
            # Prevous step
            inc = 0.0
            for epoch in range(num_passes):
                for minibatch in range(N):
                    idx = np.random.randint(N)
                    sr = self.model.subfunction_references[idx]
                    fl, dfl = self.f_df_wrapper_flattened(x.reshape((-1,1)), (sr,))
                    inc = momentum * inc - eta * dfl.reshape(x.shape)
                    x += inc
                    f[idx] = fl
                    if not np.isfinite(fl):
                        print("Non-finite subfunction.  Ending run.")
                        break
                if not np.isfinite(fl):
                    print("Non-finite subfunction.  Ending run.")
                    break
            print(np.mean(f[np.isfinite(f)]), "average finite value at last evaluation")


    def ADA(self, num_passes=20):
        """ Train model using ADAgrad with various learning rates """

        for eta in 10**np.linspace(-3,1,5):
            self.learner_name = "ADAGrad %.4f"%eta
            print("\n\n" + self.learner_name)
            self.optimizer = ADAGrad(self.f_df_wrapper_flattened, self.xinit_flat.copy(), self.model.subfunction_references, learning_rate=eta)
            x = self.optimizer.optimize(num_passes=num_passes)
            print(np.mean(self.optimizer.f), "average value at last evaluation")

