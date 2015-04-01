"""
Author: Jascha Sohl-Dickstein (2014)
This software is made available under the Creative Commons
Attribution-Noncommercial License.
( http://creativecommons.org/licenses/by-nc/3.0/ )
"""

#from __future__ import print_function
import numpy as np
from random import shuffle

# TODO remove this import.  I believe numpy functions are always
# called using "np." now, but keeping this in case I've missed 
# an edge case.
from numpy import *

import time
import warnings

class SFO(object):
    def __init__(self, f_df, theta, subfunction_references, args=(), kwargs={},
        display=2, max_history_terms=10, hessian_init=1e5, init_subf=2,
        hess_max_dev = 1e8, hessian_algorithm='bfgs',
        subfunction_selection='distance all',
        # subfunction_selection='distance both',
        max_gradient_noise=1.,
        max_step_length_ratio=2., minimum_step_length=1e-6, maximum_step_length=1e8,
        update_sdev=0, 
        lead_ahead=None, 
        # lead_ahead='nesterov accumulate only',
        subspace_momentum=20,
        global_rect=False,
        simple_hess_init=True,
        use_natgrad=True):
        """
        The main Sum of Functions Optimizer (SFO) class.

        Parameters:
        f_df - Returns the function value and gradient for a single subfunction 
            call.  Should have the form
                f, dfdtheta = f_df(theta, subfunction_references[idx],
                                      *args, **kwargs)
            where idx is the index of a single subfunction.
        theta - The initial parameters to be used for optimization.  theta can
            be either a NumPy array, an array of NumPy arrays, a dictionary
            of NumPy arrays, or a nested variation thereof.  The gradient returned
            by f_df should have the same form as theta.
        subfunction_references - A list containing an identifying element for
            each subfunction.  The elements in this list could be, eg, numpy
            matrices containing minibatches, or indices identifying the
            subfunction, or filenames from which target data should be read.

        Optional Parameters, roughly in order of decreasing utility:
        args=() - This list will be passed through as *args to f_df.
        kwargs={} - This dictionary will be passed through as **kwargs to f_df.
        display=2 - Display level.  1 = once per SFO call, 2 = once per
            subfunction call, larger = debugging info
        max_history_terms=10 - The number of history terms to use for the
            BFGS updates.
        minimum_step_length=1e-8 - The shortest step length allowed. Any update
            steps shorter than this will be made this length. Set this so as to
            prevent numerical errors when computing the difference in gradients
            before and after a step.
        max_step_length_ratio=10 - The length of the longest allowed update step, 
            relative to the average length of prior update steps. Takes effect 
            after the first full pass through the data.
        hessian_init=1e5 - The initial estimate of the Hessian for the first
            init_subf subfunctions is set to this value times the identity
            matrix.  This number should be large, but not so large that there
            are numerical errors when the first update step has length
            [initial gradient]/[hessian_init].
        init_subf=2 - The number of active subfunctions at the start of
            learning.  Inactive subfunctions will be activated as the signal to
            noise in the optimization steps drop during optimization.
        hessian_algorithm='bfgs' - Use BFGS ('bfgs') or rank 1 ('rank1') updates
            to compute the approximate Hessian.
        subfunction_selection='distance' - The algorithm to use to choose the
            next subfunction to evaluated.  This can be maximum distance
            ('distance'), random ('random'), or cyclic ('cyclic').
        max_gradient_noise=1. - The maximum ratio of standard error in the
            gradient across subfunctions to the length of the average gradient
            across subfunctions.  If this ratio is exceeded, the number of
            active subfunctions is increased.

        See README.md for example code.
        """

        self.display = display
        self.f_df = f_df
        self.args = args
        self.kwargs = kwargs
        self.max_history = max_history_terms
        self.max_gradient_noise = max_gradient_noise
        self.hess_max_dev = hess_max_dev
        self.hessian_init = hessian_init
        self.N = len(subfunction_references)
        self.sub_ref = subfunction_references
        self.hessian_algorithm = hessian_algorithm.lower()
        self.subfunction_selection = subfunction_selection.lower()
        self.max_step_length_ratio = max_step_length_ratio
        # theta, in its original format
        self.theta_original = theta
        # theta, as a list of numpy arrays
        self.theta_list = self.theta_original_to_list(self.theta_original)
        # theta, flattented into a 1d array
        self.theta = self.theta_list_to_flat(self.theta_list)
        # theta from the previous learning step -- initialize to theta
        self.theta_prior_step = self.theta.copy()
        # number of data dimensions
        self.M = self.theta.shape[0]
        self.global_rect = global_rect

        self.use_natgrad = use_natgrad
        self.geometric_average_natgrad = True

        # if subspace_momentum == True:
        #     subspace_momentum = int(np.ceil(np.log(self.N)/np.log(2)))
        self.subspace_momentum = subspace_momentum

        if self.subspace_momentum:
            subspace_dimensionality = 2*self.subspace_momentum+2  # 2 to include current location
        else:
            subspace_dimensionality = 2*self.N+2  # 2 to include current location

        # subspace can't be larger than the full space
        subspace_dimensionality = int(np.min([subspace_dimensionality, self.M]))
        # the update steps will be rescaled by this
        self.step_scale = 1.
        # "very small" for various tasks, most importantly identifying when
        # update steps are too small to be used for Hessian updates without
        # incurring large numerical errors.
        self.minimum_step_length = minimum_step_length
        self.maximum_step_length = maximum_step_length
        self.update_sdev = update_sdev
        self.lead_ahead = lead_ahead
        self.eps = 1e-12

        self.simple_hess_init = simple_hess_init

        # # DEBUG NOTE float32 cause errors to accumulate, AND makes it run much slower on the cpu
        self.Pdtype = np.float64
        # self.Pdtype = np.float32 
        # self.eps = 1e-9

        # DEBUG
        self.reduced_curvature = False

        # the min and max dimenstionality for the subspace
        self.K_min = subspace_dimensionality
        self.K_max = np.ceil(self.K_min*1.5)
        self.K_max = int(np.min([self.K_max, self.M]))
        # self.P holds the subspace
        self.P = np.zeros((self.M,self.K_max), dtype=self.Pdtype)

        # store the minimum and maximum eigenvalue from each approximate
        # Hessian
        self.min_eig_sub = np.zeros((self.N))
        self.max_eig_sub = np.zeros((self.N))

        # store the total time spent in optimization, and the amount of time
        # spent in the objective function
        self.time_pass = 0.
        self.time_func = 0.

        # how many steps since the active set size was increased
        self.iter_since_active_growth = 0

        # which subfunctions are active
        self.active = np.zeros((self.N)).astype(bool)
        inds = np.random.permutation(self.N)[:init_subf]
        self.active[inds] = True
        self.min_eig_sub[inds] = hessian_init
        self.max_eig_sub[inds] = hessian_init

        # the total path length traveled during optimization
        self.total_distance = 0.
        # number of function evaluations for each subfunction
        self.eval_count = np.zeros((self.N))
        self.eval_count_total = 0
        # how many times the diagonal hessian approximation has been updated
        self.natgrad_count = 0

        # the current dimensionality of the subspace
        self.K_current = 1
        # set the first column of the subspace to be the initial
        # theta
        rr = np.sqrt(np.sum(self.theta**2))
        if rr > 0:
            self.P[:,[0]] = self.theta/rr
        else:
            # initial theta is 0 -- initialize randomly
            self.P[:,[0]] = np.random.randn(self.theta.shape[0],1)
            self.P[:,[0]] /= np.sqrt(np.sum(self.P[:,[0]]**2))

        if self.M == self.K_max:
            # if the subspace spans the full space, then just make
            # P the identity matrix
            if self.display > 1:
                print("subspace spans full space"),
            self.P = np.eye(self.M, dtype=self.Pdtype)
            self.K_current = self.M+1

        self.df_accum = 0.
        if self.subspace_momentum:
            # weighted directions
            self.momentum_theta = self.theta * np.ones((1,self.subspace_momentum))
            self.momentum_df = np.random.randn(self.M, self.subspace_momentum)*self.eps

        # theta projected into current working subspace
        self.theta_proj = np.dot(self.P.T, self.theta)
        # holds the last position and the last gradient for all the objective functions
        self.last_theta = np.tile(self.theta_proj, ((1,self.N)))
        self.last_df = np.zeros((self.K_max,self.N))
        # the history of theta changes for each subfunction
        self.hist_deltatheta = np.zeros((self.K_max,max_history_terms,self.N))
        # the history of gradient changes for each subfunction
        self.hist_deltadf = np.zeros((self.K_max,max_history_terms,self.N))
        # the history of function values for each subfunction
        self.hist_f = np.ones((self.N, max_history_terms))*np.nan
        # a flat history of all returned subfunction values for debugging/diagnostics
        self.hist_f_flat = []

        self.lossy = np.ones((self.N,)).astype(bool)

        # the approximate Hessian for each subfunction is stored
        # as np.dot(self.b[:.:.index], self.b[:.:.inedx].T)
        self.b = np.zeros((self.K_max,self.K_max,self.N)).astype(complex)
        # self.b = np.zeros((self.K_max,2*self.max_history,self.N)).astype(complex)
        # TODO(jascha) self.b could be real if another diagonal term carrying sign
        # information was introduced

        # the full Hessian (sum over all the subfunctions)
        self.full_H = np.zeros((self.K_max,self.K_max))

        # holds diagnostic information (eg, step failure, when the subspace is collapsed)
        self.events = dict()

        # used to keep track of the current subfunction if the update order is
        # cyclic (subfunction_selection='cyclic')
        self.cyclic_subfunction_index = 0

        self.nat_grad_rescale = 1.
        self.total_grad_variance = 1e-4
        self.ddf_total_sqr = self.eps
        self.ddtheta_total_sqr = self.eps

        if False: #self.N < 25 and self.display > 0:
            print(
                "\nIn experiments, performance suffered when the data was broken up into fewer\n"
                "than 25 minibatches.  See Figure 2c in SFO paper.\n"
                "You may want to use more than the current %d minibatches.\n"%(self.N))


    def optimize(self, num_passes = 10, num_steps = None):
        """
        Optimize the objective function.  num_steps is the number of subfunction calls to make,
        and num_passes is the number of effective passes through all subfunctions to make.  If
        both are provided then num_steps takes precedence.

        This, __init__, and check_grad are the only three functions that should be called by
        the user
        """
        if num_steps==None:
            num_steps = int(num_passes*self.N)
        for i in range(num_steps):
            if self.display > 1:
                print("pass {0}, step {1},".format(float(self.eval_count_total)/self.N, i)),
            self.optimization_step()
            if self.display > 1:
                print("active {0}/{1}, sfo time {2} s, func time {3} s, f {4}, <f> {5}".format(np.sum(self.active), self.active.shape[0], self.time_pass - self.time_func, self.time_func, self.hist_f_flat[-1], np.mean(self.hist_f[self.eval_count>0,0])))
        if num_steps < 1:
            print("No optimization steps performed.  Change num_passes or num_steps.")
        # elif self.display > 0:
        #     print("active {0}/{1}, pass #{2}, sfo {3} s, func {4} s, <f> {5}".format(np.sum(self.active), self.active.shape[0], float(self.eval_count_total)/self.N, self.time_pass - self.time_func, self.time_func, np.mean(self.hist_f[self.eval_count>0,0])))
        #     if (self.time_pass - self.time_func) > self.time_func and self.N >= 25 and self.time_pass > 60:
        #         print("More time was spent in SFO than the objective function.")
        #         print("You may want to consider breaking your data into fewer minibatches to reduce overhead.")

        # reverse the flattening transformation on theta
        theta_flat_unnat = self.theta/self.nat_grad_rescale
        return self.theta_flat_to_original(theta_flat_unnat)


    def check_grad(self, small_diff = None):
        """
        A diagnostic function to check the gradients for the subfunctions.  It
        checks the subfunctions in random order, and the dimensions of each
        subfunction in random order.  This way, a representitive set of
        gradients can be checked quickly, even for high dimensional objectives.

        small_diff specifies the size of the step size used for finite difference
        comparison.
        """

        if small_diff is None:
            # step size to use for gradient check
            small_diff = self.eps*1e6
        print("Testing step size %g"%small_diff)

        for i in np.random.permutation(range(self.N)):
            fl, dfl = self.f_df_wrapper(self.theta, i, return_full=True)
            ep = np.zeros((self.M,1))
            dfl_obs = np.zeros((self.M,1))
            dfl_err = np.zeros((self.M,1))
            for j in np.random.permutation(range(self.M)):
                ep[j] = small_diff
                fl2, _ = self.f_df_wrapper(self.theta + ep, i, return_full=True)
                dfl_obs[j] = (fl2 - fl)/small_diff
                dfl_err[j] = dfl_obs[j] - dfl[j]
                if not np.isfinite(dfl_err[j]):
                    print("non-finite "),
                elif np.abs(dfl_err[j]) > small_diff * 1e4:
                    print("large diff "),
                else:
                    print("           "),
                print("  gradient subfunction {0}, dimension {1}, analytic {2}, finite diff {3}, error {4}".format(i, j, dfl[j], dfl_obs[j], dfl_err[j]))
                ep[j] = 0.
            gerr = np.sqrt(np.sum((dfl - dfl_obs)**2))
            print("subfunction {0}, total L2 gradient error {1}".format(i, gerr))
            print


    def replace_subfunction(self, idx, keep_history=False, new_reference=None):
        """
        Reset the history for the subfunction indicated by idx.  This could be used
        e.g. to apply SFO in an online fashion, where this function is used to refresh
        the data in minibatches.

        Parameters:
            idx - The subfunction number to replace (0 <= idx < # subfunctions)
            keep_history - Set this to True to keep the history of gradient and iterate
                updates, and only replace the most recent position and gradient so that
                future updates are based on the new subfunction.
            new_reference - If specified, the subfunction_reference that is passed
                to the objective function will be replaced with this value.
                new_reference could for instance hold a new minibatch of training data.
        """

        if not new_reference is None:
            # if a new one has been supplied, then
            # replace the reference for this subfunction
            self.sub_ref[idx] = new_reference

        if self.display > 5:
            print("replacing %d"%idx)

        if self.eval_count[idx] < 1:
            # subfunction not active yet -- no history
            return

        if not keep_history:
            # destroy the history of delta-gradients and delta-positions for this subfunction
            self.hist_deltatheta[:,:,idx] = 0.
            self.hist_deltadf[:,:,idx] = 0.

        # reset eval count.
        # 0, not 1, so that it will also look in the direction this new gradient
        # pushes us
        self.eval_count[idx] = 0

        # # evaluate this subfunction at the last location
        # # (use the last location, to avoid weird interactions with rejected updates)
        # f, df_proj = self.f_df_wrapper(self.theta_prior_step, idx)
        # # replace the last function, gradient, and position with the one just
        # # measured.  set skip_delta so that the change in gradient over 
        # # the change in subfunction is not used.
        # theta_lastpos_proj = np.dot(self.P.T, self.theta_prior_step)
        # self.update_history(idx, theta_lastpos_proj, f, df_proj, skip_delta=True)
        # TODO get rid of skip_delta in update_history.  that's now handled
        # automagically by the eval_count = 0


    def apply_subspace_transformation(self,T_left,T_right):
        """
        Apply change-of-subspace transformation.  This function is called when
        the subspace is collapsed to project into the new lower dimensional
        subspace.
        T_left - The covariant subspace to subspace projection matrix.
        T_right - The contravariant subspace projection matrix.

        (note that currently T_left = T_right always since the subspace is
        orthogonal.  This will change if eg the code is adapted to also
        incorporate a "natural gradient" based parameter space transformation.)
        """

        ss = T_left.shape[1]
        tt = T_left.shape[0]

        # project history terms into new subspace
        self.last_df = np.dot(T_right.T, self.last_df)
        self.last_theta = np.dot(T_left, self.last_theta)
        self.hist_deltadf = np.dot(T_right.T, self.hist_deltadf.reshape((ss,-1))).reshape((tt,-1,self.N))
        self.hist_deltatheta = np.dot(T_left, self.hist_deltatheta.reshape((ss,-1))).reshape((tt,-1,self.N))
        # project stored hessian for each subfunction in to new subspace
        self.b = np.dot(T_right.T, self.b.reshape((ss,-1))).reshape((tt,self.K_max,self.N))

        # # project stored full hessian in to new subspace
        # # TODO recompute full hessian from scratch to avoid accumulating numerical errors?
        # self.full_H = np.dot(T_right.T, self.full_H)
        # self.full_H = np.dot(T_right.T, self.full_H.T).T
        # # project low dimensional representation of current theta in to new subspace
        # self.theta_proj = np.dot(T_left, self.theta_proj)

        ## To avoid slow accumulation of numerical errors, recompute full_H
        ## and theta_proj when the subspace is collapsed.  Should not be a
        ## leading time cost.
        # theta projected into current working subspace
        self.theta_proj = np.dot(self.P.T, self.theta)
        # full approximate hessian
        self.full_H = np.real(np.dot(self.b.reshape((ss,-1)), self.b.reshape((ss,-1)).T))


    def reorthogonalize_subspace(self):
        # check if the subspace has become non-orthogonal
        CC = np.dot(self.P.T, self.P)
 
        # check if the subspace has become non-orthogonal
        subspace_eigs, subspace_evecs = np.linalg.eigh(CC)
        # TODO(jascha) this may be a stricter cutoff than we need
        if np.max(subspace_eigs) <= 1 + self.eps:
            return

        if self.display > 2:
            print("Subspace has become non-orthogonal. Reorthogonalizing.\n")

        Pl = np.zeros((self.K_max, self.K_max))

        # DEBUG -- changed reorthogonalization method to be hopefully cheaper
        # Porth, _ = np.linalg.qr(self.P[:,:self.K_current])
        # Pl[:,:self.K_current] = np.dot(self.P.T, Porth)

        CC_sqrtm = np.dot(subspace_evecs*np.sqrt(subspace_eigs).reshape((1,-1)), subspace_evecs.T)
        CC_sqrtm = CC_sqrtm[:self.K_current,:self.K_current]
        Pl[:self.K_current,:self.K_current] = np.linalg.inv(CC_sqrtm)

        # update the subspace;
        # self.P[:,:self.K_current] = Porth
        # self.P[:,:self.K_current] = np.dot(self.P[:,:self.K_current], Pl)
        self.P = np.dot(self.P, Pl)

        # Pl is the projection matrix from old to new basis.  apply it to all the history
        # terms
        self.apply_subspace_transformation(Pl.T, Pl);


    def collapse_subspace(self, xl=None):
        """
        Collapse the subspace to its smallest dimensionality.

        xl is a new direction that may not be in the history yet, so we pass
        it in explicitly to make sure it's included.
        """

        if self.display > 2:
            print()
            print("collapsing subspace"),

        # the projection matrix from old to new subspace
        Pl = np.zeros((self.K_max,self.K_max))

        # yy will hold all the directions to pack into the subspace.
        # initialize it with random noise, so that it still spans K_min
        # dimensions even if not all the subfunctions are active yet
        yy = np.random.randn(self.K_max,self.K_min)
        if xl is None:
            xl = np.random.randn(self.K_max,1)

        if self.subspace_momentum:
            # the multiscale history of recent positions and gradients
            # as well as the current position and gradient (which will not be saved in the history yet)

            momentum_theta_proj = np.dot(self.P.T,self.momentum_theta)
            momentum_df_proj = np.dot(self.P.T,self.momentum_df)
            yz = np.hstack((momentum_theta_proj, momentum_df_proj, xl, np.dot(self.P.T,self.theta)))
        else:
            # the most recent position and gradient for all active subfunctions,
            # as well as the current position and gradient (which will not be saved in the history yet)
            yz = np.hstack((self.last_df[:,self.active], self.last_theta[:,self.active], xl, np.dot(self.P.T,self.theta)))

        yy[:,:yz.shape[1]] = yz
        Pl[:,:self.K_min] = np.linalg.qr(yy)[0]

        # update the subspace
        self.P = np.dot(self.P, Pl.astype(self.Pdtype)).astype(self.Pdtype)

        # Pl is the projection matrix from old to new basis.  apply it to all the history
        # terms
        self.apply_subspace_transformation(Pl.T, Pl)

        # update the stored subspace size
        self.K_current = self.K_min

        # re-orthogonalize the subspace if it's accumulated small errors
        self.reorthogonalize_subspace()

        # since we're losing parameter consistency anyway,
        # this is a great time to reset the natural gradient
        if self.use_natgrad:
            self.update_natgrad()
        # make sure we don't use a change over a subspace reset
        self.lossy[:] = True


    def update_subspace(self, x_in):
        """
        Update the low dimensional subspace by adding a new direction.
        x_in - The new vector to incorporate into the subspace.
        """
        if self.K_current >= self.M:
            # no need to update the subspace if it spans the full space
            return
        if np.sum(~np.isfinite(x_in)) > 0:
            # bad vector!  bail.
            return
        x_in_length = np.sqrt(np.sum(x_in**2))
        if x_in_length < self.eps:
            # if the new vector is too short, nothing to do
            return
        # make x unit length
        xnew = x_in/x_in_length

        # Find the component of x pointing out of the existing subspace.
        # We need to do this multiple times for numerical stability.
        for i in range(3):
            xnew -= np.dot(self.P, np.dot(self.P.T, xnew))
            ss = np.sqrt(np.sum(xnew**2))
            if ss < self.eps:
                # it barely points out of the existing subspace
                # no need to add a new direction to the subspace
                return
            # make it unit length
            xnew /= ss
            # if it was already largely orthogonal then numerical
            # stability will be good enough
            # TODO replace this with a more principled test
            if ss > 0.1:
                break

        # add a new column to the subspace containing the new direction
        self.P[:,self.K_current] = xnew[:,0]
        self.K_current += 1

        self.events['collapse subspace'] = False
        if self.K_current >= self.K_max:
            # the subspace has exceeded its maximum allowed size -- collapse it
            self.events['collapse subspace'] = True
            # xl may not be in the history yet, so we pass it in explicitly to make
            # sure it's used
            xl = np.dot(self.P.T, x_in)
            self.collapse_subspace(xl=xl)


    def get_full_H_with_diagonal(self):
        """
        Get the full approximate Hessian, including the diagonal terms.
        (note that self.full_H is stored without including the diagonal terms)
        """
        full_H_combined = self.full_H + np.eye(self.K_max)*np.sum(self.min_eig_sub[self.active])

        if self.global_rect:
            # rectify the eigenvalues
            U, V = np.linalg.eigh(full_H_combined)
            # set any too-small eigenvalues to the median positive
            # eigenvalue
            # DEBUG
            U_median = np.median(U[U>0])
            U[(U<(np.max(np.abs(U))/self.hess_max_dev))] = U_median
            # U_min = np.sum(self.min_eig_sub[self.active])
            # U[(U<(np.max(np.abs(U))/self.hess_max_dev))] = U_min
            full_H_combined = np.dot(V*U, V.T)

        return full_H_combined


    def get_predicted_subf(self, indx, theta_proj):
        """
        Get the predicted value of subfunction idx at theta_proj
        (where theat_proj is in the subspace)
        """
        dtheta = theta_proj - self.last_theta[:,[indx]]
        bdtheta = np.dot(self.b[:,:,indx].T, dtheta)
        Hdtheta = np.real(np.dot(self.b[:,:,indx], bdtheta))
        Hdtheta += dtheta*self.min_eig_sub[indx] # the diagonal contribution
        # df_pred = self.last_df[:,[indx]] + Hdtheta
        f_pred = self.hist_f[indx,0] + np.dot(self.last_df[:,[indx]].T, dtheta)[0,0] + 0.5*np.dot(dtheta.T, Hdtheta)[0,0]
        return f_pred


    def update_history(self, indx, theta_proj, f, df_proj, skip_delta=False):
        """
        Update history of position differences and gradient differences
        for subfunction indx.
        """

        # differences in gradient and position
        ddf = df_proj - self.last_df[:,[indx]]
        ddt = theta_proj - self.last_theta[:,[indx]]
        # length of gradient and position change vectors
        lddt = np.sqrt(np.sum(ddt**2))
        lddf = np.sqrt(np.sum(ddf**2))
        diag_approx = lddf/lddt

        # there needs to be at least one earlier measurement from this
        # subfunction to compute position and gradient differences.
        new_bfgs_term = False
        if      self.eval_count[indx] > 1 \
                and not skip_delta \
                and (not self.lossy[indx] or not self.subspace_momentum):

            corr_ddf_ddt = np.dot(ddf.T, ddt)[0,0]/(lddt*lddf)

            if self.display > 3 and corr_ddf_ddt < 0:
                print("Warning!  Negative dgradient dtheta inner product.  Adding it anyway."),            
            if lddt < self.eps:
                if self.display > 2:
                    print("Largest change in theta too small ({0}).  Not adding to history.".format(lddt))
            elif lddf < self.eps:
                if self.display > 2:
                    print("Largest change in gradient too small ({0}).  Not adding to history.".format(lddf))
            elif np.abs(corr_ddf_ddt) < self.eps:
                if self.display > 2:
                    print("Inner product between dgradient and dtheta too small ({0}).  Not adding to history.".format(corr_ddf_ddt))
            else:
                if self.display > 3:
                    print("subf ||dtheta|| {0}, subf ||ddf|| {1}, corr(ddf,dtheta) {2},".format(lddt, lddf, np.sum(ddt*ddf)/(lddt*lddf))),

                # shift the history by one timestep
                self.hist_deltatheta[:,1:,indx] = self.hist_deltatheta[:,:-1,indx]
                # store the difference in theta since the subfunction was last evaluated
                self.hist_deltatheta[:,[0],indx] = ddt          / lddt # DEBUG -- make unit length
                # do the same thing for the change in gradient
                self.hist_deltadf[:,1:,indx] = self.hist_deltadf[:,:-1,indx]
                self.hist_deltadf[:,[0],indx] = ddf             / lddt # DEBUG
                new_bfgs_term = True

                if self.use_natgrad:
                    if self.geometric_average_natgrad:
                        mom = 1. / (self.eval_count_total + 1.)
                        self.ddf_total_sqr     = self.ddf_total_sqr    *(1.-mom) + np.log(np.dot(self.P, ddf)**2 + self.eps)*mom
                        self.ddtheta_total_sqr = self.ddtheta_total_sqr*(1.-mom) + np.log(np.dot(self.P, ddt)**2 + self.eps)*mom
                    else:
                        self.ddf_total_sqr     += np.dot(self.P, ddf)**2
                        self.ddtheta_total_sqr += np.dot(self.P, ddt)**2

        # else:
        #     self.min_eig_sub[indx] = diag_approx

        self.last_theta[:,[indx]] = theta_proj
        self.last_df[:,[indx]] = df_proj
        self.hist_f[indx,1:] = self.hist_f[indx,:-1]
        self.hist_f[indx,0] = f
        self.lossy[indx] = False

        # the current contribution from this subfunction to the total Hessian approximation
        H_pre_update = np.real(np.dot(self.b[:,:,indx], self.b[:,:,indx].T))
        ## update this subfunction's Hessian estimate
        self.update_hessian(indx, new_bfgs_term, diag_approx)
        # the new contribution from this subfunction to the total approximate hessian
        H_new = np.real(np.dot(self.b[:,:,indx], self.b[:,:,indx].T))   
        # update total Hessian using this subfunction's updated contribution
        self.full_H += H_new - H_pre_update


    def update_hessian(self, indx, new_bfgs_term, diag_approx):
        """
        Update the Hessian approximation for a single subfunction.
        indx - The index of the target subfunction for Hessian update.
        """

        if np.sum(self.hist_deltatheta[:,0,indx]**2) <= 0:
        # if np.sum(self.hist_deltatheta[:,1,indx]**2) <= 0:
            # no history -- initialize with the median eigenvalue from full Hessian
            # H_full = self.get_full_H_with_diagonal()
            # U, V = np.linalg.eigh(H_full)
            # diag_init = np.median(U)/np.sum(self.active)
            diag_init = np.min(self.min_eig_sub[self.min_eig_sub>0])
            # diag_init = np.min([
            #     np.min(self.min_eig_sub[self.min_eig_sub>0]),
            #     diag_approx    ])
            # print [
            #     np.min(self.min_eig_sub[self.min_eig_sub>0]),
            #     diag_approx    ]
            if self.display > 2:
                print(" no history "),
                print "setting diag_init %g"%diag_init,
            self.min_eig_sub[indx] = diag_init
            self.max_eig_sub[indx] = diag_init
            self.b[:,:,indx] = 0.
            return

        if not new_bfgs_term:
            return

        s = self.hist_deltatheta[:,[0],indx].astype(complex)
        y = self.hist_deltadf[:,[0],indx].astype(complex)
        # diag_approx = np.sqrt(np.sum(y**2)/np.sum(s**2))

        # if np.sum(self.hist_deltatheta[:,1,indx]**2) <= 0:
        #     # this is the first real gradient evaluation
        #     self.min_eig_sub[indx] = diag_approx

        # load up the current Hessian approximation
        b_p = self.b[:,:,indx]
        H = np.real(np.dot(b_p, b_p.T)) + np.eye(self.K_max)*self.min_eig_sub[indx]

        # update the Hessian with the most recent update
        # this is the BFGS step
        # for numerical stability
        rscl = np.sqrt(np.sum(s**2))
        s = s/rscl
        y = y/rscl
        Hs = np.dot(H, s)
        term1 = y / np.sqrt(np.sum(y*s))
        sHs = np.sum(s*Hs)
        term2 = np.sqrt(complex(-1.)) * Hs / np.sqrt(sHs)
        if np.sum(~np.isfinite(term1)) > 0 or np.sum(~np.isfinite(term2)) > 0:
            self.min_eig_sub[indx] = np.max(self.max_eig_sub) #np.median(H2w)
            if self.display > 1:
                print("invalid bfgs history term.  should never get here!")
            return
        H += np.dot(term1, term1.T)
        H += np.dot(term2, term2.T)

        # constrain it to be positive definite
        U, V = np.linalg.eigh(H)
        if np.max(U) <= 0.:
            # if there aren't any positive eigenvalues, then
            # set them all to be the same conservative diagonal value
            U[:] = self.max_eig_sub[indx]
            if self.display > 3:
                print("no positive eigenvalues after BFGS"),
        if not self.global_rect:
            # set any too-small eigenvalues to the median positive
            # eigenvalue
            U_median = np.median(U[U>0])
            U[(U<(np.max(np.abs(U))/self.hess_max_dev))] = U_median
        print "EIGS %g %g %g %g %g"%(np.min(U), np.median(U), np.max(U), self.min_eig_sub[indx], np.max(self.min_eig_sub)),
        # the Hessian after it's been forced to be positive definite
        H_posdef = np.dot(V*U, V.T)

        # update min_eig_sub
        # self.min_eig_sub[indx] = np.min(U)
        # self.min_eig_sub[indx] = np.median(U)
        # geometric mean
        # neval = float(self.eval_count[indx])
        # print
        # print diag_approx, self.min_eig_sub[indx], np.exp((
        #     np.log(diag_approx)*1./neval + np.log(self.min_eig_sub[indx])*(neval-1)/neval
        #     )), neval,
        # diag_approx = np.exp((
        #     np.log(diag_approx)*1./neval + np.log(self.min_eig_sub[indx])*(neval-1)/neval
        #     ))
        # self.min_eig_sub[indx] = diag_approx
        # self.min_eig_sub[indx] = 1. # the diagonal hessian approx should make this correct
        self.min_eig_sub[indx] = diag_approx/np.sum(self.active)
        self.max_eig_sub[indx] = np.max(U)

        # now break it apart into matrices b and a diagonal term again
        B_pos = H_posdef - np.eye(self.K_max)*self.min_eig_sub[indx]
        U, V = np.linalg.eigh(B_pos)
        b_p = V*np.sqrt(U.reshape((1,-1)).astype(complex))

        self.b[:,:,indx] = b_p

        return


    def theta_original_to_list(self, theta_original):
        """
        Convert from the original parameter format into a list of numpy arrays.
        The original format can be a numpy array, or a dictionary of numpy arrays,
        or a list of numpy arrays, or any nested variation thereof.
        """
        if isinstance(theta_original, np.ndarray):
            return [theta_original,]
        elif isinstance(theta_original, list) or isinstance(theta_original, tuple):
            theta_list = []
            for theta_element in theta_original:
                theta_list.extend(self.theta_original_to_list(theta_element))
            return theta_list
        elif isinstance(theta_original, dict):
            theta_list = []
            for key in sorted(theta_original.keys()):
                theta_list.extend(self.theta_original_to_list(theta_original[key]))
            return theta_list
        else:
            # if it doesn't match anything else, assume it's a scalar
            # TODO(jascha) error checking here
            return [asarray(theta_original).reshape((1,)),]
    def theta_list_to_original_recurse(self, theta_list, theta_original):
        """
        Recursively convert from a list of numpy arrays into the original parameter format.
        
        Use theta_list_to_original() instead of calling this function directly.
        """
        if isinstance(theta_original, list) or isinstance(theta_original, tuple):
            theta_new = []
            for theta_element in theta_original:
                if isinstance(theta_element, np.ndarray):
                    theta_new.append(theta_list[0])
                    theta_list = theta_list[1:]
                elif isinstance(theta_element, dict) or \
                        isinstance(theta_element, list) or \
                        isinstance(theta_element, tuple):
                    theta_item, theta_list = self.theta_list_to_original_recurse(theta_list, theta_element)
                    theta_new.append(theta_item)
                else:
                    # if it doesn't match anything else, assume it's a scalar
                    theta_new.append(theta_list[0][0])
                    theta_list = theta_list[1:]                    
            return theta_new, theta_list
        elif isinstance(theta_original, dict):
            theta_dict = dict()
            for key in sorted(theta_original.keys()):
                theta_element = theta_original[key]
                if isinstance(theta_element, np.ndarray):
                    theta_dict[key] = theta_list[0]
                    theta_list = theta_list[1:]
                elif isinstance(theta_element, dict) or \
                        isinstance(theta_element, list) or \
                        isinstance(theta_element, tuple):
                    theta_item, theta_list = self.theta_list_to_original_recurse(theta_list, theta_original[key])
                    theta_dict[key] = theta_item
                else:
                    # if it doesn't match anything else, assume it's a scalar
                    theta_dict[key] = theta_list[0][0]
                    theta_list = theta_list[1:]
            return theta_dict, theta_list
        else:
            raise Exception("invalid data format for theta")
    def theta_list_to_original(self, theta_list):
        """
        Convert from a list of numpy arrays into the original parameter format.
        """
        if isinstance(self.theta_original, np.ndarray):
            return theta_list[0]
        else:
            # we do this recursively to handle nested arrays or dictionaries of parameters
            theta_new, _ = self.theta_list_to_original_recurse(theta_list, self.theta_original)
            return theta_new

    def theta_list_to_flat(self, theta_list):
        """
        Convert from a list of numpy arrays into a 1d numpy array.
        """
        num_el = 0
        for el in theta_list:
            num_el += np.prod(el.shape)
        theta_flat = np.zeros((num_el, 1))
        start_indx = 0
        for el in theta_list:
            stop_indx = start_indx + np.prod(el.shape)
            theta_flat[start_indx:stop_indx, 0] = el.ravel()
            start_indx = stop_indx
        return theta_flat
    def theta_flat_to_list(self, theta_flat):
        """
        Convert from a 1d numpy arfray into a list of numpy arrays.
        """
        if len(theta_flat.shape) == 1:
            # make it Nx1 rather than N, for consistency
            theta_flat = theta_flat.reshape((-1,1))
        theta_list = []
        start_indx = 0
        for el in self.theta_list:
            stop_indx = start_indx + np.prod(el.shape)
            theta_list.append(theta_flat[start_indx:stop_indx,0].reshape(el.shape))
            start_indx = stop_indx
        return theta_list
    def theta_original_to_flat(self, theta_original):
        """
        Convert from the original parameter format into a 1d array.
        """
        return self.theta_list_to_flat(self.theta_original_to_list(theta_original))
    def theta_flat_to_original(self, theta_flat):
        """
        Convert from a 1d array into the original parameter format.
        """
        return self.theta_list_to_original(self.theta_flat_to_list(theta_flat))


    def update_natgrad(self):

        diag_rescale_full = 1./self.nat_grad_rescale

        theta_unnat = self.theta/self.nat_grad_rescale
        theta_prior_unnat = self.theta_prior_step/self.nat_grad_rescale

        if self.natgrad_count == 0:
            if self.display > 6:
                print "merging each parameter type"
            ddf_total_sqr_list = self.theta_flat_to_list(self.ddf_total_sqr)
            ddtheta_total_sqr_list = self.theta_flat_to_list(self.ddtheta_total_sqr)
            for dd in ddf_total_sqr_list:
                # dd[:] = np.mean(dd)
                # dd[:] = np.maximum(dd, np.mean(dd))
                dd[:] = np.max(dd)
            for dd in ddtheta_total_sqr_list:
                # dd[:] = np.mean(dd)
                # dd[:] = np.maximum(dd, np.mean(dd))
                dd[:] = np.max(dd)
            self.ddf_total_sqr = self.theta_list_to_flat(ddf_total_sqr_list)
            self.ddtheta_total_sqr = self.theta_list_to_flat(ddtheta_total_sqr_list)

        # # compute the natural gradient for each block of parameters
        # grad_variance_list = self.theta_flat_to_list(self.total_grad_variance / np.mean(self.total_grad_variance))
        # nat_grad_rescale_list = []
        # for ii in xrange(len(grad_variance_list)):
        #     scl = np.sqrt(np.mean(grad_variance_list[ii]))
        #     # scl = np.sqrt(np.max(grad_variance_list[ii]))
        #     if self.display > 3:
        #         print "nat scale %g, "%(scl),
        #     nat_grad_rescale_list += [scl*np.ones(grad_variance_list[ii].shape)]
        # self.nat_grad_rescale = self.theta_list_to_flat(nat_grad_rescale_list)
        # self.nat_grad_rescale = np.sqrt(self.total_grad_variance/self.eval_count_total)
        # self.nat_grad_rescale = np.sqrt(self.total_grad_variance/np.mean(self.total_grad_variance))
        old_diag_hess = self.nat_grad_rescale**2

        if self.geometric_average_natgrad:
            diag_hess = np.sqrt(np.exp(self.ddf_total_sqr - self.ddtheta_total_sqr))
        else:
            diag_hess = np.sqrt(self.ddf_total_sqr / self.ddtheta_total_sqr)
        # prevent changes in overall scale, only adapt relative scales
        if self.geometric_average_natgrad:
            diag_hess /= np.exp(np.mean(np.log(diag_hess))) 
        else:
            diag_hess /= np.mean(diag_hess)

        # # if self.natgrad_count == 0:
        # #     self.nat_grad_rescale = np.sqrt(diag_hess)
        # # else:
        # # decay time of one pass through data
        # # nact = float(np.sum(self.active))
        # nact = np.ceil(float(np.sum(self.active))/(self.K_max-self.K_min))
        # self.nat_grad_rescale = np.exp(
        #     np.log(self.nat_grad_rescale)*(nact-1.)/nact +
        #     np.log(np.sqrt(diag_hess))*1./nact
        #     )

        self.nat_grad_rescale = np.sqrt(diag_hess) # DEBUG


        self.theta = theta_unnat*self.nat_grad_rescale
        self.theta_prior_step = theta_prior_unnat*self.nat_grad_rescale

        diag_rescale_full *= self.nat_grad_rescale
        oldP_scaled = self.P.copy()*diag_rescale_full

        self.update_subspace(self.theta)

        PP_proj = np.dot( self.P.T, oldP_scaled )
        self.last_theta = np.dot(PP_proj, self.last_theta)
        if self.display > 5:
            print "approximate last_theta projection"

        self.theta_proj = np.dot(self.P.T, self.theta)

        # if self.natgrad_count == 0:
        #     # we probably just made an enormous change to parameter scaling
        #     # let's blow away the history...
        #     if self.display > 3:
        #         print "Establishing initial diagonal Hessian approximation. Clearing non-diagonal Hessian."
        #     self.b[:] = 0
        #     self.last_df[:] = 0
        #     self.last_theta = np.tile(self.theta_proj, ((1,self.N)))
        #     self.min_eig_sub[self.active] = 1. # assuming the diagonal approx is exactly right
        #     self.hist_deltatheta[:] = 0
        #     self.hist_deltadf[:] = 0
        #     self.full_H[:] = 0
        # # else:

        # scale up the hessian to counteract
        max_ratio = np.max(old_diag_hess/diag_hess)
        min_ratio = np.min(old_diag_hess/diag_hess)
        max_abs_ratio = np.max([max_ratio, 1./min_ratio])

        # shrink theta towards the current location
        self.last_theta = self.theta_proj + (self.last_theta - self.theta_proj)/max_abs_ratio
        if self.natgrad_count == 0:
            self.last_df /= max_ratio
            if self.display > 5:
                print "scaling ",

        # make sure the mean doesn't shift
        if self.display > 5:
            min_ratio = np.min(old_diag_hess/diag_hess)
            print "min, max, maxabs ratio %g, %g, %g, "%(min_ratio, max_ratio, max_abs_ratio)

        self.natgrad_count += 1

        # subspace_theta_shift = self.theta_proj - theta_proj_old

        # DEBUG
        # self.last_theta += subspace_theta_shift

        # self.last_theta[:,:] = np.dot(self.P.T, self.theta)
        # self.theta_proj = np.dot(self.P.T, self.theta)

        # self.eval_count[:] = 0
        # self.last_df[:] = 0.

        # # evaluate all the (active) gradients at the current location
        # for idx in np.flatnonzero(self.active):
        #     if self.display > 3:
        #         print "nat eval %d"%(idx)

        #     # evaluate subfunction value and gradient at new position
        #     f, df_proj = self.f_df_wrapper(self.theta, idx)
        #     # add the change in theta and the change in gradient to the history for this subfunction
        #     self.update_history(idx, self.theta_proj, f, df_proj)

        # self.hist_deltatheta[:] = 0.
        # self.hist_deltadf[:] = 0.

        # # self.b = np.zeros((self.K_max,2*self.max_history,self.N)).astype(complex)
        # # self.full_H = np.zeros((self.K_max,self.K_max))
        # # self.min_eig_sub = self.max_eig_sub

        # self.step_scale /= np.sum(self.active)


    def f_df_wrapper(self, theta_in, idx, return_full=False):
        """
        A wrapper around the subfunction objective f_df, that handles the transformation
        into and out of the flattened parameterization used internally by SFO.
        """

        theta_unnat = theta_in/self.nat_grad_rescale
        theta = self.theta_flat_to_original(theta_unnat)
        # evaluate
        time_func_start = time.time()
        f, df = self.f_df(theta, self.sub_ref[idx], *self.args, **self.kwargs)
        time_diff = time.time() - time_func_start
        self.time_func += time_diff # time spent in function evaluation
        df = self.theta_original_to_flat(df)

        self.total_grad_variance += df**2

        df = df / self.nat_grad_rescale

        if return_full:
            # only used by check_grad
            return f, df

        bd = np.flatnonzero(~np.isfinite(df))
        if bd.shape[0] > 0:
            print "setting non-finite gradient elements to parameters, and setting f to be nonfinite",
            df[bd] = theta_in[bd] # encourage decay towards zero parameters
            f = np.nan

        if self.subspace_momentum:
            self.update_subspace_momentum(theta_in, df)

        # update the subspace with the new gradient direction
        self.update_subspace(df)
        # gradient projected into the current subspace
        df_proj = np.dot( self.P.T, df )
        # keep a record of function evaluations
        self.hist_f_flat.append(f)
        self.eval_count[idx] += 1
        self.eval_count_total += 1

        return f, df_proj

    def update_subspace_momentum(self, theta, df):

        # multiscale time averages
        scl = np.exp(-np.linspace(0, np.log(np.sum(self.active)*2), self.subspace_momentum+1))
        scl = scl[1:].reshape((1,-1))
        # scl = 2**(-np.arange(1,self.subspace_momentum+1))
        # scl = scl.reshape((1,-1))

        # DEBUG scale gradients to unit length ?

        self.momentum_theta *= (1.-scl)
        self.momentum_df *= (1.-scl)
        self.momentum_theta += scl*theta
        self.momentum_df += scl*df

    def get_target_index(self):
        """ Choose which subfunction to update this iteration. """

        # if an active subfunction has one evaluation, get a second
        # so we can have a Hessian estimate
        gd = np.flatnonzero((self.eval_count == 1) & self.active)
        if len(gd) > 0:
            return gd[0]
        # If an active subfunction has less than two observations, then
        # evaluate it.  We want to get to two evaluations per subfunction
        # as quickly as possibly so that it's possible to estimate a Hessian
        # for it
        gd = np.flatnonzero((self.eval_count < 2) & self.active)
        if len(gd) > 0:
            return np.random.permutation(gd)[0]

        if self.subspace_momentum:
            # if an active subfunction has an odd number of evaluations, then make 
            # it even. we always want to do evaluations in pairs, so we get valid
            # Hessian information before the subspace resets
            gd = np.flatnonzero(np.mod(self.eval_count, 2).astype(bool) & self.active)
            if len(gd) > 0:
                return np.random.permutation(gd)[0]

        subfunction_selection = self.subfunction_selection
        if subfunction_selection == 'distance both':
            if np.random.randn() < 0:
                subfunction_selection = 'distance'
            else:
                subfunction_selection = 'single distance'
        elif subfunction_selection == 'distance all':
            subfunction_selection = np.random.choice([
                'distance',
                'single distance',
                'fewest evaluations',
                'largest gap',
                'most positive'
                ])

        if self.display > 7:
            print subfunction_selection,

        if subfunction_selection == 'distance':
            # the default case -- use the subfunction evaluated farthest
            # from the current location, weighted by the Hessian

            # difference between current theta and most recent evaluation
            # for all subfunctions
            dtheta = self.theta_proj - self.last_theta
            # full Hessian
            full_H_combined = self.get_full_H_with_diagonal()
            # squared distance
            distance = np.sum(dtheta*np.dot(full_H_combined, dtheta), axis=0)
            # sort the distances from largest to smallest
            dist_ord = np.argsort(-distance)
            # and keep only the indices that belong to active subfunctions
            dist_ord = dist_ord[self.active[dist_ord]]
            # and choose the active subfunction from farthest away
            indx = dist_ord[0]
            if np.max(distance[self.active]) < self.eps and np.sum(~self.active)>0 and self.eval_count[indx]>0:
                if self.display > 2:
                    print("all active subfunctions evaluated here.  expanding active set."),
                indx = np.random.permutation(np.flatnonzero(~self.active))[0]
                self.active[indx] = True
            return indx

        if subfunction_selection == 'fewest evaluations':
            indx = np.flatnonzero(self.eval_count == np.min(self.eval_count[self.eval_count>0]))[0]
            return indx

        if subfunction_selection == 'single distance':
            max_dist = -1
            indx = -1
            for i in range(self.N):
                dtheta = self.theta_proj - self.last_theta[:,[i]]
                bdtheta = np.dot(self.b[:,:,i].T, dtheta)
                dist = np.sum(bdtheta**2) + np.sum(dtheta**2)*self.min_eig_sub[i]
                if (dist > max_dist) and self.active[i]:
                    max_dist = dist
                    indx = i
            if indx == -1:
                # DEBUG
                1./0
            return indx

        if subfunction_selection == 'largest gap':
            # choose the subfunction with the largest predicted improvement between
            # its last measured function value and its function value at the
            # current location
            max_improvement = -np.inf
            indx = -1
            for i in np.flatnonzero(self.active):
                f_pred = self.get_predicted_subf(i, self.theta_proj)
                f_improvement = self.hist_f[i,0] - f_pred

                if (f_improvement > max_improvement):
                    max_improvement = f_improvement
                    indx = i
            if indx == -1:
                # DEBUG
                1./0
            return indx

        if subfunction_selection == 'most positive':
            maxf = np.nanmax(self.hist_f[:,0])
            indx = np.flatnonzero(self.hist_f[:,0] == maxf)[0]
            return indx

        if subfunction_selection == 'random':
            # choose an index to update at random
            indx = np.random.permutation(np.flatnonzero(self.active))[0]
            return indx

        if subfunction_selection == 'cyclic':
            # choose indices to update in a cyclic fashion
            active_list = np.flatnonzero(self.active)
            indx = active_list[self.cyclic_subfunction_index]
            self.cyclic_subfunction_index += 1
            self.cyclic_subfunction_index %= np.sum(self.active)
            return indx

        throw("unknown subfunction choice method")

    def handle_step_failure(self, f, df_proj, indx):
        """
        Check whether an update step failed.  Update current position if it did.
        """

        # check to see whether the step should be a failure
        step_failure = False
        if not np.isfinite(f) or np.sum(~np.isfinite(df_proj))>0:
            # step is a failure if function or gradient is non-finite
            step_failure = True
            if self.display > 2:
                print("non-finite function value or gradient"),
        elif self.eval_count[indx] == 1:
            # the step is a candidate for failure if it's a new subfunction, and it's
            # much larger than expected
            if np.max(self.eval_count) > 1:
                if f > np.mean(self.hist_f[self.eval_count>1,0]) + 3.*np.std(self.hist_f[self.eval_count>1,0]):
                    step_failure = True
        elif f > self.hist_f[indx,0]:
            # if this subfunction has increased in value, then look whether it's larger
            # than its predicted value by enough to trigger a failure
            # calculate the predicted value of this subfunction
            f_pred = self.get_predicted_subf(indx, self.theta_proj)
            # if the subfunction exceeds its predicted value by more than the predicted average gain
            # in the other subfunctions, then mark the step as a failure
            # (note that it's been about N steps since this has been evaluated, and that this subfunction can lay
            # claim to about 1/N fraction of the objective change)
            predicted_improvement_others = self.f_predicted_total_improvement - (self.hist_f[indx,0] - f_pred)

            if f - f_pred > predicted_improvement_others:
                step_failure = True

        if not step_failure:
            # decay the step_scale back towards 1
            NN = np.sqrt(np.sum(self.active))
            # self.step_scale = (1./NN) + self.step_scale * (1. - (1./NN))
            # DEBUG
            self.step_scale = (1./NN)*(1./NN) + self.step_scale * (1. - (1./NN))
        else:
            # shorten the step length
            self.step_scale /= 2.
            self.events['step_failure'] = True

            # the subspace may be updated during the function calls
            # so store this in the full space
            df = np.dot(self.P, df_proj)

            f_lastpos, df_lastpos_proj = self.f_df_wrapper(self.theta_prior_step, indx)
            df_lastpos = np.dot(self.P, df_lastpos_proj)

            ## if the function value exploded, then back it off until it's a
            ## reasonable order of magnitude before adding anything to the history
            f_pred = self.get_predicted_subf(indx, self.theta_proj)
            if np.isfinite(self.hist_f[indx,0]):
                predicted_f_diff = np.abs(f_pred - self.hist_f[indx,0])
            else:
                predicted_f_diff = np.abs(f - f_lastpos)
            if not np.isfinite(predicted_f_diff) or predicted_f_diff < self.eps:
                predicted_f_diff = self.eps

            for i_ls in range(10):
                if f - f_lastpos < 10.*predicted_f_diff:
                    # the failed update is already within an order of magnitude
                    # of the target update value -- no backoff required
                    break
                if self.display > 4:
                    print("ls {0} f_diff {1} predicted_f_diff {2} ".format(i_ls, f - f_lastpos, predicted_f_diff))
                # make the step length a factor of 100 shorter
                self.theta = 0.99*self.theta_prior_step + 0.01*self.theta
                self.theta_proj = np.dot(self.P.T, self.theta)
                # and recompute f and df at this new location
                f, df_proj = self.f_df_wrapper(self.theta, indx)
                df = np.dot(self.P, df_proj)

            # we're done with function calls.  can move these back into the subspace.
            df_proj = np.dot(self.P.T, df)
            df_lastpos_proj = np.dot(self.P.T, df_lastpos)

            if not np.isfinite(f):
                print "still non-finite f.",
                if self.subspace_momentum:
                    print " stepping back through history...",
                    for ii in xrange(self.momentum_theta.shape[1]):
                        print ii,
                        self.theta = self.momentum_theta[:,[ii]]
                        # and recompute f and df at this new location
                        f, df_proj = self.f_df_wrapper(self.theta, indx)
                        df = np.dot(self.P, df_proj)
                        if np.isfinite(f):
                            # new parameters may no longer be in subspace
                            self.update_subspace(self.theta)
                            self.theta_proj = np.dot(self.P.T, self.theta)
                            break
                else:
                    print "reverting to historical theta"
                    for ii in np.random.permutation(np.flatnonzero(self.eval_count>1)):
                        print ii,
                        self.theta = np.dot(self.P, self.last_theta[:,[ii]])
                        self.theta_proj = self.last_theta[:,[ii]]
                        # and recompute f and df at this new location
                        f, df_proj = self.f_df_wrapper(self.theta, indx)
                        df = np.dot(self.P, df_proj)
                        if np.isfinite(f):
                            break

            if f < f_lastpos:
                # the original objective was better -- but add the newly evaluated point to the history,
                # just so it's not a wasted function call
                theta_lastpos_proj = np.dot(self.P.T, self.theta_prior_step)
                self.update_history(indx, theta_lastpos_proj, f_lastpos, df_lastpos_proj)
                if self.display > 2:
                    print("step failed, but last position was even worse ( f {0}, std f {1}),".format(f_lastpos, np.std(self.hist_f[self.eval_count>0,0]))),
            else:
                # add the change in theta and the change in gradient to the history for this subfunction
                # before failing over to the last position
                if np.isfinite(f) and np.sum(~np.isfinite(df_proj))==0:
                    self.update_history(indx, self.theta_proj, f, df_proj)
                if self.display > 2:
                    print("step failed, proposed f {0}, std f {1},".format(f, np.std(self.hist_f[self.eval_count>0,0]))),
                if self.display > -1 and np.sum(self.eval_count>1) < 2:
                    print(  "\nStep failed on the very first subfunction.  This is\n"
                            "either due to an incorrect gradient, or a very large\n"
                            "Hessian.  Try:\n"
                            "   - Calling check_grad() (see README.md for details)\n"
                            "   - Initializing SFO with a larger initial Hessian, by\n"
                            "calling it with hessian_init=1e12 (or other large number)")
                f = f_lastpos
                df_proj = df_lastpos_proj
                self.theta = self.theta_prior_step
                self.theta_proj = np.dot(self.P.T, self.theta)

        # don't let steps get so short that they don't provide any usable Hessian information
        # TODO use a more principled cutoff here
        self.step_scale = np.max([self.step_scale, 1e-5])

        return step_failure, f, df_proj


    def expand_active_subfunctions(self, full_H_inv, step_failure):
        """
        expand the set of active subfunctions as appropriate
        """
        # power in the average gradient direction
        df_avg = np.mean(self.last_df[:,self.active], axis=1).reshape((-1,1))
        p_df_avg = np.sum(df_avg * np.dot(full_H_inv, df_avg))
        # power of the standard error
        ldfs = self.last_df[:,self.active] - df_avg
        num_active = np.sum(self.active)
        p_df_sum = np.sum(ldfs * np.dot(full_H_inv, ldfs)) / num_active / (num_active - 1)
        # if the standard errror in the estimated gradient is the same order of magnitude as the gradient,
        # we want to increase the size of the active set
        increase_desirable = p_df_sum >= p_df_avg*self.max_gradient_noise 
        # increase the active set on step failure
        increase_desirable = increase_desirable or step_failure
        # increase the active set if we've done a full pass without updating it
        increase_desirable = increase_desirable or self.iter_since_active_growth > num_active
        # make sure all the subfunctions have enough evaluations for a Hessian approximation
        # before bringing in new subfunctions
        eligibile_for_increase = np.min(self.eval_count[self.active]) > 2
        # one more iteration has passed since the active set was last expanded
        self.iter_since_active_growth += 1
        if increase_desirable and eligibile_for_increase:
            # the index of the new subfunction to activate
            new_gd = np.random.permutation(np.flatnonzero(~self.active))[:1]
            if len(new_gd) > 0:
                self.iter_since_active_growth = 0
                self.active[new_gd] = True


    def optimization_step(self):
        """
        Perform a single optimization step.  This function is typically called by SFO.optimize().
        """
        time_pass_start = time.time()

        # 1./0

        ## choose an index to update
        indx = self.get_target_index()

        if self.display > 2:
            print("||dtheta|| {0},".format(np.sqrt(np.sum((self.theta - self.theta_prior_step)**2)))),

            # print("||dtheta sub|| {0},".format(np.sqrt(np.sum((np.dot(self.P.T, self.theta) - np.dot(self.P.T, self.theta_prior_step))**2)))),
            # print("||dtheta oldsub|| {0},".format(np.sqrt(np.sum((self.theta_proj - np.dot(self.P.T, self.theta_prior_step))**2)))),
            # print("||dtheta oldold|| {0},".format(np.sqrt(np.sum((self.theta_proj - self.last_theta[:,[indx]])**2)))),
            # try:
            #     print("||dtheta oldraw|| {0},".format(np.sqrt(np.sum((self.theta_proj - self.theta_proj_raw)**2)))),
            #     print("||dtheta raw|| {0},".format(np.sqrt(np.sum((self.theta - self.theta_proj_raw_full)**2)))),
            # except:
            #     pass

            print("index {0}, last f {1},".format(indx, self.hist_f[indx,0])),
            print("step scale {0},".format(self.step_scale)),

        # events are for debugging -- eg, so that the user supplied objective can check
        # for step failure itself
        self.events['step_failure'] = False

        # evaluate subfunction value and gradient at new position
        f, df_proj = self.f_df_wrapper(self.theta, indx)

        # check for a failed update step, and adjust f, df, and self.theta
        # as appropriate if one occurs.
        step_failure, f, df_proj = self.handle_step_failure(f, df_proj, indx)

        if not self.lead_ahead is None:
            if 'nesterov' in self.lead_ahead:
                # DEBUG -- should do this in the full space. this is wasteful.
                self.df_accum += np.dot(self.P, df_proj)*self.step_scale

        # print("||proj error|| {0},".format(np.sqrt(np.sum((self.theta_proj - np.dot(self.P.T, self.theta))**2)))),
        # print("||proj2 error|| {0},".format(np.sqrt(np.sum((self.theta - np.dot(self.P, np.dot(self.P.T, self.theta)))**2)))),
        # self.theta_proj_raw = self.theta_proj.copy() # DEBUG deleteme
        # self.theta_proj_raw_full = self.theta.copy() # DEBUG deleteme

        # add the change in theta and the change in gradient to the history for this subfunction
        self.update_history(indx, self.theta_proj, f, df_proj)

        # increment the total distance traveled using the last update
        self.total_distance += np.sqrt(np.sum((self.theta - self.theta_prior_step)**2))

        # # the current contribution from this subfunction to the total Hessian approximation
        # H_pre_update = np.real(np.dot(self.b[:,:,indx], self.b[:,:,indx].T))
        # ## update this subfunction's Hessian estimate
        # self.update_hessian(indx)
        # # the new contribution from this subfunction to the total approximate hessian
        # H_new = np.real(np.dot(self.b[:,:,indx], self.b[:,:,indx].T))   
        # # update total Hessian using this subfunction's updated contribution
        # self.full_H += H_new - H_pre_update

        if self.reduced_curvature:
            curvature_scale = np.sum(self.active)/2.
        else:
            curvature_scale = 1.

        # calculate the total gradient, total Hessian, and total function value at the current location
        full_df = 0.
        for i in range(self.N):
            dtheta = self.theta_proj - self.last_theta[:,[i]]
            bdtheta = np.dot(self.b[:,:,i].T, dtheta)
            Hdtheta = np.real(np.dot(self.b[:,:,i], bdtheta))/curvature_scale
            Hdtheta += dtheta*self.min_eig_sub[i]/curvature_scale # the diagonal contribution
            full_df += Hdtheta + self.last_df[:,[i]]

        full_H_combined = self.get_full_H_with_diagonal()/curvature_scale
        # TODO - Use Woodbury identity instead of recalculating full inverse
        full_H_inv = np.linalg.inv(full_H_combined)

        # backup the prior position, in case this is a failed step
        # try:
        #     self.theta_prior_step = self.theta_no_lead_ahead.copy()
        # except:
        #     # TODO just initialize sensibly
        #     self.theta_prior_step = self.theta.copy()
        self.theta_prior_step = self.theta.copy()

        # calculate an update step
        dtheta_proj = -np.dot(full_H_inv, full_df)

        # # the update to theta, in the full dimensional space
        # dtheta_no_lead_ahead = np.dot(self.P, dtheta_proj)*self.step_scale
        # # update theta to the new location
        # self.theta_no_lead_ahead = self.theta + dtheta_no_lead_ahead


        # if we're doing aggressive / predictive location choosing
        if self.lead_ahead == 'nesterov accumulate only':
            df_accum_proj = np.dot(self.P.T, self.df_accum)
            # # df_accum_proj /= 10. # DEBUG maybe / np.sum(self.active) ??
            # dv = np.max([np.sum(self.active), 10.])
            # df_accum_proj /= dv
            pp = -np.dot(full_H_inv, df_accum_proj)
            print("||nest|| %g, "%np.sqrt(np.sum(pp**2))),
            dtheta_proj = pp
        elif self.lead_ahead == 'nesterov accumulate':
            df_accum_proj = np.dot(self.P.T, self.df_accum)
            # df_accum_proj /= 10. # DEBUG maybe / np.sum(self.active) ??
            dv = np.max([np.sum(self.active), 10.])
            df_accum_proj /= dv
            pp = -np.dot(full_H_inv, df_accum_proj)
            print("||nest|| %g, "%np.sqrt(np.sum(pp**2))),
            dtheta_proj += pp
        elif self.lead_ahead == 'nesterov':
            # choose our next location as if we will continue along current
            # average trajectory
            mean_last_theta = np.mean(self.last_theta[:,(self.eval_count>0)], axis=1).reshape((-1,1))
            mean_diff = self.theta_proj + dtheta_proj - mean_last_theta
            mean_diff /= np.sum(self.active)/2. # per step, not per pass, otherwise will get quite long
            dtheta_proj += mean_diff
        elif self.lead_ahead == 'sag':
            # choose our next location *ahead* of our current one, in the direction we seem to be traveling
            df_sum = np.sum(self.last_df, axis=1).reshape((-1,1))
            dtheta_proj += -np.dot(full_H_inv, df_sum)
        elif self.lead_ahead == 'sag power law':
            # choose our next location *ahead* of our current one, in the direction we seem to be traveling
            df_mean = np.mean(self.momentum_df, axis=1).reshape((-1,1))
            df_mean_proj = np.dot(self.P.T, df_mean)
            dtheta_proj += -np.dot(full_H_inv, df_mean_proj) * np.sum(self.active)
        elif self.lead_ahead == None:
            pass
        else:
            1./0

        dtheta_proj *= self.step_scale

        dtheta_proj_length = np.sqrt(np.sum(dtheta_proj**2))
        if dtheta_proj_length < self.minimum_step_length:
            dtheta_proj *= self.minimum_step_length/dtheta_proj_length
            dtheta_proj_length = self.minimum_step_length
            if self.display > 3:
                print("forcing minimum step length"),
        elif dtheta_proj_length > self.maximum_step_length:
            dtheta_proj *= self.maximum_step_length/dtheta_proj_length
            dtheta_proj_length = self.maximum_step_length
            if self.display > 3:
                print("forcing maximum step length"),
        elif self.eval_count_total > 2.*np.sum(self.active) and dtheta_proj_length > self.eps:
            # only allow a step to be up to a factor of max_step_length_ratio longer than the
            # average step length
            avg_length = self.total_distance / float(self.eval_count_total)

            # DEBUG
            if self.eval_count_total < 20: #3.*np.sum(self.active):
                avg_length = 0.2

            length_ratio = dtheta_proj_length / avg_length
            ratio_scale = self.max_step_length_ratio
            if length_ratio > ratio_scale:
                if self.display > 3:
                    print("truncating step length from %g to %g"%(dtheta_proj_length, ratio_scale*avg_length)),
                ## but increment avg_length by the true increase
                dist_inc = dtheta_proj_length
                dtheta_proj_length /= length_ratio/ratio_scale
                dtheta_proj /= length_ratio/ratio_scale
                # and kill the accumulator by same factor
                self.df_accum /= length_ratio/ratio_scale
                ## but don't increment avg_length twice
                dist_inc -= dtheta_proj_length
                ## and don't go TOO cray
                dist_inc = np.min([dist_inc, 0.1*self.total_distance])
                self.total_distance += dist_inc

        # the update to theta, in the full dimensional space
        dtheta = np.dot(self.P, dtheta_proj)

        # update theta to the new location
        self.theta += dtheta

        if self.update_sdev > 0:
            # corrupt theta as appropriate
            noise_vec = np.random.normal(size=(self.M, 1))/np.sqrt(self.M)
            self.update_subspace(noise_vec)
            self.theta += noise_vec*self.update_sdev
            self.theta_proj = np.dot(self.P.T, self.theta)
        else:
            self.theta_proj += dtheta_proj

        # the predicted improvement from this update step
        self.f_predicted_total_improvement = 0.5 * np.dot(dtheta_proj.T, np.dot(full_H_combined, dtheta_proj))

        ## expand the set of active subfunctions as appropriate
        self.expand_active_subfunctions(full_H_inv, step_failure)

        # record how much time was taken by this learning step
        time_diff = time.time() - time_pass_start
        self.time_pass += time_diff

        # print "theta dev %g"%(np.sqrt(np.sum((
        #     self.theta - np.dot(self.P, np.dot(self.P.T, self.theta))
        #     )**2))),



