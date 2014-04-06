"""
Author: Jascha Sohl-Dickstein (2014)
This software is made available under the Creative Commons
Attribution-Noncommercial License.
( http://creativecommons.org/licenses/by-nc/3.0/ )
"""

#from __future__ import print_function
from numpy import *
from random import shuffle
from collections import defaultdict
import scipy.linalg
import time
import warnings

class SFO(object):
    def __init__(self, f_df, theta, subfunction_references, args=(), kwargs={},
        display=2, max_history_terms=10, max_gradient_noise=1.,
        hessian_init=1e6, init_subf=2, hess_max_dev = 1e8,
        hessian_algorithm='bfgs', subfunction_selection='distance'):
        """
        The main Sum of Functions Optimizer (SFO) class.

        Parameters:
        f_df - Returns the function value and gradient for a single subfunction 
            call.  Should have the form
                [f, dfdtheta] = f_df(theta, subfunction_references[idx],
                                      *args, **kwargs)
            where idx is the index of a single subfunction.
        theta - The initial parameters to be used for optimization.  theta can
            be either a NumPy array, an array of NumPy arrays, or a dictionary
            of NumPy arrays.  The gradient returned by f_df should have the
            same form as theta.
        subfunction_references - A list containing an identifying element for
            each subfunction.  The elements in this list could be, eg, numpy
            matrices containing minibatches, or indices identifying the
            subfunction, or filenames from which target data should be read.
            If each subfunction corresponds to a minibatch, then the number
            of subfunctions should be approximately
            [number subfunctions] = sqrt([dataset size])/10.

        Optional Parameters, roughly in order of decreasing utility:
        args=() - This list will be passed through as *args to f_df.
        kwargs={} - This dictionary will be passed through as **kwargs to f_df.
        display=2 - Display level.  1 = once per SFO call, 2 = once per
            subfunction call, larger = debugging info
        max_history_terms=10 - The number of history terms to use in the
            BFGS algorithm.
        max_gradient_noise=1. - The maximum ratio of standard error in the
            gradient across subfunctions to the length of the average gradient
            across subfunctions before the number of active subfunctions is
            increased.
        hessian_init=1e6 - The initial estimate of the Hessian for the first
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
        subspace_dimensionality = 2*self.N+2  # 2 to include current location
        # subspace can't be larger than the full space
        subspace_dimensionality = int(min([subspace_dimensionality, self.M]))
        # the uptdate steps will be rescaled by this
        self.step_scale = 1.
        # "very small" for various tasks, most importantly identifying when
        # update steps are too small to be used for Hessian updates without
        # incurring large numerical errors.
        self.eps = 1e-12

        # the min and max dimenstionality for the subspace
        self.K_min = subspace_dimensionality
        self.K_max = ceil(self.K_min*1.5)
        self.K_max = int(min([self.K_max, self.M]))
        # self.P holds the subspace
        self.P = zeros((self.M,self.K_max))

        # store the minimum and maximum eigenvalue from each approximate
        # Hessian
        self.min_eig_sub = zeros((self.N))
        self.max_eig_sub = zeros((self.N))

        # store the total time spent in optimization, and the amount of time
        # spent in the objective function
        self.time_pass = 0.
        self.time_func = 0.

        # how many steps since the active set size was increased
        self.iter_since_active_growth = 0

        # which subfunctions are active
        self.active = zeros((self.N)).astype(bool)
        inds = random.permutation(self.N)[:init_subf]
        self.active[inds] = True
        self.min_eig_sub[inds] = hessian_init
        self.max_eig_sub[inds] = hessian_init

        # the total path length traveled during optimization
        self.total_distance = 0.
        # number of function evaluations for each subfunction
        self.eval_count = zeros((self.N))

        # the current dimensionality of the subspace
        self.K_current = 1
        # set the first column of the subspace to be the initial
        # theta
        rr = sqrt(sum(self.theta**2))
        if rr > 0:
            self.P[:,[0]] = self.theta/rr
        else:
            # initial theta is 0 -- initialize randomly
            self.P[:,[0]] = random.randn(self.theta.shape[0],1)
            self.P[:,[0]] /= sqrt(sum(self.P[:,[0]]**2))

        if self.M == self.K_max:
            # if the subspace spans the full space, then just make
            # P the identity matrix
            if self.display > 1:
                print("subspace spans full space"),
            self.P = eye(self.M)
            self.K_current = self.M+1

        # store last theta (in the subspace)
        theta_proj = dot( self.P.T, self.theta )
        # holds the last position and the last gradient for all the objective functions
        self.last_theta = tile(theta_proj, ((1,self.N)))
        self.last_df = zeros((self.K_max,self.N))
        # the history of theta changes for each subfunction
        self.hist_deltatheta = zeros((self.K_max,max_history_terms,self.N))
        # the history of gradient changes for each subfunction
        self.hist_deltadf = zeros((self.K_max,max_history_terms,self.N))
        # the history of function values for each subfunction
        self.hist_f = ones((self.N, max_history_terms))*nan
        # a flat history of all returned subfunction values
        self.hist_f_flat = []

        # the approximate Hessian for each subfunction is stored
        # as dot(self.b[:.:.index], self.b[:.:.inedx].T)
        self.b = zeros((self.K_max,2*self.max_history,self.N)).astype(complex)
        # TODO self.b could be real if another diagonal term carrying sign
        # information was introduced

        # the full Hessian (sum over all the subfunctions)
        self.full_H = zeros((self.K_max,self.K_max))

        # holds diagnostic information (eg, step failure, when the subspace is collapsed)
        self.events = defaultdict(lambda: False)

        # used to keep track of the current subfunction if the update order is
        # cyclic (subfunction_selection='cyclic')
        self.cyclic_subfunction_index = 0


    def optimize(self, num_passes = 10, num_steps = None):
        """
        Optimize the objective function.  num_steps is the number of subfunction calls to make,
        and num_passes is the number of effective passes through all subfunctions to make.
        """
        if num_steps==None:
            num_steps = int(num_passes*self.N)
        for i in range(num_steps):
            if self.display > 1:
                print("pass {}, step {},".format(float(sum(self.eval_count))/self.N, i)),
            self.optimization_step()
            if self.display > 1:
                print("f {}, <f> {}, active {}/{}, sfo {} s, func {} s".format(self.hist_f_flat[-1], mean(self.hist_f[self.eval_count>0,0]), sum(self.active), self.active.shape[0], self.time_pass - self.time_func, self.time_func))
        if num_steps < 1:
            print "No optimization steps performed.  Change num_passes or num_steps."
        elif self.display > 0:
            print("optimize active {}/{}, f {}, <f> {}, pass #{}, sfo {} s, func {} s".format(sum(self.active), self.active.shape[0], self.hist_f_flat[-1], mean(self.hist_f[self.eval_count>0,0]), float(sum(self.eval_count))/self.N, self.time_pass - self.time_func, self.time_func))

        if self.time_pass - self.time_func > self.time_func:
            print "Optimization is spending too much time in SFO (%g s) relative to evaluating the objective function (%g s)!"%(self.time_pass - self.time_func, self.time_func)
            print "Try reducing the number of subfunctions or minibatches."

        # reverse the flattening transformation on theta
        return self.theta_flat_to_original(self.theta)


    def check_grad(self, small_diff = None):
        """
        A diagnostic function to check the gradients for the subfunctions.  It
        checks the subfunctions in random order, and the dimensions of each
        subfunction in random order.  This way, a representitive set of
        gradients can be checked quickly, even for high dimensional objectives.
        """

        if small_diff is None:
            # step size to use for gradient check
            small_diff = self.eps*1e6
        print "Testing step size %g"%small_diff

        for i in random.permutation(range(self.N)):
            fl, dfl = self.f_df_wrapper(self.theta, i)
            ep = zeros((self.M,1))
            dfl_obs = zeros((self.M,1))
            dfl_err = zeros((self.M,1))
            for j in random.permutation(range(self.M)):
                ep[j] = small_diff
                fl2, _ = self.f_df_wrapper(self.theta + ep, i)
                dfl_obs[j] = (fl2 - fl)/small_diff
                dfl_err[j] = dfl_obs[j] - dfl[j]
                if abs(dfl_err[j]) > small_diff * 1e4:
                    print("large diff "),
                else:
                    print("           "),
                print("  gradient subfunction {}, dimension {}, analytic {}, finite diff {}, error {}".format(i, j, dfl[j], dfl_obs[j], dfl_err[j]))
                ep[j] = 0.
            gerr = sqrt(sum((dfl - dfl_obs)**2))
            print("subfunction {}, total L2 gradient error {}".format(i, gerr))
            print

    def apply_subspace_transformation(self,T_left,T_right):
        """
        Apply change-of-subspace transformation.  This function is called when
        the subspace is collapsed to project into the new lower dimensional
        subspace.
        T_left - The covariant subspace to subspace projection matrix.
        T_right - The contravariant subspace projection matrix.
        """

        ss = T_left.shape[1]
        tt = T_left.shape[0]

        # project history terms into new subspace
        self.last_df = dot(T_right.T, self.last_df)
        self.last_theta = dot(T_left, self.last_theta)
        self.hist_deltadf = dot(T_right.T, self.hist_deltadf.reshape((ss,-1))).reshape((tt,-1,self.N))
        self.hist_deltatheta = dot(T_left, self.hist_deltatheta.reshape((ss,-1))).reshape((tt,-1,self.N))
        self.b = dot(T_right.T, self.b.reshape((ss,-1))).reshape((tt,2*self.max_history,self.N))
        self.full_H = dot(T_right.T, self.full_H)
        self.full_H = dot(T_right.T, self.full_H.T).T

    def collapse_subspace(self, xl=None):
        """
        Collapse the subspace to its smallest dimensionality.
        """
        if self.display > 2:
            print()
            print("collapsing subspace"),

        # the projection matrix from old to new subspace
        Pl = zeros((self.K_max,self.K_max))

        # yy will hold all the directions to pack
        # into the subspace.  initialize it with random noise,
        # so that it still spans K_min dimensions even if not all the subfunctions
        # are active yet
        # TODO -- reduce the dimensionality below K_min before all the subfunctions
        # are active?
        yy = random.randn(self.K_max,self.K_min)
        if xl == None:
            xl = random.randn(self.K_max,1)
        # the most recent position and gradient for all active subfunctions,
        # as well as the current position and gradient (which will not be saved in the history yet)
        yz = hstack((self.last_df[:,self.active], self.last_theta[:,self.active], xl, dot(self.P.T,self.theta)))
        yy[:,:yz.shape[1]] = yz
        Pl[:,:self.K_min] = linalg.qr(yy)[0]

        # TODO -- we know the end of this is zeros
        # only need to project up to K_min dimensions
        # and could zero out the remaining columns.
        # (small win?)
        self.P = dot(self.P, Pl)

        # projection matrix from old to new basis
        # (because Pl.T is orthonormal, co- and contra-variant transformations are identical)
        self.apply_subspace_transformation(Pl.T, Pl)

        # update the recorded subspace size
        self.K_current = self.K_min


    def update_subspace(self, x_in):
        """
        Update the low dimensional subspace by adding a new direction.
        x_in - The new vector to incorporate into the subspace.
        """
        if self.K_current >= self.M:
            # no need to update the subspace if it spans the full space
            return
        x_in_length = sqrt(sum(x_in**2))
        if x_in_length < self.eps:
            # if the new vector is too short, nothing to do
            return
        # make x unit length
        xnew = x_in/x_in_length

        # find the component of x pointing out of the existing subspace
        #DEBUG for i in range(4):
        for i in range(2): # do this multiple times for numerical stability
            ## NOTE this next line is the most time consuming line in the whole code
            xnew -= dot(self.P[:,:self.K_current], dot(self.P[:,:self.K_current].T, xnew))
            #xnew -= dot(self.P, dot(self.P.T, xnew))
            ss = sqrt(sum(xnew**2))
            if ss < self.eps:
                # it barely points out of the existing subspace
                return
            xnew /= ss

        # add a new column to the subspace containing the new direction
        self.P[:,self.K_current] = xnew[:,0]
        self.K_current += 1

        self.events['collapse subspace'] = False
        if self.K_current >= self.K_max:
            # the subspace has exceeded its maximum allowed size -- collapse it
            self.events['collapse subspace'] = True
            # xl may not be in the history yet, so we pass it in explicitly to make
            # sure it's used
            xl = dot(self.P.T, x_in)
            self.collapse_subspace(xl=xl)

        # set the historical coordinates along this new dimension
        self.last_theta += dot(self.P.T, xnew) * dot(xnew.T, self.theta)[0,0]

    def get_full_H_with_diagonal(self):
        """
        Get the full approximate Hessian, including the diagonal terms.
        (note that self.full_H is stored without the diagonal terms)
        """
        full_H_combined = self.full_H + eye(self.K_max)*sum(self.min_eig_sub[self.active])
        return full_H_combined


    def get_predicted_subf(self, indx, theta_proj):
        """
        Get the predicted value of subfunction idx at theta_proj
        (where theat_proj is in the subspace)
        """
        dtheta = theta_proj - self.last_theta[:,[indx]]
        bdtheta = dot(self.b[:,:,indx].T, dtheta)
        Hdtheta = real(dot(self.b[:,:,indx], bdtheta))
        Hdtheta += dtheta*self.min_eig_sub[indx] # the diagonal contribution
        df_pred = self.last_df[:,[indx]] + Hdtheta
        f_pred = self.hist_f[indx,0] + dot(self.last_df[:,[indx]].T, dtheta)[0,0] + 0.5*dot(dtheta.T, Hdtheta)[0,0]
        return f_pred


    def update_history(self, indx, theta_proj, df_proj):
        """
        Update history of position differences and gradient differences
        for subfunction indx.
        """
        # there needs to be at least one earlier measurement from this
        # subfunction to compute position and gradient differences.
        if self.eval_count[indx] > 0:
            # differences in gradient and position
            ddf = df_proj - self.last_df[:,[indx]]
            ddt = theta_proj - self.last_theta[:,[indx]]
            # length of gradient and position change vectors
            lddt = sqrt(sum(ddt**2))
            lddf = sqrt(sum(ddf**2))

            if self.display > 3 and dot(ddf.T, ddt) < 0:
                print("Warning!  Negative dgradient dtheta inner product.  Adding it anyway."),            
            if lddt < self.eps:
                print("Largest change in theta too small ({}).  Not adding.".format(lddt)),
                return
            if lddf < self.eps:
                print("Largest change in gradient too small {}.  Not adding.".format(lddf)),
                return
            if self.display > 3:
                print("subf ||dtheta|| {}, subf ||ddf|| {}, corr(ddf,dtheta) {},".format(lddt, lddf, sum(ddt*ddf)/(lddt*lddf))),

            # shift the history by one timestep
            self.hist_deltatheta[:,1:,indx] = self.hist_deltatheta[:,:-1,indx]
            # store the difference in theta since the subfunction was last evaluated
            self.hist_deltatheta[:,[0],indx] = ddt
            # do the same thing for the change in gradient
            self.hist_deltadf[:,1:,indx] = self.hist_deltadf[:,:-1,indx]
            self.hist_deltadf[:,[0],indx] = ddf


    def update_hessian(self,indx):
        """
        Update the Hessian approximation for a single subfunction.
        indx - The index of the target subfunction for Hessian update.
        """

        gd = flatnonzero(sum(self.hist_deltatheta[:,:,indx]**2, axis=0)>0)
        if len(gd) == 0:
            # if no history, initialize with the median eigenvalue
            if self.display > 2:
                print(" no history "),
            self.b[:,:,indx] = 0.
            H = self.get_full_H_with_diagonal()
            U, V = linalg.eigh(H)
            self.min_eig_sub[indx] = median(U)/sum(self.active)
            self.max_eig_sub[indx] = self.min_eig_sub[indx]
            if self.eval_count[indx] > 2:
                if self.display > 1:
                    print("Subfunction evaluated %d times, but has no stored history.  This should never happen.")
                # # DEBUG
                # self.min_eig_sub[indx] *= 2.
            return

        # work in subspace of history for this
        gd = sum(self.hist_deltatheta[:,:,indx]**2,axis=0)>0
        P_hist = linalg.qr(hstack((self.hist_deltatheta[:,gd,indx],self.hist_deltadf[:,gd,indx])))[0]
        deltatheta_P = dot(P_hist.T, self.hist_deltatheta[:,gd,indx])
        deltadf_P = dot(P_hist.T, self.hist_deltadf[:,gd,indx])

        # get an approximation to the smallest eigenvalue.
        # This will be used on the diagonal for initialization.
        try:
            # calculate Hessian using pinv and squared equation.  just to get
            # smallest eigenvalue.
            # df = H dx
            # df^T df = dx^T H^T H dx = dx^T H^2 dx
            pdelthet = linalg.pinv(deltatheta_P)
            dd = dot(deltadf_P, pdelthet)
            H2 = dot(dd.T, dd)
            #H2 = dot( pdelthet.T, dot( dot( self.hist_deltadf[:,gd,indx].T, self.hist_deltadf[:,gd,indx] ), pdelthet))
            H2w, H2v = linalg.eigh(H2)
            H2w = sqrt(abs(H2w))
        except:
            H2w = array([0,1])
        if min(H2w) == 0:
            # there was a failure using this history.  use the largest of
            # the initializations from other functions
            H2w += max(self.min_eig_sub[self.active])

        num_gd = sum(gd)
        if num_gd > H2w.shape[0]:
            num_gd = H2w.shape[0]
        try:
            self.min_eig_sub[indx] = min(H2w[H2w >= sort(H2w)[-num_gd]])
            self.max_eig_sub[indx] = max(H2w)
        except:
            self.min_eig_sub[indx] = max(self.max_eig_sub[self.active])
            self.max_eig_sub[indx] = self.min_eig_sub[indx]
        if self.min_eig_sub[indx] < self.max_eig_sub[indx]/self.hess_max_dev:
            # constrain using allowed ratio
            self.min_eig_sub[indx] = self.max_eig_sub[indx]/self.hess_max_dev
            if self.display > 3:
                print("constraining Hessian initialization"),

        # recalculate Hessian
        num_hist = deltatheta_P.shape[1]
        b_p = zeros((P_hist.shape[1], num_hist*2)).astype(complex)
        for hist_i in reversed(range(num_hist)):
            s = deltatheta_P[:,[hist_i]].astype(complex)
            y = deltadf_P[:,[hist_i]].astype(complex)

            # for numerical stability
            rscl = sqrt(sum(s**2))
            s = s.copy()/rscl
            y = y.copy()/rscl

            Hs = s*self.min_eig_sub[indx] + dot(b_p, dot(b_p.T, s))

            if self.hessian_algorithm == 'bfgs':
                term1 = y / sqrt(sum(y*s))
                sHs = sum(s*Hs)
                term2 = sqrt(complex(-1.)) * Hs / sqrt(sHs)
                assert(sum(abs(array(Hs.shape) - array(s.shape)))==0)
                if sum(~isfinite(term1)) > 0 or sum(~isfinite(term2)) > 0:
                    self.min_eig_sub[indx] = max(H2w)
                    continue
                b_p[:,[2*hist_i+1]] = term1
                b_p[:,[2*hist_i]] = term2
            elif self.hessian_algorithm == 'rank1':
                diff = y - Hs
                b_p[:,[hist_i]] = diff/sqrt(sum(s*diff))
            else:
                raise(Exception("invalid Hessian update algorithm"))

        H = real(dot(b_p, b_p.T)) + eye(b_p.shape[0])*self.min_eig_sub[indx]
        try:
            # make sure it's positive definite
            U, V = linalg.eigh(H)
            U_median = median(U[U>0])
            U[(U<(max(abs(U))/self.hess_max_dev))] = U_median
        except:
            V = eye(H.shape[0])
            U = self.max_eig_sub[indx]*ones((H.shape[0],))
            if self.display > 3:
                print("bad eigenvalues in hessian"),
        B_pos = dot(V*U, V.T) - eye(b_p.shape[0])*self.min_eig_sub[indx]
        U, V = linalg.eigh(B_pos)
        nonzero_indx = ~(U==0)
        U = U[nonzero_indx]
        V = V[:,nonzero_indx]
        b_p = V*sqrt(U.reshape((1,-1)).astype(complex))

        self.b[:,:,indx] = 0.
        self.b[:,:b_p.shape[1],indx] = dot(P_hist, b_p)

        return


    def theta_original_to_list(self, theta_original):
        """
        Convert from the original parameter format into a list of numpy arrays.
        The original format can be a numpy array, or a dictionary of numpy arrays,
        or a list of numpy arrays.
        """
        if isinstance(theta_original, ndarray):
            return [theta_original,]
        elif isinstance(theta_original, list) or isinstance(theta_original, tuple):
            return theta_original
        elif isinstance(theta_original, dict):
            theta_list = []
            for key in sorted(theta_original.keys()):
                theta_list.append(theta_original[key])
            return theta_list
        else:
            raise "invalid data format for theta"
    def theta_list_to_original(self, theta_list):
        """
        Convert from a list of numpy arrays into the original parameter format.
        """
        if isinstance(self.theta_original, ndarray):
            return theta_list[0]
        elif isinstance(self.theta_original, list) or isinstance(self.theta_original, tuple):
            return theta_list
        elif isinstance(self.theta_original, dict):
            theta_dict = dict()
            list_indx = 0
            for key in sorted(self.theta_original.keys()):
                theta_dict[key] = theta_list[list_indx]
                list_indx += 1
            return theta_dict
        else:
            raise "invalid data format for theta"
    def theta_list_to_flat(self, theta_list):
        """
        Convert from a list of numpy arrays into a 1d numpy array.
        """
        num_el = 0
        for el in theta_list:
            num_el += prod(el.shape)
        theta_flat = zeros((num_el, 1))
        start_indx = 0
        for el in theta_list:
            stop_indx = start_indx + prod(el.shape)
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
            stop_indx = start_indx + prod(el.shape)
            theta_list.append(theta_flat[start_indx:stop_indx,0].reshape(el.shape))
            start_indx = stop_indx
        return theta_list
    def theta_original_to_flat(self, theta_original):
        """
        Convert from the original parameter format into a 1d array.
        """
        return self.theta_list_to_flat(self.theta_original_to_list(theta_original))
    def theta_flat_to_original(self, theta_original):
        """
        Convert from a 1d array into the original parameter format.
        """
        return self.theta_list_to_original(self.theta_flat_to_list(theta_original))

    def f_df_wrapper(self, theta_in, idx):
        """
        A wrapper around the subfunction objective f_df, that handles the transformation
        into and out of the flattened parameterization used internally by SFO.
        """

        theta = self.theta_flat_to_original(theta_in)
        # evaluate
        time_func_start = time.time()
        f, df = self.f_df(theta, self.sub_ref[idx], *self.args, **self.kwargs)
        time_diff = time.time() - time_func_start
        self.time_func += time_diff # time spent in function evaluation
        df = self.theta_original_to_flat(df)

        # keep a record of function evaluations
        self.hist_f_flat.append(f)
        return f, df

    def optimization_step(self):
        """
        Perform a single optimization step.  This function is typically called by SFO.optimize().
        """
        time_pass_start = time.time()

        ## choose an index to update
        # if an active subfunction has less than two observations, then
        # evaluate it.do a second observation at that subfunction,
        # so that it's possible to estimate a Hessian for it
        gd = flatnonzero((self.eval_count < 2) & self.active)
        if len(gd) > 0:
            indx = random.permutation(gd)[0]
        elif self.subfunction_selection == 'distance':
            # the default case -- use the subfunction evaluated farthest
            # from the current location, weighted by the Hessian

            # theta projected into current working subspace
            theta_proj = dot(self.P.T, self.theta)
            # difference between current theta and most recent evaluation
            # for all subfunctions
            dtheta = theta_proj - self.last_theta
            # full Hessian
            full_H_combined = self.get_full_H_with_diagonal()
            # squared distance
            distance = sum(dtheta*dot(full_H_combined, dtheta), axis=0)
            # sort the distances from largest to smallest
            dist_ord = argsort(-distance)
            # and keep only the indices that belong to active subfunctions
            dist_ord = dist_ord[self.active[dist_ord]]
            # and choose the active subfunction from farthest away
            indx = dist_ord[0]
            if max(distance[self.active]) < self.eps and sum(~self.active)>0 and self.eval_count[indx]>0:
                if self.display > 2:
                    print("all active subfunctions evaluated here.  expanding active set."),
                indx = random.permutation(flatnonzero(~self.active))[0]
                self.active[indx] = True
        elif self.subfunction_selection == 'random':
            # choose an index to update at random
            indx = random.permutation(flatnonzero(self.active))[0]
        elif self.subfunction_selection == 'cyclic':
            # choose indices to update in a cyclic fashion
            active_list = flatnonzero(self.active)
            indx = active_list[self.cyclic_subfunction_index]
            self.cyclic_subfunction_index += 1
            self.cyclic_subfunction_index %= sum(self.active)
        else:
            throw("unknown subfunction choice method")


        if self.display > 2:
            print("||dtheta|| {},".format(sqrt(sum((self.theta - self.theta_prior_step)**2)))),

        df_failed = None
        if self.display > 2:
            print("index {}, last f {},".format(indx, self.hist_f[indx,0])),
        self.events['step_failure'] = False
        f, df = self.f_df_wrapper(self.theta, indx)

        if self.display > 2:
            print("step scale {},".format(self.step_scale)),

        # check to see whether the step should be a failure
        step_failure = False
        if not isfinite(f) or sum(~isfinite(df))>0:
            # step is a failure if function or gradient is non-finite
            step_failure = True
        elif self.eval_count[indx] == 0:
            # the step is a candidate for failure if it's a new subfunction, and it's
            # much larger than expected
            if max(self.eval_count) > 0 and f > mean(self.hist_f[self.eval_count>0,0]) + 3.*std(self.hist_f[self.eval_count>0,0]):
                step_failure = True
        elif f > self.hist_f[indx,0]:
            # if this subfunction has increased in value, then look whether it's larger
            # than its predicted value by enough to trigger a failure
            # calculate the predicted value of this subfunction
            theta_proj = dot( self.P.T, self.theta )
            f_pred = self.get_predicted_subf(indx, theta_proj)
            # if the subfunction exceeds its predicted value by more than the predicted average gain
            # then mark the step as a failure
            # (note that it's been about N steps since this has been evaluated, and that this subfunction can lay
            # claim to about 1/N fraction of the objective change)
            if f - f_pred > self.f_predicted_total_improvement:
                step_failure = True

        if step_failure:
            self.events['step_failure'] = True
            theta_lastpos = self.theta_prior_step
            f_lastpos, df_lastpos = self.f_df_wrapper(theta_lastpos, indx)
            if f_lastpos < f or not isfinite(f) or sum(~isfinite(df))>0:
                # if the function was smaller at the prior theta position, then this step was a failure
                if self.display > 1:
                    print("step failed, proposed f {}, std f {},".format(f, std(self.hist_f[self.eval_count>0,0]))),
                # shorten the step length
                self.step_scale /= 2.

                # we will add the rejected update step to the history matrices
                # for this subfunction as well.
                # if the function value exploded, then shrink the update step to
                # a reasonable order of magnitude before doing so
                theta_proj = dot( self.P.T, self.theta )
                f_pred = self.get_predicted_subf(indx, theta_proj)
                predicted_f_diff = abs(f_pred - self.hist_f[indx,0])
                if not isfinite(predicted_f_diff) or predicted_f_diff < self.eps:
                    predicted_f_diff = self.eps
                for i_ls in range(10):
                    if self.display > 4:
                        print("ls {} f_diff {} predicted_f_diff {} ".format(i_ls, f - f_lastpos, predicted_f_diff))
                    if f - f_lastpos < 10.*predicted_f_diff:
                        # the failed update is already with an order of magnitude
                        # of the target update value -- no backoff required
                        break
                    # make the step length a factor of 100 shorter
                    self.theta = 0.99*theta_lastpos + 0.01*self.theta
                    # and recompute f and df at this new location
                    f, df = self.f_df_wrapper(self.theta, indx)

                # replace self.theta with its value at the last position
                theta_failed = self.theta
                df_failed = df
                f_failed = f
                self.theta = theta_lastpos
                f = f_lastpos
                df = df_lastpos
            else:
                if self.display > 2:
                    print("step candidate for failure but kept, last position f {}, std f {},".format(f_lastpos, std(self.hist_f[self.eval_count>0,0]))),
                step_failure = False
                # we're still going to store the extra observation in the history
                # so put it in the appropriate update variables
                theta_failed = theta_lastpos
                df_failed = df_lastpos
                f_failed = f_lastpos

                # decay the step_scale back towards 1
                self.step_scale = 1./self.N + self.step_scale * (1. - 1./self.N)
        else:
            # decay the step_scale back towards 1
            self.step_scale = 1./self.N + self.step_scale * (1. - 1./self.N)

        # increment the total distance traveled using the last update
        self.total_distance += sqrt(sum((self.theta - self.theta_prior_step)**2))

        # update the subspace with the new gradient direction
        self.update_subspace(df)
        if not df_failed is None:
            # also incorporate the gradient direction from a failed
            # update into the subspace
            self.update_subspace(df_failed)

        # theta and gradient projected into the current subspace
        theta_proj = dot( self.P.T, self.theta )
        df_proj = dot( self.P.T, df )

        # the contribution from this subfunction to the total Hessian approximation
        H_pre_update = real(dot(self.b[:,:,indx], self.b[:,:,indx].T))

        # add the change in theta and the change in gradeint to the history for this subfunction
        self.update_history(indx, theta_proj, df_proj)
        ## store information about the current position
        self.last_theta[:,[indx]] = theta_proj
        self.last_df[:,[indx]] = df_proj
        self.hist_f[indx,1:] = self.hist_f[indx,:-1]
        self.hist_f[indx,0] = f
        self.eval_count[indx] += 1
        if not df_failed is None:
            # step was a candidate for failure, and we did an
            # extra gradient evaluation.  use it to improve
            # BFGS
            # theta_failed and gradient projected into the current subspace
            theta_failed_proj = dot( self.P.T, theta_failed )
            df_failed_proj = dot( self.P.T, df_failed )
            self.update_history(indx, theta_failed_proj, df_failed_proj)

        ## update this subfunction's Hessian estimate
        self.update_hessian(indx)
        # update total Hessian using this subfunction's new contribution
        H_new = real(dot(self.b[:,:,indx], self.b[:,:,indx].T))   
        self.full_H += H_new - H_pre_update

        # calculate the total gradient, total Hessian, and total function value at the current location
        full_df = 0.
        for i in range(self.N):
            dtheta = theta_proj - self.last_theta[:,[i]]
            bdtheta = dot(self.b[:,:,i].T, dtheta)
            Hdtheta = real(dot(self.b[:,:,i], bdtheta))
            Hdtheta += dtheta*self.min_eig_sub[i] # the diagonal contribution
            full_df += Hdtheta + self.last_df[:,[i]]
        full_H_combined = self.get_full_H_with_diagonal()
        # TODO - Use Woodbury identity instead of recalculating full inverse
        full_H_inv = linalg.inv(full_H_combined)

        # calculate an update step
        dtheta_proj = -dot(full_H_inv, full_df) * self.step_scale

        #DEBUG
        # dtheta_proj_length = sqrt(sum(dtheta_proj**2))
        # if sum(self.eval_count) > self.N and dtheta_proj_length > 1e-20:
        #     # only allow a step to be up to a factor of 2 longer than the
        #     # average step length
        #     avg_length = self.total_distance / float(sum(self.eval_count))
        #     length_ratio = dtheta_proj_length / avg_length
        #     ratio_scale = 10.
        #     if length_ratio > ratio_scale:
        #         if self.display > 3:
        #             print "truncating step length from %g to %g"%(dtheta_proj_length, ratio_scale*avg_length),
        #         dtheta_proj_length /= length_ratio/ratio_scale
        #         dtheta_proj /= length_ratio/ratio_scale

        # the update to theta, in the full dimensional space
        dtheta = dot(self.P, dtheta_proj)

        # backup the prior position, in case this is a failed step
        self.theta_prior_step = self.theta.copy()
        # update theta to the new location
        self.theta += dtheta
        # the predicted improvement from this update step
        self.f_predicted_total_improvement = 0.5 * dot(dtheta_proj.T, dot(full_H_combined, dtheta_proj))

        ## expand the set of active subfunctions as appropriate
        # power in the average gradient direction
        df_avg = mean(self.last_df[:,self.active], axis=1).reshape((-1,1))
        p_df_avg = sum(df_avg * dot(full_H_inv, df_avg))
        # power of the standard error
        ldfs = self.last_df[:,self.active] - df_avg
        num_active = sum(self.active)
        p_df_sum = sum(ldfs * dot(full_H_inv, ldfs)) / num_active / (num_active - 1)
        # if the standard errror in the estimated gradient is the same order of magnitude as the gradient,
        # we want to increase the size of the active set
        increase_desirable = p_df_sum >= p_df_avg*self.max_gradient_noise 
        # increase the active set on step failure
        increase_desirable = increase_desirable or step_failure
        # increase the active set if we've done a full pass without updating it
        increase_desirable = increase_desirable or self.iter_since_active_growth > num_active
        # make sure all the subfunctions have enough evaluations for a Hessian approximation
        # before bringing in new subfunctions
        eligibile_for_increase = min(self.eval_count[self.active]) >= 2
        # one more iteration has passed since the active set was last expanded
        self.iter_since_active_growth += 1
        if increase_desirable and eligibile_for_increase:
            # the index of the new subfunction to activate
            new_gd = random.permutation(flatnonzero(~self.active))[:1]
            if len(new_gd) > 0:
                self.iter_since_active_growth = 0
                self.active[new_gd] = True

        # record how much time was taken by this learning step
        time_diff = time.time() - time_pass_start
        self.time_pass += time_diff
