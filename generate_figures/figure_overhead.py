"""
Generate and save PDFs for the figures showing computational overhead for SFO.

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

import matplotlib
matplotlib.use('Agg')  # no displayed figures -- need to call before loading pylab
import matplotlib.pyplot as plt

from sfo import SFO
import numpy as np

import models
import warnings

def explore_MN(burnin_steps=2, test_steps=2):

    M_arr = []
    N_arr = []
    N = 100
    #N = 50
    for M in np.linspace(1, 1e6, 5):
    #for M in np.linspace(1, 1e3, 4):
        M_arr.append(int(M))
        N_arr.append(int(N))
    M = 1e6
    #M = 1e3
    for N in np.linspace(1,200,5):
    #for N in np.linspace(1,50,4):
        M_arr.append(int(M))
        N_arr.append(int(N))

    T_arr = []

    for ii in range(len(M_arr)):
        M = M_arr[ii]
        N = N_arr[ii]

        print "case %d of %d, M=%g, N=%g"%(ii+1, len(M_arr), M, N)

        # make the model
        model = models.toy(num_subfunctions=N, num_dims=M)
        # initialize the optimizer
        optimizer = SFO(model.f_df, model.theta_init, model.subfunction_references, display=1)
        # burn in the optimizer, to make sure the subspace has eg. reached its full size
        optimizer.optimize(num_passes=burnin_steps)

        # time spent in optimizer during burning
        t0 = optimizer.time_pass - optimizer.time_func
        steps0 = np.sum(optimizer.eval_count)
        optimizer.optimize(num_passes=test_steps)
        t1 = optimizer.time_pass - optimizer.time_func
        t_diff = t1 - t0
        steps1 = np.sum(optimizer.eval_count)
        actual_test_steps = float(steps1 - steps0)/float(N)
        T_arr.append(t_diff/actual_test_steps)
        print T_arr[-1]
        
    return np.array(M_arr), np.array(N_arr), np.array(T_arr)

def convert_num(tt):
    if tt > 1e5:
        return "$10^{%d}$"%int(np.log10(tt))
    return "$%d$"%tt

def plot_shared(M, M_arr, N_arr, T_arr, v_fixed, v_change):
    figsize=(2.5,2.,)

    idx = np.flatnonzero(M_arr == M)
    ord = np.argsort(N_arr[idx])
    idx = idx[ord]
    if idx.shape[0] > 1:
        plt.figure(figsize=figsize)
        plt.plot(N_arr[idx], T_arr[idx], 'x--')
        plt.grid()
        ax = plt.axis()
        plt.axis([0, ax[1], 0, ax[3]])
        nn = np.linspace(0, np.max(T_arr[idx]), 5)
        nnt = ["$%d$"%tt for tt in nn]
        nnt[1] = ''
        nnt[3] = ''
        plt.yticks(nn, nnt)
        nn = np.linspace(0, np.max(N_arr[idx]), 5)
        nnt = [convert_num(tt) for tt in nn]
        nnt[1] = ''
        nnt[2] = ''
        nnt[3] = ''
        plt.xticks(nn, nnt)
        plt.axes().set_axisbelow(True)
        plt.xlabel('$%s$'%v_change)
        plt.ylabel('Overhead (s)')
        if M > 10**3:
            plt.title('Fixed $%s=10^{%d}$'%(v_fixed, int(np.log10(M))))
        else:
            plt.title('Fixed $%s=%d$'%(v_fixed, M))
        try:
            plt.tight_layout()
        except:
            warnings.warn('tight_layout failed.  try running with an Agg backend.')
        plt.savefig('figure_overhead_fixed%s.pdf'%(v_fixed))


def make_plots(M_arr, N_arr, T_arr):
    for M in np.unique(M_arr):
        plot_shared(M, M_arr, N_arr, T_arr, 'M', 'N')
    for N in np.unique(N_arr):
        plot_shared(N, N_arr, M_arr, T_arr, 'N', 'M')


if __name__ == '__main__':
    M_arr, N_arr, T_arr = explore_MN()
    make_plots(M_arr, N_arr, T_arr)

"""
import figure_overhead
reload(figure_overhead)

M_arr, N_arr, T_arr = figure_overhead.explore_MN()
figure_overhead.make_plots(M_arr, N_arr, T_arr)



"""
