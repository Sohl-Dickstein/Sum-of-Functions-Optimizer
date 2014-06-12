"""
This code generates the convergence comparison for SFO with different numbers
of subfunctions N.

Adjust the arrays "models_to_train" and "N_set" in the function
"generate_data_SFO_N()" in order to choose which objective functions and
numbers of subfunctions are tested.

Author: Jascha Sohl-Dickstein (2014)
Web: http://redwood.berkeley.edu/jascha
This software is made available under the Creative Commons
Attribution-Noncommercial License.
( http://creativecommons.org/licenses/by-nc/3.0/ )
"""

import convergence_utils

import models
import optimization_wrapper

import datetime
import numpy as np

def generate_data_SFO_N(num_passes=51, base_fname='num_minibatches', store_x=True):
    """
    Same as generate_data(), but compares SFO with different numbers of minibatches
    rather than SFO to other optimizers.
    """
    models_to_train = ( models.logistic, models.Hopfield )
    models_to_train = ( models.logistic, ) # DEBUG

    # the different numbers of minibatches to experiment with
    N_set = np.round(np.logspace(0, np.log10(200), 6)).astype(int)
    #N_set = np.round(np.logspace(0, 2, 3)).astype(int)

    for model_class in models_to_train:
        # # first do LBFGS
        # np.random.seed(0) # make experiments repeatable
        # model = model_class(scale_by_N=False)
        # trainer = optimization_wrapper.train(model, full_objective_per_pass=1)
        # optimizer = trainer.LBFGS
        # print("\n\n\n" + model.name + "\n" + str(optimizer))
        # optimizer(num_passes=num_passes)
        # save_results(trainer, base_fname=base_fname, store_x=store_x)

        # then do SFO with different minibatch sizes
        for N in N_set:
            np.random.seed(0) # make experiments repeatable
            model = model_class(num_subfunctions=N, scale_by_N=False)
            trainer = optimization_wrapper.train(model, full_objective_per_pass=1)
            optimizer = trainer.SFO
            np.random.seed(0) # make experiments exactly repeatable
            print("\n\n\n" + model.name + "\n" + str(optimizer))
            optimizer(num_passes=num_passes, learner_name='SFO $N=%d$'%(N))
            convergence_utils.save_results(trainer, base_fname=(base_fname+'_N=%d'%(N)), store_x=store_x)


def train_and_plot_SFO_N(num_passes=51, base_fname='num_minibatches'):
    """
    Same as train_and_plot(), but compares SFO with different numbers of minibatches
    rather than SFO to other optimizers.
    """
    base_fname += "_%s_"%(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    generate_data_SFO_N(num_passes=num_passes, base_fname=base_fname)
    history_nested, num_subfunctions, full_objective_period = convergence_utils.load_results(base_fname=base_fname)
    convergence_utils.make_plots(history_nested, num_subfunctions, full_objective_period, name_prefix='num_minibatches_')


if __name__ == '__main__':
    # compare convergence for different numbers of subfunctions
    train_and_plot_SFO_N()
    print "plots saved for number of minibatches comparison"
