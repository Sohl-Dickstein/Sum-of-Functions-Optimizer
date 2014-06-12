"""
This code generates the convergence comparison for SFO with different design
choices.

Adjust the arrays "models_to_train" and "N_set" in the function
"generate_data_SFO_variations()" in order to choose which objective functions
are tested.

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


def generate_data_SFO_variations(num_passes=51, base_fname='convergence_variations', store_x=True):
    """
    Same as generate_data(), but compares different variations of SFO to each
    other, rather than SFO to other optimizers.
    """
    models_to_train = ( models.logistic, models.Hopfield )
    models_to_train = ( models.logistic, ) #DEBUG
    
    for model_class in models_to_train:
        np.random.seed(0) # make experiments repeatable
        model = model_class()
        trainer = optimization_wrapper.train(model)
        optimizers_to_use = [trainer.SFO_variations,]
        for optimizer in optimizers_to_use:
            np.random.seed(0) # make experiments exactly repeatable
            print("\n\n\n" + model.name + "\n" + str(optimizer))
            optimizer(num_passes=num_passes)
            # save_results doesn't need to be called until outside this loop,
            # but this way we can peak at partial results
            # also, note that saved files for the same model but different optimizers
            # can be combined in plots, just by calling load_results with all the saved files
            convergence_utils.save_results(trainer, base_fname=base_fname, store_x=store_x)


def train_and_plot_SFO_variations(num_passes=51, base_fname='convergence_variations'):
    """
    Same as train_and_plot(), but compares different variations of SFO to each
    other, rather than SFO to other optimizers.
    """
    base_fname += "_%s_"%(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    generate_data_SFO_variations(num_passes=num_passes, base_fname=base_fname)
    history_nested, num_subfunctions, full_objective_period = convergence_utils.load_results(base_fname=base_fname)
    convergence_utils.make_plots(history_nested, num_subfunctions, full_objective_period, name_prefix='variations_')


if __name__ == '__main__':
    # compare convergence for different variations on SFO
    train_and_plot_SFO_variations()
    print "plots saved for variations on SFO algorithm comparison"
