"""
This code generates the optimizer comparison convergence figures in the SFO paper.

Adjust the arrays "models_to_train" and "optimizers_to_use" in the function
"generate_data()" in order to choose which objective functions and optimizers
are tested.

It can take a long time (up to weeks) for all optimizers to run with all
hyperparameters and all objectives.  I recommend starting with just the 
logistic regression objective (the default configuration).

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


def generate_data(num_passes=51, base_fname='figure_data_', store_x=True):
    """
    train all the models in models_to_train using all the
    optimizers, and save the resulting function value traces
    in figure_data_*.npz files.
    """
    models_to_train = ( 
                        #models.DeepAE,
                        models.CIFARConvNet,
                        models.ICA,
                        models.toy,
                        models.logistic,
                        models.MLP_soft,
                        # models.MLP_hard,
                        models.ContractiveAutoencoder,
                        models.Hopfield,
                        )
    models_to_train = ( models.logistic, ) # DEBUG
    
    for model_class in models_to_train:
        np.random.seed(0) # make experiments repeatable
        model = model_class()
        trainer = optimization_wrapper.train(model)
        optimizers_to_use = (
                                trainer.SFO,
                                trainer.LBFGS,
                                trainer.LBFGS_minibatch,
                                trainer.ADA,
                                trainer.SGD,
                                trainer.SAG,
                                trainer.SGD_momentum,
                            )
        for optimizer in optimizers_to_use:
            np.random.seed(0) # make experiments exactly repeatable
            print("\n\n\n" + model.name + "\n" + str(optimizer))
            optimizer(num_passes=num_passes)
            # save_results doesn't need to be called until outside this loop,
            # but this way we can peak at partial results
            # also, note that saved files for the same model but different optimizers
            # can be combined in plots, just by calling load_results with all the saved files
            convergence_utils.save_results(trainer, base_fname=base_fname, store_x=store_x)


def train_and_plot(num_passes=51, base_fname='convergence'):
    """
    Train all the models.  Save the function value and parameter settings from
    convergence.  Reload the resutls, plot them, and save the plots.

    Note that plots can be generated from previously saved histories, using load_results.
    To change the models to train and the optimizers to use
    """
    base_fname += "_%s_"%(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    generate_data(num_passes=num_passes, base_fname=base_fname)
    history_nested, num_subfunctions, full_objective_period = convergence_utils.load_results(base_fname=base_fname)
    convergence_utils.make_plots(history_nested, num_subfunctions, full_objective_period)


if __name__ == '__main__':
    # compare convergence for different optimizers
    train_and_plot()
    print "plots saved for optimizer comparison"
