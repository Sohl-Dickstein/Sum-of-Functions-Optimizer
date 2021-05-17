"""
This code generates the convergence comparison for SFO with different design
choices.

Adjust the arrays "models_to_train" and "N_set" in the function
"generate_data_SFO_variations()" in order to choose which objective functions
are tested.

Copyright 2014 Jascha Sohl-Dickstein
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
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
