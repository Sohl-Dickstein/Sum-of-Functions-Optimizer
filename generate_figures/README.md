Reproducible Science
================================

The code in this directory reproduces all of the figures from the paper
> Jascha Sohl-Dickstein, Ben Poole, and Surya Ganguli<br>
> An adaptive low dimensional quasi-Newton sum of functions optimizer<br>
> International Conference on Machine Learning (2014)<br>
> arXiv preprint arXiv:1311.2115 (2013)<br>
> http://arxiv.org/abs/1311.2115

-  **figure\_compare_optimizers.py** produces the convergence comparison of different optimizers in Figure 3.
-  **figure\_compare_sfo_N.py** produces the convergence comparison for SFO with different numbers of subfunctions or minibatches in Figure 2(c).
-  **figure\_compare_sfo_variations.py** produces the convergence comparison for SFO with different design choices in supplemental Figure C.1.
-  **figure\_overhead.py** produces the computational overhead analysis in Figure 2(a,b).
-  **figure\_cartoon.py** produces the cartoon illustration of the SFO algorithm in Figure 1.

To include a new objective function in the convergence comparison figure:

1.  Add a class to **models.py** which provides the objective function and gradient, and initialization, for the new objective.  The *toy* class is a good template to modify.
2.  Add the new objective class to *models_to_train* in **figure_compare_optimizers.py**.
3.  (optional) Modify plot characteristics in *make_plot_single_model* in **convergence_utils.py**.
4.  Run **figure_compare_optimizers.py**.

To include a new optimizer in the convergence comparison figure:

1.  Add a function implementing the optimizer to **optimization_wrapper.py**.  The *SGD* function is a good template to modify.
2.  Add the model to *optimizers_to_use* in **figure_compare_optimizers.py**.
3.  (optional) Modify plot characteristics in *make_plot_single_model* in **convergence_utils.py**.  (for instance, to select only the best performing hyperparameters, add the new optimizer to the list of *best_traces* function calls)
4.  Run **figure_compare_optimizers.py**.

For more documentation on SFO in general, see the README.md in the parent directory.

Several of the figures rely on a subdirectory **figure_data/** with training data.  This can be downloaded from https://www.dropbox.com/sh/h9z4djlgl2tagmu/GlVAJyErf8 .
