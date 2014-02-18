## generate figures for paper



from figures_convergence import *
fnames = []
fnames += glob.glob('npz_for_figures/*.npz')
#fnames += glob.glob('npz_for_figures/*Hopfield*.npz')
#fnames += ['npz_for_figures/sparse_Hopfield_SGD_20140125-171643_Hopfield.npz',]
history_nested, num_subfunctions, full_objective_period = load_results(fnames=fnames)
make_plots(history_nested, num_subfunctions, full_objective_period)






# below here no longer needed






from figures import *
fnames = []
fnames += glob.glob('npz_for_figures/*Hopf*.npz')
history_nested, num_subfunctions, full_objective_period = load_results(fnames=fnames)
make_plots(history_nested, num_subfunctions, full_objective_period)








from figures import *

fnames = []
fnames += glob.glob('everything_small2/*.npz')
fnames += glob.glob('everything_small/*.npz')
fnames += glob.glob('sfo_small_nodoublefailure/*npz')
fnames += glob.glob('momentum/*npz')
fnames += glob.glob('convnet/*npz')
fnames += glob.glob('SFO_newdefaults/*npz')
history_nested = load_results(fnames=fnames)
make_plots(history_nested)



## generate figures for paper
from figures import *
fnames = []
fnames += glob.glob('final_data/*npz')
#fnames += glob.glob('final_data_ica/*npz')
#fnames += glob.glob('final_data_ica/50passes_ICA_20131107-162207_ICA.npz')
#fnames += glob.glob('final_data_ica/50passes_ICA_standard_constraint_min_eig_set_20131108-185545_ICA.npz')
#fnames += glob.glob('final_data_ica/*ICA_nowhite_20131107-221701_ICA*npz')
#fnames += glob.glob('final_data_rerun/*.npz')
#fnames += glob.glob('final_data_ica/50passes_ICA_20131107-162207_ICA.npz')
#fnames += glob.glob('final_data_rerun2/*.npz')
#fnames += glob.glob('data_final_rerun2/*.npz')
fnames += glob.glob('final_data_ica/50passes_ICA_20131107-162207_ICA.npz')
#fnames += glob.glob('final_data_rerun3/*.npz')
#fnames += glob.glob('final_data_hopfield/*.npz')
fnames += glob.glob('final_data_mlp_large/*.npz')
fnames += glob.glob('final_data_rerun4/*.npz')
history_nested = load_results(fnames=fnames)
make_plots(history_nested)




## generate figures for paper
from figures import *
fnames = []
fnames += glob.glob('final_data/*convn*npz')
#fnames += glob.glob('final_data_ica/*npz')
#fnames += glob.glob('final_data_ica/50passes_ICA_20131107-162207_ICA.npz')
#fnames += glob.glob('final_data_ica/50passes_ICA_standard_constraint_min_eig_set_20131108-185545_ICA.npz')
#fnames += glob.glob('final_data_ica/*ICA_nowhite_20131107-221701_ICA*npz')
#fnames += glob.glob('final_data_rerun/*.npz')
#fnames += glob.glob('final_data_ica/50passes_ICA_20131107-162207_ICA.npz')
#fnames += glob.glob('final_data_rerun2/*.npz')
#fnames += glob.glob('data_final_rerun2/*.npz')
#fnames += glob.glob('final_data_ica/50passes_ICA_20131107-162207_ICA.npz')
#fnames += glob.glob('final_data_rerun3/*.npz')
#fnames += glob.glob('final_data_hopfield/*.npz')
fnames += glob.glob('final_data_mlp_large/*convn*.npz')
fnames += glob.glob('final_data_rerun4/*convn*.npz')
history_nested = load_results(fnames=fnames)
make_plots(history_nested)



rsync --archive -v -r godot:Dropbox/sum_function_optimizer/GLM ./
from figures_GLM import *
fnames = []
fnames += glob.glob('GLM/GLM_single_neuron_20131117-003513_GLM.npz') #  (1) a single neuron with correlated filters (sinusoids)
history_nested = load_results(fnames=fnames)
make_plots(history_nested, display_trajectory=True)


from figures import *
fnames = []
fnames += glob.glob('GLM/*.npz')
fnames += glob.glob('GLM_SFO_rerun_20131116-120252_GLM.npz')
history_nested = load_results(fnames=fnames)
make_plots(history_nested, display_trajectory=True)


from figures_GLM import *
fnames = []
#fnames += glob.glob('/Users/jascha/Dropbox/sfo_npz/*.npz')
fnames += glob.glob('/Users/jascha/Dropbox/sfo_npz/*nrocinu*.npz')
history_nested = load_results(fnames=fnames)
make_plots(history_nested, display_trajectory=True)


from figures_GLM import *
fnames = []
fnames += glob.glob('GLM_v6/*.npz')
history_nested = load_results(fnames=fnames)
make_plots(history_nested, display_trajectory=True)



from figures import *
fnames = []
fnames += glob.glob('final_data/*Hop*npz')
history_nested = load_results(fnames=fnames)
make_plots(history_nested, display_trajectory=True)


from figures import *
fnames = []
#fnames += glob.glob('final_data/*ICA*npz')
fnames += glob.glob('final_data_ica/50passes_ICA_20131107-162207_ICA.npz')
#fnames += glob.glob('final_data_ica/50passes_ICA_standard_constraint_min_eig_set_20131108-185545_ICA.npz')
#fnames += glob.glob('final_data_ica/50passes_ICA_SFO_only_20131107-214234_ICA.npz')
#fnames += glob.glob('final_data_rerun/*ICA*.npz')
history_nested = load_results(fnames=fnames)
make_plots(history_nested, display_trajectory=True)


## really scrap
from figures import *
fnames = []
fnames += glob.glob('SFO_newdefaults/*npz')
history_nested = load_results(fnames=fnames)
make_plots(history_nested)



from figures import *
fnames = []
fnames += glob.glob('convnet/*npz')
history_nested = load_results(fnames=fnames)
make_plots(history_nested)

from figures import *
fnames = []
fnames += glob.glob('SFO_variations/*npz')
history_nested = load_results(fnames=fnames)
make_plots(history_nested)




# test time
from sfo import SFO
def f_df(theta, ref):
	f = np.sum((theta+ref)**2)
	df = 2*(theta+ref)
	return f, df
refs = range(100)
theta = np.random.randn(1e6,1)
opt = SFO(f_df, theta, refs)
x = opt.optimize(num_passes=1)



