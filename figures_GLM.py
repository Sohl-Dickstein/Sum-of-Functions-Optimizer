"""
This code generates the GLM convergence figures.

Author: Niru Maheswaranathan, Jascha Sohl-Dickstein (2014)
This software is made available under the Creative Commons
Attribution-Noncommercial License.
( http://creativecommons.org/licenses/by-nc/3.0/ )
"""


import figures_models
import figures_train

import datetime
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')  # no displayed figures -- need to call before loading pylab
import matplotlib.pyplot as plt
import re
import warnings

from collections import defaultdict
from itertools import cycle


def sorted_nicely(strings): 
    "Sort strings the way humans are said to expect."
    return sorted(strings, key=natural_sort_key)

def natural_sort_key(key):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', key)]

def best_traces(prop, hist_f, styledict, color, neighbor_dist = 1):
    """
    Prunes the history to only the best trace in prop, and its
    neighbors within neighbor_dist in the sorted order.  It sets
    only the best trace to a dark line, and makes them all color.
    """
    if len(prop)>0:
        minnm = prop[0]
        prop = sorted_nicely(prop)
        # get the best one
        minf = np.inf
        for nm in prop:
            ff = np.asarray(hist_f[nm])
            minf2 = ff[-1] #np.min(ff)
            print(nm, minf2)
            if minf2 < minf:
                minnm = nm
                minf = minf2
        ii = prop.index(minnm)
        for i in range(len(prop)):
            if np.abs(i-ii) > neighbor_dist:
                del hist_f[prop[i]]
        for jj in range(1,neighbor_dist+1):
            try:
                styledict[prop[ii-jj]] = {'color':color, 'ls':':'}
            except:
                print("failure around", prop[0])
            try:
                styledict[prop[ii+jj]] = {'color':color, 'ls':'-.'}
            except:
                print("failure around", prop[-1])
        styledict[prop[ii]] = {'color':color, 'ls':'-', 'linewidth':4}

def make_plot_single_model(hist_f, hist_x_projection, hist_events, model_name, num_subfunctions = 100, display_events=False, display_trajectory=False, figsize=(4.5,3.5), external_legend=True):
    """
    Plot the different optimizers against each other for a single
    model.
    """

    # DEBUG xlabel was getting cut off
    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})

    # set the title
    title = model_name
    if model_name == 'protein':
        title = 'Logistic Regression, Protein Dataset'
    elif model_name == 'mnist':
        title = 'Autoencoder, 784 vis, 256 hid, MNIST Digits'
    elif model_name == 'GLM':
        title = 'GLM, soft rectifying nonlinearity'
        model_name = 'GLM_soft'
    #elif 'hard' in model_name:
        #title = 'Multi-Layer Perceptron, Rectified Linear'
    #elif 'soft' in model_name:
        #title = 'Multi-Layer Perceptron, Sigmoid'
    elif model_name == 'sumf':
        title = 'Sum of Norms'
    elif model_name == 'Pylearn_conv':
        title = 'Convolutional Network, CIFAR-10'

    # set up linestyle cycler
    styles = [
    {'color':'r', 'ls':'--'}, 
    {'color':'g', 'ls':'-'}, 
    {'color':'b', 'ls':'-.'}, 
    {'color':'k', 'ls':':'}, 
    {'color':'y', 'ls':'--'}, 
    {'color':'r', 'ls':'-'}, 
    {'color':'g', 'ls':'-.'}, 
    {'color':'b', 'ls':':'}, 
    {'color':'k', 'ls':'--'}, 
    {'color':'y', 'ls':'-'}, 
    {'color':'r', 'ls':'-.'}, 
    {'color':'g', 'ls':':'}, 
    {'color':'b', 'ls':'--'}, 
    {'color':'k', 'ls':'-'}, 
    {'color':'y', 'ls':'-.'}, 
    ]
    stylecycler = cycle(styles)
    styledict = defaultdict(lambda: next(stylecycler))

    ## override the default cycled styles for specific optimizers
    # LBFGS
    prop = [nm for nm in hist_f.keys() if 'LBFGS' in nm]
    for nm in prop:
        styledict[nm] = {'color':'r', 'ls':'-', 'linewidth':3}
        if 'batch' in nm:
            styledict[nm]['ls'] = ':'
    best_traces(prop, hist_f, styledict, 'r')
    # for nm in prop:
    #     styledict[nm] = {'color':'r', 'ls':'steps-', 'linewidth':4}
    # SAG
    prop = [nm for nm in hist_f.keys() if 'SAG' in nm]
    best_traces(prop, hist_f, styledict, 'g', neighbor_dist=0)
    # SGD
    prop = [nm for nm in hist_f.keys() if 'SGD' in nm and 'momentum' not in nm]
    best_traces(prop, hist_f, styledict, 'c', neighbor_dist=0)
    # SGD_momentum
    prop = [nm for nm in hist_f.keys() if 'SGD' in nm and 'momentum' in nm]
    best_traces(prop, hist_f, styledict, 'm', neighbor_dist=0)
    # ADA
    prop = [nm for nm in hist_f.keys() if 'ADA' in nm]
    best_traces(prop, hist_f, styledict, 'b', neighbor_dist=0)
    # SFO
    prop = [nm for nm in hist_f.keys() if nm == 'SFO'] # in nm and '_' not in nm]
    for nm in prop:
        styledict[nm] = {'color':'k', 'ls':'-', 'linewidth':4}
        # if 'BFGS' in nm:
        #     styledict[nm]['ls'] = '--'
        # if 'nat' in nm:
        #     #styledict[nm]['ls'] = '-.'
        #     styledict[nm]['ls'] = '-'
        #     styledict[nm]['linewidth'] = 4
        # if 'eta' in nm:
        #     styledict[nm]['ls'] = '--'
        #     styledict[nm]['color'] = 'k'
        #     # DEBUG
        #     #del hist_f[nm]
        # if 'nat' in nm and 'eta' in nm:
        #     styledict[nm] = {'color':'m', 'ls':'-', 'linewidth':4}
    # prop = [nm for nm in hist_f.keys() if 'SFO random' in nm]
    # for nm in prop:
    #     styledict[nm] = {'color':'y', 'ls':'-', 'linewidth':2}
    # prop = [nm for nm in hist_f.keys() if 'SFO cycle' in nm]
    # for nm in prop:
    #     styledict[nm] = {'color':'0.75', 'ls':'-', 'linewidth':2}
    # prop = [nm for nm in hist_f.keys() if 'SFO euc' in nm]
    # for nm in prop:
    #     styledict[nm] = {'color':'0.25', 'ls':'-', 'linewidth':2}

    zorder = dict()
    sorted_nm = sorted(hist_f.keys(), key=lambda nm: np.asarray(hist_f[nm])[-1], reverse=True)
    for ii, nm in enumerate(sorted_nm):
        zorder[nm] = ii

    # plot the learning trace, and save pdf
    fig = plt.figure(figsize=figsize)

    lines = []
    labels = []

    for nm in sorted_nicely(hist_f.keys()):
        ff = np.asarray(hist_f[nm])
        ff[ff>1.5*ff[0]] = np.nan
        xx = (50*np.arange(len(ff)).astype(float)+1)/num_subfunctions
        #line = plt.semilogy( xx, ff, label=nm, zorder=zorder[nm], **styledict[nm] )
        line = plt.semilogy( xx, ff, label=nm, zorder=zorder[nm], **styledict[nm] )
        lines.append(line[0])
        labels.append(nm)
        if display_events:
            # add special events
            for jj, events in enumerate(hist_events[nm]):
                st = {'s':100}
                if events.get('natural gradient subspace update', False):
                    st['marker'] = '>'
                    st['c'] = 'r'
                elif events.get('collapse subspace', False):
                    st['marker'] = '*'
                    st['c'] = 'y'
                elif events.get('step_failure', False):
                    st['marker'] = '<'
                    st['c'] = 'c'
                else:
                    continue
                plt.scatter(xx[jj], ff[jj], **st)

    ax = plt.axis()
    plt.axis([0, 50, 300, 3700])
    plt.yticks(np.arange(500, 4000, 500.0), ["%d"%tt for tt in np.arange(500, 4000, 500.0)])


    plt.ylabel( 'Full Batch Objective' )
    plt.xlabel( 'Effective Passes Through Data' )
    plt.title(title)
    plt.grid()
    plt.axes().set_axisbelow(True)

    # ax = plt.axis()
    # plt.axis([0, 50, ax[2], ax[3]])
    # ax = plt.axis()
    # if "Autoencoder" in title:
    #     plt.yticks(np.arange(10, 46, 5.0), ["%d"%tt for tt in np.arange(10, 46, 5.0)])
    #     plt.axis([ax[0], ax[1], 12.5, 35])
    # # elif "ICA" in title:
    # #     plt.axis([ax[0], ax[1], 0.23, 140])
    # elif "hard" in title:
    #     plt.axis([ax[0], ax[1], 1e-13, 1e1])
    # elif "Perceptron" in title:
    #     plt.axis([ax[0], ax[1], 1e-13, 1e1])
    try:
        plt.tight_layout()
    except:
        warnings.warn('tight_layout failed.  try running with an Agg backend.')

    # update the labels to prettier text
    labels = map(lambda x: str.replace(x, "SGD_momentum ", r"SGD+mom "), labels)
    labels = map(lambda x: str.replace(x, "SGD ", r"SGD eta="), labels)
    labels = map(lambda x: str.replace(x, "ADAGrad ", r"ADAGrad eta="), labels)
    labels = map(lambda x: str.replace(x, "SAG ", r"SAG L="), labels)
    labels = map(lambda x: str.replace(x, "L=", r"$L=$"), labels)
    labels = map(lambda x: str.replace(x, "eta=", r"$\eta=$"), labels)
    labels = map(lambda x: str.replace(x, "mu=", r"$\mu=$"), labels)
    def number_shaver(ch, regx = re.compile('(?<![\d.])0*(?:'
                                        '(\d+)\.?|\.(0)'
                                        '|(\.\d+?)|(\d+\.\d+?)'
                                        ')0*(?![\d.])')  ,
                      repl = lambda mat: mat.group(mat.lastindex)
                                         if mat.lastindex!=3
                                         else '0' + mat.group(3) ):
        return regx.sub(repl,ch)
    labels = map(number_shaver, labels)
    labels = map(lambda x: x + "$\ $", labels) # mixing TeX and non-TeX labels looks bad, so make them all TeX

    labels_original = labels
    labels = []
    for lab in labels_original:
        labels.append(lab)

    if external_legend:
        #TODO set figure size based on legend size
        legendsize=(6.25,2.6)
        figleg = plt.figure(figsize=legendsize)
        figleg.legend( lines, labels, 'center', ncol=2 )
        figleg.savefig(('figure_' + model_name + '_legend.pdf').replace(' ', '-'))
    else:
        fig.legend( loc='best' )

    fig.savefig(('figure_' + model_name + '_true.pdf').replace(' ', '-'))

    # find the minimum function value
    minf = np.inf
    for nm in hist_f.keys():
        ff = np.asarray(hist_f[nm])
        minf2 = np.nanmin(ff)
        minf = np.nanmin([minf, minf2])
    # plot the trace relative to minimum, and save pdf
    fig = plt.figure(figsize=figsize)
    for nm in sorted_nicely(hist_f.keys()):
        ff = np.asarray(hist_f[nm])
        ff[ff>1.5*ff[0]] = np.nan
        xx = (50*np.arange(len(ff)).astype(float)+1)/num_subfunctions
        plt.semilogy( xx, ff-minf, label=nm, zorder=zorder[nm], **styledict[nm] )
    plt.ylabel( 'Full Batch Objective - Minimum' )
    plt.xlabel( 'Effective Passes Through Data' )
    if not external_legend:
        plt.legend( loc='best' )
    plt.title(title)
    plt.grid()
    plt.axes().set_axisbelow(True)
    ax = plt.axis()
    plt.axis([0, 50, ax[2], ax[3]])
    try:
        plt.tight_layout()
    except:
        warnings.warn('tight_layout failed.  try running with an Agg backend.')
    fig.savefig(('figure_' + model_name + '_diff.pdf').replace(' ', '-'))

    if display_trajectory:
        # SAG
        prop = [nm for nm in hist_f.keys() if 'SAG' in nm]
        best_traces(prop, hist_f, styledict, 'g', neighbor_dist=0)
        # SGD
        prop = [nm for nm in hist_f.keys() if 'SGD' in nm]
        best_traces(prop, hist_f, styledict, 'c', neighbor_dist=0)
        # ADA
        prop = [nm for nm in hist_f.keys() if 'ADA' in nm]
        best_traces(prop, hist_f, styledict, 'b', neighbor_dist=0)

        # plot the learning trajectory in low-d projections, and save pdf
        # make the line styles appropriate
        for nm in hist_f.keys():
            styledict[nm].pop('ls', None)
            styledict[nm].pop('linewidth', None)
            #styledict[nm]['linestyle'] = 'None'
            styledict[nm]['marker'] = '.' # ','
            styledict[nm]['alpha'] = 0.5 # ','
        fig = plt.figure(figsize=figsize)
        nproj = 3 # could make larger
        for i1 in range(nproj):
            for i2 in range(nproj):
                plt.subplot(nproj, nproj, i1 + nproj*i2 + 1)
                for nm in sorted_nicely(hist_f.keys()):
                    xp = np.asarray(hist_x_projection[nm])
                    plt.plot( xp[:,i1], xp[:,i2], label=nm, zorder=zorder[nm], **styledict[nm] )
                    if display_events:
                        # add special events
                        for jj, events in enumerate(hist_events[nm]):
                            st = {'s':100}
                            if events.get('natural gradient subspace update', False):
                                st['marker'] = '>'
                                st['c'] = 'r'
                            elif events.get('collapse subspace', False):
                                st['marker'] = '*'
                                st['c'] = 'y'
                            elif events.get('step_failure', False):
                                st['marker'] = '<'
                                st['c'] = 'c'
                            else:
                                continue
                            plt.scatter(xp[jj,i1], xp[jj,i2], **st)
        if not external_legend:
            plt.legend( loc='best' )
        plt.suptitle(title)
        try:
            plt.tight_layout()
        except:
            warnings.warn('tight_layout failed.  try running with an Agg backend.')
        fig.savefig(('figure_' + model_name + '_trajectory.pdf').replace(' ', '-'))


def make_plots(history_nested, *args, **kwargs):
    for model_name in history_nested:
        history = history_nested[model_name]
        fig = make_plot_single_model(history['f'], history['x_projection'], history['events'], model_name, *args, **kwargs)

def load_results(fnames=None, base_fname='figure_data_'):
    """
    Load the function value traces during optimization for the 
    set of models and optimizers provided by fnames.  Find all
    files with matching filenames in the current directory if
    fnames not passed in.

    Output is a dictionary of dictionaries, where the inner dictionary
    contains different optimizers for each model, and the outer dictionary
    contains different models.

    Note that the files loaded can be as granular as a single optimizer
    for a single model for each file.
    """

    if fnames==None:
        fnames = glob.glob(base_fname + '*.npz')

    history_nested = {}
    for fn in fnames:
        data = np.load(fn)
        model_name = data['model_name'].tostring()
        model_name = 'GLM'
        print("loading", model_name)
        #history_nested[model_name] = data['history'][()].copy()
        if not model_name in history_nested:
            history_nested[model_name] = data['history'][()].copy()
        else:
            print("updating")
            for subkey in history_nested[model_name].keys():
                if subkey == 'x':
                    continue
                history_nested[model_name][subkey].update(data['history'][()].copy()[subkey])
        data.close()

    return history_nested

def save_results(trainer, base_fname='figure_data_'):
    """
    Save the function trace for different optimizers for a 
    given model to a .npz file.
    """
    fname = base_fname + trainer.model.name + ".npz"
    np.savez(fname, history=trainer.history, model_name=trainer.model.name)

def generate_data(num_passes=20, base_fname='figure_data_'):
    """
    train all the models in models_to_train using all the
    optimizers, and save the resulting function value traces
    in figure_data_*.npz files.
    """
    # models_to_train = ( 
    #                     figures_models.logistic,
    #                     figures_models.ContractiveAutoencoder,
    #                     figures_models.MLP_soft,
    #                     figures_models.MLP_hard,
    #                     #figures_models.CIFARConvNet,
    #                     figures_models.toy,
    #                     figures_models.ICA,
    #                     figures_models.Hopfield,
    #                     )
    #                     #)
    #                     #figures_models.MLP,)

    # models_to_train = ( 
    #                     #figures_models.CIFARConvNet,
    #                     #figures_models.Hopfield,
    #                     #figures_models.toy,
    #                     #figures_models.logistic,
    #                     #figures_models.ContractiveAutoencoder,
    #                     #figures_models.ICA,
    #                     figures_models.MLP_hard,
    #                     figures_models.MLP_soft,
    #                     )


    models_to_train = (figures_models.GLM,)
    #models_to_train = (figures_models.Hopfield,)
    #models_to_train = (figures_models.CIFARConvNet,)
    #models_to_train = (figures_models.toy,)
    
    for model_class in models_to_train:
        np.random.seed(0) # make experiments repeatable
        model = model_class()
        trainer = figures_train.figures_train(model)


        optimizers_to_use = (
                                trainer.SFO,
                                trainer.ADA,
                                trainer.SAG,
                                trainer.SGD,
                                trainer.LBFGS,
                                trainer.LBFGS_minibatch,
                                trainer.SGD_momentum,
                            )
        optimizers_to_use = (
                                trainer.LBFGS,
                                trainer.SGD,
                                trainer.LBFGS_minibatch,
                                trainer.SAG,
                                trainer.ADA,
                                trainer.SFO,
                                trainer.SGD_momentum,
                            )


        # optimizers_to_use = (
        #                         trainer.SFO,
        #                         trainer.SFO_nat,
        #                         trainer.SFO_eta,
        #                     )


        #optimizers_to_use = [trainer.ADA,trainer.SAG,trainer.SGD,]
        #optimizers_to_use = [trainer.SFO_variations,trainer.LBFGS,]
        optimizers_to_use = [trainer.SAG,]
        #optimizers_to_use = [trainer.SFO,trainer.LBFGS,trainer.ADA,]
        #optimizers_to_use = [trainer.LBFGS,trainer.LBFGS_minibatch]
        for optimizer in optimizers_to_use:
            np.random.seed(0) # make experiments exactly repeatable
            print("\n\n\n" + model.name + "\n" + str(optimizer))
            optimizer(num_passes=num_passes)
            # save_results doesn't need to be called until outside this loop,
            # but this way we can peak at partial results
            # also, note that saved files for the same model but different optimizers
            # can be combined in plots, just by calling load_results with all the saved files
            save_results(trainer, base_fname=base_fname)

            # # DEBUG
            # hist_f_nested = load_results(base_fname=base_fname)
            # make_plots(hist_f_nested)
            # plt.show()

def train_and_plot(num_passes=51, base_fname='GLM_nrocinu_4_shipweight'):
    base_fname += "_%s_"%(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    generate_data(num_passes=num_passes, base_fname=base_fname)
    history_nested = load_results(base_fname=base_fname)
    make_plots(history_nested)
    plt.show()

if __name__ == '__main__':
    train_and_plot()
