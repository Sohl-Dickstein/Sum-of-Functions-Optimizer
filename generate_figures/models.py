"""
Model classes for each of the demo cases.  Each class contains
an objective function f_df, initial parameters theta_init, a
reference for each subfunction subfunction_references, and a
set of full_objective_references that are evaluated every update step
to make the plots of objective function vs. learning iteration.

This is designed to be called by figure_convergence.py.


Author: Jascha Sohl-Dickstein (2014)
This software is made available under the Creative Commons
Attribution-Noncommercial License.
( http://creativecommons.org/licenses/by-nc/3.0/ )
"""

import numpy as np
import scipy.special
import warnings
import random
from figures_cae import CAE
from os.path import join

try:
    import pyGLM.simGLM as glm
    import pyGLM.gabor as gabor
except:
    pass

# numpy < 1.7 does not have np.random.choice
def my_random_choice(n, k, replace):
    perm = np.random.permutation(n)
    return perm[:k]
if hasattr(np.random, 'choice'):
    random_choice = np.random.choice
else:
    random_choice = my_random_choice


class toy:
    """
    Toy problem.  Sum of squared errors from random means, raised
    to random powers.
    """

    def __init__(self, num_subfunctions=100, num_dims=10):
        self.name = '||x-u||^a, a in U[1.5,4.5]'

        # create the array of subfunction identifiers
        self.subfunction_references = []
        N = num_subfunctions
        for i in range(N):
            npow = np.random.rand()*3. + 1.5
            mn = np.random.randn(num_dims,1)
            self.subfunction_references.append([npow,mn])
        self.full_objective_references = self.subfunction_references

        ## initialize parameters
        self.theta_init = np.random.randn(num_dims,1)

    def f_df(self, x, args):
        npow = args[0]/2.
        mn = args[1]
        f = np.sum(((x-mn)**2)**npow)
        df = npow*((x-mn)**2)**(npow-1.)*2*(x-mn)
        scl = 1. / np.prod(x.shape)
        return f*scl, df*scl


class Hopfield:
    def __init__(self, num_subfunctions=100, reg=1., scale_by_N=True):
        """
        Train a Hopfield network/Ising model using MPF.

        Adapted from code by Chris Hillar, Kilian Koepsell, Jascha Sohl-Dickstein, 2011

        TODO insert Hopfield and MPF references.
        """
        self.name = 'Hopfield'
        self.reg = reg/num_subfunctions

        # Load data
        X, _ = load_mnist()

        # binarize data
        X = (np.sign(X-0.5)+1)/2

        # only keep units which actually change state
        gd = ((np.sum(X,axis=1) > 0) & (np.sum(1-X,axis=1) > 0))
        X = X[gd,:]
        # TODO -- discard units with correlation of exactly 1?

        # break the data up into minibatches
        self.subfunction_references = []
        for mb in range(num_subfunctions):
            self.subfunction_references.append(X[:, mb::num_subfunctions].T)
        #self.full_objective_references = (X[:, random_choice(X.shape[1], 10000, replace=False)].copy().T,)
        self.full_objective_references = self.subfunction_references

        if scale_by_N:
            self.scl = float(num_subfunctions) / float(X.shape[1])
        else:
            self.scl = 100. / float(X.shape[1])

        # parameter initialization
        self.theta_init = np.random.randn(X.shape[0], X.shape[0])/np.sqrt(X.shape[0])/10.
        self.theta_init = (self.theta_init + self.theta_init.T)/2.


    def f_df(self, J, X):
        J = (J + J.T)/2.
        X = np.atleast_2d(X)
        S = 2 * X - 1
        Kfull = np.exp(-S * np.dot(X, J.T) + .5 * np.diag(J)[None, :])
        dJ = -np.dot(X.T, Kfull * S) + .5 * np.diag(Kfull.sum(0))
        dJ = (dJ + dJ.T)/2.
        dJ = np.nan_to_num(dJ)

        K = Kfull.sum()
        if not np.isfinite(K):
            K = 1e50

        K *= self.scl
        dJ *= self.scl

        K += self.reg * np.sum(J**2)
        dJ += 2. * self.reg * J
        return K, dJ


class logistic:
    """
    logistic regression on "protein" dataset
    """

    def __init__(self, num_subfunctions=100, scale_by_N=True):
        self.name = 'protein logistic regression'

        try:
            data = np.loadtxt('figure_data/bio_train.dat')
        except:
            raise Exception("Missing data.  Download from  and place in figure_data subdirectory.")

        target = data[:,[2]]
        feat = data[:,3:]
        feat = self.whiten(feat)
        feat = np.hstack((feat, np.ones((data.shape[0],1))))

        # create the array of subfunction identifiers
        self.subfunction_references = []
        N = num_subfunctions
        nper = float(feat.shape[0])/float(N)
        if scale_by_N:
            lam = 1./nper**2
            scl = 1./nper
        else:
            default_N = 100.
            nnp = float(feat.shape[0])/default_N
            lam = (1./nnp**2) * (default_N / float(N))
            scl = 1./nnp
        for i in range(N):
            i_s = int(np.floor(i*nper))
            i_f = int(np.floor((i+1)*nper))
            if i == N-1:
                # don't drop any at the end
                i_f = target.shape[0]
            l_targ = target[i_s:i_f,:]
            l_feat = feat[i_s:i_f,:]
            self.subfunction_references.append([l_targ, l_feat, lam, scl, i])
        self.full_objective_references = self.subfunction_references

        self.theta_init = np.random.randn(feat.shape[1],1)/np.sqrt(feat.shape[1])/10. # parameters initialization

    # remove first order dependencies from X, and scale to unit norm
    def whiten(self, X,fudge=1E-10):
        max_whiten_lines = 10000

        # zero mean
        X -= np.mean(X, axis=0).reshape((1,-1))

        # the matrix X should be observations-by-components
        # get the covariance matrix
        Xsu = X[:max_whiten_lines,:]
        Xcov = np.dot(Xsu.T,Xsu)/Xsu.shape[0]
        # eigenvalue decomposition of the covariance matrix
        try:
            d,V = np.linalg.eigh(Xcov)
        except:
            print("could not get eigenvectors and eigenvalues using numpy.linalg.eigh of ", Xcov.shape, Xcov)
            d,V = np.linalg.eig(Xcov)
        d = np.nan_to_num(d+fudge)
        d[d==0] = 1
        V = np.nan_to_num(V)

        # a fudge factor can be used so that eigenvectors associated with
        # small eigenvalues do not get overamplified.
        # TODO(jascha) D could be a vector not a matrix
        D = np.diag(1./np.sqrt(d))

        D = np.nan_to_num(D)

        # whitening matrix
        W = np.dot(np.dot(V,D),V.T)
        # multiply by the whitening matrix
        Y = np.dot(X,W)
        return Y

    def sigmoid(self, u):
        return 1./(1.+np.exp(-u))
    def f_df(self, x, args):
        target = args[0]
        feat = args[1]
        lam = args[2]
        scl = args[3]

        feat = feat*(2*target - 1)
        ip = -np.dot(feat, x.reshape((-1,1)))
        et = np.exp(ip)
        logL = np.log(1. + et)
        etrat = et/(1.+et)
        bd = np.nonzero(ip[:,0]>50)[0]
        logL[bd,:] = ip[bd,:]
        etrat[bd,:] = 1.
        logL = np.sum(logL)
        dlogL = -np.dot(feat.T, etrat)

        logL  *= scl
        dlogL *= scl

        reg = lam*np.sum(x**2)
        dreg = 2.*lam*x

        return logL + reg, dlogL+dreg


class ICA:
    """
    ICA with Student's t-experts on MNIST images.
    """

    def __init__(self, num_subfunctions=100):
        self.name = 'ICA'

        # Load data
        #X = load_cifar10_imagesonly()
        X, _ = load_mnist()

        # do PCA to eliminate degenerate dimensions and whiten
        C = np.dot(X, X.T) / X.shape[1]
        w, V = np.linalg.eigh(C)
        # take only the non-negligible eigenvalues
        mw = np.max(np.real(w))
        max_ratio = 1e4
        gd = np.nonzero(np.real(w) > mw/max_ratio)[0]
        # # whiten
        # P = V[gd,:]*(np.real(w[gd])**(-0.5)).reshape((-1,1))
        # don't whiten -- make the problem harder
        P = V[gd,:]
        X = np.dot(P, X)
        X /= np.std(X)

        # break the data up into minibatches
        self.subfunction_references = []
        for mb in range(num_subfunctions):
            self.subfunction_references.append(X[:, mb::num_subfunctions])
        # compute the full objective on all the data
        self.full_objective_references = self.subfunction_references
        # # the subset of the data used to compute the full objective function value
        # idx = random_choice(X.shape[1], 10000, replace=False)
        # self.full_objective_references = (X[:,idx].copy(),)

        ## initialize parameters
        num_dims = X.shape[0]
        self.theta_init = {'W':np.random.randn(num_dims, num_dims)/np.sqrt(num_dims),
                  'logalpha':np.random.randn(num_dims,1).ravel()}

        # rescale the objective and gradient so that the same hyperparameter ranges work for
        # ICA as for the other objectives
        self.scale = 1. / self.subfunction_references[0].shape[1] / 100.


    def f_df(self, params, X):
        """
        ICA objective function and gradient, using a Student's t-distribution prior.
        The energy function has form:
            E = \sum_i \alpha_i \log( 1 + (\sum_j W_{ij} x_j )^2 )

        params is a dictionary containing the filters W and the log of the
        X is the training data, with each column corresponding to a sample.

        L is the average negative log likelihood.
        dL is its gradient.
        """
        W = params['W']
        logalpha = params['logalpha'].reshape((-1,1))
        alpha = np.exp(logalpha)+0.5

        ## calculate the energy
        ff = np.dot(W, X)
        ff2 = ff**2
        off2 = 1 + ff2
        lff2 = np.log( off2 )
        alff2 = lff2 * alpha.reshape((-1,1))
        E = np.sum(alff2)

        ## calculate the energy gradient
        # could just sum instead of using the rscl 1s vector.  this is functionality
        # left over from MPF MATLAB code.  May want it again in a future project though.
        rscl = np.ones((X.shape[1],1))
        lt = (ff/off2) * alpha.reshape((-1,1))
        dEdW = 2 * np.dot(lt * rscl.T, X.T)
        dEdalpha = np.dot(lff2, rscl)
        dEdlogalpha = (alpha-0.5) * dEdalpha

        ## calculate log Z
        nu = alpha * 2. - 1.
        #logZ = -np.log(scipy.special.gamma((nu + 1.) / 2.)) + 0.5 * np.log(np.pi) + \
        #    np.log(scipy.special.gamma(nu/2.))
        logZ = -scipy.special.gammaln((nu + 1.) / 2.) + 0.5 * np.log(np.pi) + \
            scipy.special.gammaln((nu/2.))
        logZ = np.sum(logZ)
        ## DEBUG slogdet has memory leak!
        ## eg, call "a = np.linalg.slogdet(random.randn(5000,5000))"
        ## repeatedly, and watch memory usage.  So, we do this with an
        ## explicit eigendecomposition instead
        ## logZ += -np.linalg.slogdet(W)[1]
        W2 = np.dot(W.T, W)
        W2eig, _ = np.linalg.eig(W2)
        logZ += -np.sum(np.log(W2eig))/2.

        ## calculate gradient of log Z
        # log determinant contribution
        dlogZdW = -np.linalg.inv(W).T
        if np.min(nu) < 0:
            dlogZdnu = np.zeros(nu.shape)
            warnings.warn('not a normalizable distribution!')
            E = np.inf
        else:
            dlogZdnu = -scipy.special.psi((nu + 1) / 2 )/2 + \
                scipy.special.psi( nu/2 )/2
        dlogZdalpha = 2. * dlogZdnu
        dlogZdlogalpha = (alpha-0.5) * dlogZdalpha

        ## full objective and gradient
        L = (E + logZ) * self.scale
        dLdW = (dEdW + dlogZdW) * self.scale
        dLdlogalpha = (dEdlogalpha + dlogZdlogalpha) * self.scale

        ddL = {'W':dLdW, 'logalpha':dLdlogalpha.ravel()}

        if not np.isfinite(L):
            warnings.warn('numerical problems')
            L = np.inf

        return L, ddL


class DeepAE:
    """
    Deep Autoencoder from Hinton, G. E. and Salakhutdinov, R. R. (2006)
    """
    def __init__(self, num_subfunctions=50, num_dims=10, objective='l2'):
        # don't introduce a Theano dependency until we have to
        from utils import _tonp

        self.name = 'DeepAE'
        layer_sizes = [ 28*28, 1000, 500, 250, 30]
        #layer_sizes = [ 28*28, 20]
        # Load data
        X, y = load_mnist()
        # break the data up into minibatches
        self.subfunction_references = []
        for mb in range(num_subfunctions):
            self.subfunction_references.append([X[:, mb::num_subfunctions], y[mb::num_subfunctions]])
        # evaluate on subset of training data
        self.n_full = 10000
        idx = random_choice(X.shape[1], self.n_full, replace=False)
        ##use all the training data for a smoother plot
        #idx = np.array(range(X.shape[1]))
        self.full_objective_references = [[X[:,idx].copy(), y[idx].copy()]]
        from dropout.deepae import build_f_df # here so theano not required for import
        self.theano_f_df, self.model = build_f_df(layer_sizes, use_bias=True, objective=objective)
        crossent_params = False
        if crossent_params:
            history = np.load('/home/poole/Sum-of-Functions-Optimizer/sfo_output.npz')
            out = dict(history=history['arr_0'])
            params = out['history'].item()['x']['SFO']
            self.theta_init = params
        else:
            self.theta_init = [param.get_value() for param in self.model.params]

    def f_df(self, theta, args, gpu_batch_size=128):
        X = args[0].T
        y = args[1]
        rem = np.mod(len(X), gpu_batch_size)
        n_batches = (len(X) - rem) /  gpu_batch_size
        splits = np.split(np.arange(len(X) - rem), n_batches)
        if rem > 0:
            splits.append(np.arange(len(X)-rem, len(X)))
        sum_results = None
        for split in splits:
            theano_args = theta + [X[split]]
            # Convert to float32 so that this works on GPU
            theano_args = [arg.astype(np.float32) for arg in theano_args]
            results = self.theano_f_df(*theano_args)
            results = [_tonp(result) for result in results]
            if sum_results is None:
                sum_results = results
            else:
                sum_results  = [cur_res + new_res for cur_res, new_res in zip(results, sum_results)]
        # Divide by number of datapoints.
        sum_results = [result/len(X) for result in sum_results]
        return sum_results[0], sum_results[1:]

class MLP:
    """
    Multi-layer-perceptron
    """
    def __init__(self, num_subfunctions=100, num_dims=10, rectifier='soft'):
        self.name = 'MLP'
        #layer_sizes = [ 28*28, 1200, 10 ]
        #layer_sizes = [ 28*28, 1200, 10 ]
        #layer_sizes = [ 28*28, 500, 120, num_dims ]
        #layer_sizes = [ 28*28, 120, 12, num_dims ]
        layer_sizes = [ 28*28, 1200, 1200, num_dims ]
        # Load data
        X, y = load_mnist()
        # break the data up into minibatches
        self.subfunction_references = []
        for mb in range(num_subfunctions):
            self.subfunction_references.append([X[:, mb::num_subfunctions], y[mb::num_subfunctions]])
        # evaluate on subset of training data
        idx = random_choice(X.shape[1], 5000, replace=False)
        #use all the training data for a smoother plot
        #idx = np.array(range(X.shape[1]))
        self.full_objective_references = [[X[:,idx].copy(), y[idx].copy()]]
        from dropout.mlp import build_f_df # here so theano not required for import
        self.theano_f_df, self.model = build_f_df(layer_sizes, rectifier=rectifier,
            use_bias=True)
        self.theta_init = [param.get_value() for param in self.model.params]
    def f_df(self, theta, args):
        X = args[0].T
        y = args[1]
        theano_args = theta + [X, y]
        results = self.theano_f_df(*theano_args)
        return results[0], results[1:]

class MLP_hard(MLP):
    """
    Multi-layer-perceptron with rectified-linear nonlinearity
    """
    def __init__(self, num_subfunctions=100, num_dims=10):
        MLP.__init__(self, num_subfunctions=num_subfunctions, num_dims=num_dims, rectifier='hard')
        self.name += ' hard'
class MLP_soft(MLP):
    """
    Multi-layer-perceptron with sigmoid nonlinearity
    """
    def __init__(self, num_subfunctions=100, num_dims=10):
        MLP.__init__(self, num_subfunctions=num_subfunctions, num_dims=num_dims, rectifier='soft')
        self.name += ' soft'

class ContractiveAutoencoder:
    """
    Contractive autoencoder on MNIST dataset.
    """
    def __init__(self, num_subfunctions=100, num_dims=10):
        self.name = 'Contractive Autoencoder'

        # Load data
        X, y = load_mnist()
        # break the data up into minibatches
        self.subfunction_references = []
        for mb in range(num_subfunctions):
            self.subfunction_references.append(X[:, mb::num_subfunctions])
        #self.full_objective_references = (X[:,np.random.choice(X.shape[1], 1000, replace=False)].copy(),)
        self.full_objective_references = (X[:, random_choice(X.shape[1], 1000, replace=False)].copy(),)
        #self.full_objective_references = (X.copy(),)
        # Initialize CAE model
        self.cae = CAE(n_hiddens=256, jacobi_penalty=1.0)
        # Initialize parameters
        self.theta_init = self.cae.init_weights(X.shape[0], dtype=np.float64)

    def f_df(self, theta, X):
        return self.cae.f_df(theta, X.T)

class PylearnModel:
    def __init__(self, filename, load_fnc, num_subfunctions=100, num_dims=10):
        # Import here so we don't depend on pylearn elsewhere
        from nnet.model_gradient import load_model
        self.name = 'Pylearn'
        X, y = load_fnc()
        batch_size = X.shape[0] / num_subfunctions
        mg = load_model(filename, batch_size=batch_size)
        self.mg = mg
        self.model = mg.model
        self._f_df = mg.f_df


        # break the data up into minibatches
        self.subfunction_references = []
        for mb in range(num_subfunctions):
            self.subfunction_references.append([X[ mb::num_subfunctions,...], y[mb::num_subfunctions,...]])
        # evaluate on subset of training data
        idx = random_choice(X.shape[0], 1000, replace=False)
        #use all the training data for a smoother plot
        #idx = np.array(range(X.shape[1]))
        self.full_objective_references = [[X[idx,...].copy(), y[idx,...].copy()]]
        self.theta_init = [param.get_value() for param in self.mg.params]

    def f_df(self, thetas, args):
        thetas32 = [theta.astype(np.float32) for theta in thetas]
        return self._f_df(thetas32, args)

class MNISTConvNet(PylearnModel):
    def __init__(self, num_subfunctions=100, num_dims=10):
        fn = 'nnet/mnist.yaml'
        #super(ConvNet, self).__init__(fn, num_subfunctions, num_dims)
        PylearnModel.__init__(self,fn, load_mnist, num_subfunctions, num_dims)
        self.name += '_conv'

class CIFARConvNet(PylearnModel):
    def __init__(self, num_subfunctions=100, num_dims=10):
        fn = 'nnet/conv.yaml'
        #super(ConvNet, self).__init__(fn, num_subfunctions, num_dims)
        PylearnModel.__init__(self,fn, load_cifar, num_subfunctions, num_dims)
        self.name += '_conv'


class GLM:
    """
    Train a GLM on simulated data.
    """

    def __init__(self, num_subfunctions_ratio=0.05, baseDir='/home/nirum/data/retina/glm-feb-19/'):
    #def __init__(self, num_subfunctions_ratio=0.05, baseDir='/home/nirum/data/retina/glm-feb-19/small'):
        self.name = 'GLM'

        print('Initializing parameters...')

        ## FOR CUSTOM SIZES
        #self.params = glm.setParameters(m=5000, dh=10, ds=50) # small
        #self.params = glm.setParameters(m=5000, dh=10, ds=49) # small
        #self.params = glm.setParameters(m=20000, dh=100, ds=500) # large
        #self.params = glm.setParameters(m=20000, dh=25, ds=256, n=5) # huge
        #self.params = glm.setParameters(m=1e5, dh=10, ds=256, n=50) # Jascha huge
        #self.params = glm.setParameters(m=1e5, n=100, ds=256, dh=10) # shared

        ## FROM EXTERNAL DATA FILES
        # load sizes of external data
        shapes = np.load(join(baseDir, 'metadata.npz'))

        # set up GLM parameters
        self.params = glm.setParameters(dh=40, ds=shapes['stimSlicedShape'][1], n=shapes['rateShape'][1])
        #self.params = glm.setParameters(dh=40, ds=shapes['stimSlicedShape'][1], n=2)

        ## initialize parameters
        print('Generating model...')
        #self.theta_true = glm.generateModel(self.params)
        self.theta_init = glm.generateModel(self.params)
        for key in self.theta_init.keys():
            self.theta_init[key] /= 1e3

        #print('Simulating model to generate data...')
        #self.data = glm.generateData(self.theta_true, self.params)

        # load external data files as memmapped arrays
        print('Loading external data...')
        self.data = glm.loadExternalData('stimulus_sliced.dat', 'rates.dat', shapes, baseDir=baseDir)

        # all the data
        batch_size = self.data['x'].shape[0]
        #batch_size = 10000

        print('ready to go!')

        #trueMinimum, trueGrad = self.f_df(self.theta_true, (0, batch_size))
        #print('Minimum for true parameters %g'%(trueMinimum))

        #print('Norm of the gradient at the minimum:')
        #for key in trueGrad.keys():
            #print('grad[' + key + ']: %g'%(np.linalg.norm(trueGrad[key])))

        # print out some network information
        #glm.visualizeNetwork(self.theta_true, self.params, self.data)

        # break the data up into minibatches

        self.N = int(np.ceil(np.sqrt(batch_size)*num_subfunctions_ratio))
        self.subfunction_references = []
        samples_per_subfunction = int(np.floor(batch_size/self.N))
        for mb in range(self.N):
            print(mb, self.N)
            start_idx = mb*samples_per_subfunction
            end_idx = (mb+1)*samples_per_subfunction
            self.subfunction_references.append((start_idx, end_idx,))

        self.full_objective_references = self.subfunction_references
        print('initialized ...')
        #self.full_objective_references = random.sample(self.subfunction_references, int(num_subfunctions/10))


    def f_df(self, theta, idx_range):
        """
        objective assuming Poisson noise

        function [fval grad] = objPoissonGLM(theta, datapath)
         Computes the Poisson log-likelihood objective and gradient
         for the generalized linear model (GLM)
        """

        data_subf = dict()
        for key in self.data.keys():
            data_subf[key] = np.array(self.data[key][idx_range[0]:idx_range[1]])

        return glm.f_df(theta, data_subf, self.params)



# def load_fingerprints(nsamples=None):
#     """ load fingerprint image
#         1 <= m <= 10 (subjects), 1 <= n <= 8  (fingers)
#     """
#     from PIL import Image

#     img_list = []
#     for m = range(1,11):
#         for n = range(1,9):
#             fname = os.path.join('figure_data/DB2_B', '1%02d_%d.tif'%(m,n))
#             arr = np.array(Image.open(fname)) # unit8
#             img = arr.astype(np.double)
#             img = img[100:-64,28:-28]
#             return img[::2,::2]






def load_cifar10_imagesonly(nsamples=None):
    X = np.load('figure_data/cifar10_images.npy')
    if nsamples == None:
        nsamples = X.shape[0]
    perm = random_choice(X.shape[0], nsamples, replace=False)
    X = X[perm, :]
    return X.T

def load_cifar(nsamples=None):
    try:
        from pylearn2.utils import serial
    except:
        raise Exception("pylearn2 must be installed.")
    try:
        dset = serial.load('figure_data/train.pkl')
    except:
        raise Exception("Missing data.  Download CIFAR!")
    X = dset.X
    y = dset.y
    # Convert to one hot
    one_hot = np.zeros((y.shape[0], 10), dtype='float32')
    for i in range(y.shape[0]):
        one_hot[i, y[i]] = 1

    if nsamples == None:
        nsamples = X.shape[0]
    perm = random_choice(X.shape[0], nsamples, replace=False)
    X = X[perm, :]
    X = X.reshape(-1, 3,32,32)
    X = X.astype(np.float32)
    y = one_hot[perm, :]
    return X,y


def load_mnist(nsamples=None):
    """
    Load the MNIST dataset.
    """

    try:
        data = np.load('figure_data/mnist_train.npz')
    except:
        raise Exception("Missing data.  Download mnist_train.npz from  and place in figure_data subdirectory.")
    X = data['X']
    y = data['y']
    if nsamples == None:
        nsamples = X.shape[0]
    perm = random_choice(X.shape[0], nsamples, replace=False)
    X = X[perm, :]
    X = X.T
    y = y[perm]
    return X, y
