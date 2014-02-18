#!/usr/bin/env python
# encoding: utf-8
"""
cae.py

u pythonic library for Contractive Auto-Encoders. This is
for people who want to give CAEs a quick try and for people
who want to understand how they are implemented. For this
purpose we tried to make the code as simple and clean as possible.
The only dependency is numpy, which is used to perform all
expensive operations. The code is quite fast, however much better
performance can be achieved using the Theano version of this code.

Created by Yann N. Dauphin, Salah Rifai on 2012-01-17.
Copyright (c) 2012 Yann N. Dauphin, Salah Rifai. All rights reserved.
"""

import numpy as np


class CAE(object):
    """
    A Contractive Auto-Encoder (CAE) with sigmoid input units and sigmoid
    hidden units.
    """
    def __init__(self, 
                 n_hiddens=1024,
                 jacobi_penalty=0.1,
                 W=None,
                 hidbias=None,
                 visbias=None):
        """
        Initialize a CAE.
        
        Parameters
        ----------
        n_hiddens : int, optional
            Number of binary hidden units
        jacobi_penalty : float, optional
            Scalar by which to multiply the gradients coming from the jacobian
            penalty.
        W : array-like, shape (n_inputs, n_hiddens), optional
            Weight matrix, where n_inputs in the number of input
            units and n_hiddens is the number of hidden units.
        hidbias : array-like, shape (n_hiddens,), optional
            Biases of the hidden units
        visbias : array-like, shape (n_inputs,), optional
            Biases of the input units
        """
        self.n_hiddens = n_hiddens
        self.jacobi_penalty = jacobi_penalty
        self.W = W
        self.hidbias = hidbias
        self.visbias = visbias


    def _sigmoid(self, x):
        """
        Implements the logistic function.
        
        Parameters
        ----------
        x: array-like, shape (M, N)

        Returns
        -------
        x_new: array-like, shape (M, N)
        """
        return 1. / (1. + np.exp(-np.maximum(np.minimum(x, 100), -100)))

    def encode(self, x):
        """
        Computes the hidden code for the input {\bf x}.
        
        Parameters
        ----------
        x: array-like, shape (n_examples, n_inputs)

        Returns
        -------
        h: array-like, shape (n_examples, n_hiddens)
        """
        return self._sigmoid(np.dot(x, self.W) + self.hidbias)

    def decode(self, h):
        """
        Compute the reconstruction from the hidden code {\bf h}.
        
        Parameters
        ----------
        h: array-like, shape (n_examples, n_hiddens)
        
        Returns
        -------
        x: array-like, shape (n_examples, n_inputs)
        """
        return self._sigmoid(np.dot(h, self.W.T) + self.visbias)

    def reconstruct(self, x):
        """
        Compute the reconstruction of the input {\bf x}.
        
        Parameters
        ----------
        x: array-like, shape (n_examples, n_inputs)
        
        Returns
        -------
        x_new: array-like, shape (n_examples, n_inputs)
        """
        return self.decode(self.encode(x))

    def loss(self, x, h=None, r=None):
        """
        Computes the error of the model with respect
        to the total cost.
        
        Parameters
        ----------
        x: array-like, shape (n_examples, n_inputs)
        h: array-like, shape (n_examples, n_hiddens), optional
        r: array-like, shape (n_examples, n_inputs), optional
        
        Returns
        -------
        loss: array-like, shape (n_examples,)
        """
        if h == None:
            h = self.encode(x)
        if r == None:
            r = self.decode(h)
        
        def _reconstruction_loss(h, r):
            """
            Computes the error of the model with respect
            to the reconstruction (L2) cost.
            """
            return 1/2. * ((r-x)**2).sum()/x.shape[0]

        def _jacobi_loss(h):
            """
            Computes the error of the model with respect
            the Frobenius norm of the jacobian.
            """
            return ((h *(1-h))**2 *  (self.W**2).sum(0)).sum()/x.shape[0]
        recon_loss = _reconstruction_loss(h, r)
        jacobi_loss = _jacobi_loss(h)
        return (recon_loss + self.jacobi_penalty * jacobi_loss)

    def get_params(self):
        return dict(W=self.W, hidbias=self.hidbias, visbias=self.visbias)

    def set_params(self, theta):
        self.W = theta['W']
        self.hidbias = theta['hidbias']
        self.visbias = theta['visbias']

    def f_df(self, theta, x):
        """
        Compute objective and gradient of the CAE objective using the
        examples {\bf x}.
        
        Parameters
        ----------
        x: array-like, shape (n_examples, n_inputs)
        
        Parameters
        ----------
        loss: array-like, shape (n_examples,)
            Value of the loss function for each example before the step.
        """
        self.set_params(theta)
        h = self.encode(x)
        r = self.decode(h)
        def _contraction_jacobian():
            """
            Compute the gradient of the contraction cost w.r.t parameters.
            """
            a = 2*(h * (1 - h))**2 
            d = ((1 - 2 * h) * a * (self.W**2).sum(0)[None, :])
            b = np.dot(x.T / x.shape[0], d)
            c = a.mean(0) * self.W
            return (b + c), d.mean(0)
        
        def _reconstruction_jacobian():
            """                                                                 
            Compute the gradient of the reconstruction cost w.r.t parameters.      
            """
            dr = (r - x) / x.shape[0]
            dr *= r * (1-r)
            dd = np.dot(dr.T, h)
            dh = np.dot(dr, self.W) * h * (1. - h)
            de = np.dot(x.T, dh)
            return (dd + de), dr.sum(0), dh.sum(0)

        W_rec, c_rec, b_rec = _reconstruction_jacobian()
        W_con, b_con = _contraction_jacobian()
        dW = W_rec + self.jacobi_penalty * W_con
        dhidbias = b_rec + self.jacobi_penalty * b_con
        dvisbias = c_rec 
        return self.loss(x, h, r), dict(W=dW, hidbias=dhidbias, visbias=dvisbias);

    def init_weights(self, n_input, dtype=np.float32):
        self.W = np.asarray(np.random.uniform(
            #low=-4*np.sqrt(6./(n_input+self.n_hiddens)),
            #high=4*np.sqrt(6./(n_input+self.n_hiddens)),
            low=-1./np.sqrt(self.n_hiddens),
            high=1./np.sqrt(self.n_hiddens),
            size=(n_input, self.n_hiddens)), dtype=dtype)
        self.hidbias = np.zeros(self.n_hiddens, dtype=dtype)
        self.visbias = np.zeros(n_input, dtype=dtype)
        return self.get_params()
