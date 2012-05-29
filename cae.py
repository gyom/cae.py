#!/usr/bin/env python
# encoding: utf-8
"""
cae.py

A pythonic library for Contractive Auto-Encoders. This is
for people who want to give CAEs a quick try and for people
who want to understand how they are implemented. For this
purpose we tried to make the code as simple and clean as possible.
The only dependency is numpy, which is used to perform all
expensive operations. The code is quite fast, however much better
performance can be achieved using the Theano version of this code.

Created by Yann N. Dauphin, Salah Rifai on 2012-01-17.
Copyright (c) 2012 Yann N. Dauphin, Salah Rifai. All rights reserved.
"""

import sys
import os
import pdb
import numpy


class CAE(object):
    """
    A Contrative Auto-Encoder (CAE) with sigmoid input units and sigmoid
    hidden units.
    """
    def __init__(self, 
                 n_hiddens=1024,
                 W=None,
                 c=None,
                 b=None,
                 learning_rate=0.001,
                 jacobi_penalty=0.1,
                 batch_size=10,
                 epochs=200,
                 use_cae_jacobian=True):
        """
        Initialize a CAE.
        
        Parameters
        ----------
        n_hiddens : int, optional
            Number of binary hidden units
        W : array-like, shape (n_inputs, n_hiddens), optional
            Weight matrix, where n_inputs in the number of input
            units and n_hiddens is the number of hidden units.
        c : array-like, shape (n_hiddens,), optional
            Biases of the hidden units
        b : array-like, shape (n_inputs,), optional
            Biases of the input units
        learning_rate : float, optional
            Learning rate to use during learning
        jacobi_penalty : float, optional
            Scalar by which to multiply the gradients coming from the jacobian
            penalty.
        batch_size : int, optional
            Number of examples to use per gradient update
        epochs : int, optional
            Number of epochs to perform during learning
        """
        self.n_hiddens = n_hiddens
        self.W = W
        self.c = c
        self.b = b
        self.learning_rate = learning_rate
        self.jacobi_penalty = jacobi_penalty
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_cae_jacobian = use_cae_jacobian
        
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
        return 1. / (1. + numpy.exp(-x)) 
    
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
        return self._sigmoid(numpy.dot(x, self.W) + self.c)
    
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
        return self._sigmoid(numpy.dot(h, self.W.T) + self.b)
    
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
    
    def jacobian(self, x):
        """
        Compute jacobian of {\bf h} with respect to {\bf x}.
        
        Parameters
        ----------
        x: array-like, shape (n_examples, n_inputs)
        
        Returns
        -------
        jacobian: array-like, shape (n_examples, n_hiddens, n_inputs)
        """
        h = self.encode(x)
        
        return (h * (1 - h))[:, :, None] * self.W.T
    
    def reconstruction_jacobian(self, x):
        """
        Compute jacobian of {\bf r} with respect to {\bf x}.

        Parameters
        ----------
        x: array-like, shape (n_examples, n_inputs)

        Returns
        -------
        jacobian: array-like, shape (n_examples, n_inputs, n_inputs)
        """
        h = self.encode(x)
        r = self.decode(h)

        return (r * (1-r))[:,:,None] * self.W * (h * (1 - h))[:, :, None] * self.W.T
    
    def sample(self, x, sigma=1):
        """
        Sample a point {\bf y} starting from {\bf x} using the CAE
        generative process.
        
        Parameters
        ----------
        x: array-like, shape (n_examples, n_inputs)
        sigma: float
        
        Returns
        -------
        y: array-like, shape (n_examples, n_inputs)
        """
        h = self.encode(x)
        
        s = h * (1. - h)
        
        JJ = numpy.dot(self.W.T, self.W) * s[:, None, :] * s[:, :, None]
        
        alpha = numpy.random.normal(0, sigma, h.shape)
        
        delta = (alpha[:, :, None] * JJ).sum(1)
        
        return self.decode(h + delta)
    
    def loss(self, x):
        """
        Computes the error of the model with respect
        to the total cost.
        
        -------
        x: array-like, shape (n_examples, n_inputs)
        
        Returns
        -------
        loss: array-like, shape (n_examples,)
        """
        def _reconstruction_loss():
            """
            Computes the error of the model with respect
            
            to the reconstruction (cross-entropy) cost.
            
            """
            z = self.reconstruct(x)
            return ((x-z)**2).sum(1).mean()
            #return (- (x * numpy.log(z) + (1 - x) * numpy.log(1 - z)).sum(1)).mean()

        def _jacobi_loss():
            """
            Computes the error of the model with respect
            
            the Frobenius norm of the jacobian.
            
            """
            if self.use_cae_jacobian:
                j = self.jacobian(x)
            else:
                j = self.reconstruction_jacobian(x)
            return (j**2).sum(2).sum(1).mean()
        
        return _reconstruction_loss() + self.jacobi_penalty * _jacobi_loss()
    
    def _fit(self, x):
        """
        Perform one step of gradient descent on the CAE objective using the
        examples {\bf x}.
        
        Parameters
        ----------
        x: array-like, shape (n_examples, n_inputs)
        """
        def _fit_contraction():
            """
            Compute the gradient of the contraction cost w.r.t parameters.
            """
            h = self.encode(x)

            a = (h * (1 - h))**2 

            d = ((1 - 2 * h) * a * (self.W**2).sum(0)[None, :])

            b = x[:, :, None] * d[:, None, :]

            c = a[:, None, :] * self.W

            return (b+c).mean(0), (d).mean(0)
            
        def _fit_reconstruction():
            """                                                                 
            Compute the gradient of the reconstruction cost w.r.t parameters.      
            """

            h = self.encode(x)
            r = self.decode(h)

            #dedr = -( x/r - (1 - x)/(1 - r) ) 
            dedr = 2*(r-x)

            a = r*(1-r)
            b = h*(1-h)
            
            od = a * dedr
            oe = b * numpy.dot(od, self.W)

            gW = x[ :, :, None]  * oe[ :, None, : ] + od[:, :, None]*h[:, None, :]
            # gW = x[ :, :, None]  * oe[ :, None, : ]

            return gW.mean(0), od.mean(0), oe.mean(0)

        W_rec, b_rec, c_rec = _fit_reconstruction()
        W_jac, c_jac = _fit_contraction()
        self.W -= self.learning_rate * (W_rec + self.jacobi_penalty * W_jac)
        self.c -= self.learning_rate * (c_rec + self.jacobi_penalty * c_jac)
        self.b -= self.learning_rate * b_rec


    def fit(self, X, verbose=False):
        """
        Fit the model to the data X.
        
        Parameters
        ----------
        X: array-like, shape (n_examples, n_inputs)
            Training data, where n_examples in the number of examples
            and n_inputs is the number of features.
        """
        if self.W == None:
            self.W = numpy.random.uniform(
                low=-4*numpy.sqrt(6./(X.shape[1]+self.n_hiddens)),
                high=4*numpy.sqrt(6./(X.shape[1]+self.n_hiddens)),
                size=(X.shape[1], self.n_hiddens))
            self.c = numpy.zeros(self.n_hiddens)
            self.b = numpy.zeros(X.shape[1])
        
        inds = range(X.shape[0])
        
        numpy.random.shuffle(inds)
        
        n_batches = len(inds) / self.batch_size
        
        for epoch in range(self.epochs):
            for minibatch in range(n_batches):
                self._fit(X[inds[minibatch::n_batches]])
            
            if verbose:
                loss = self.loss(X).mean()
                sys.stdout.flush()
                print "Epoch %d, Loss = %.2f" % (epoch, loss)
    
    def mu_C_for_quadratic_loss(self, X):
        #r0 = self.reconstruct(numpy.zeros(self.b.shape))
        #h0 = self.encode(numpy.zeros(self.b.shape))
        #
        # We can probably code this in a vectorial way later by being a bit
        # clever with the numpy.diag .
        (n,d) = X.shape
        results_mu = numpy.zeros((n,d))
        results_cov = numpy.zeros((n,d,d))
        for i in numpy.arange(0,n):
            r0 = self.reconstruct(X[i,:])
            h0 = self.encode(X[i,:])
            A = numpy.diag(r0*(1-r0))
            B = numpy.diag(h0*(1-h0))
            J = numpy.dot(numpy.dot(numpy.dot(self.W,B),self.W.T),A)
            Z = numpy.linalg.inv(numpy.eye(len(r0)) - J)
            mu = X[i,:] + numpy.dot(Z,r0-X[i,:])
            #C = numpy.dot(numpy.dot(numpy.dot(numpy.dot(numpy.linalg.pinv(self.W.T), B), self.W.T), numpy.linalg.inv(A)), Z)
            C = numpy.dot(J, Z)
            #
            results_mu[i,:] = mu
            results_cov[i,:,:] = C
        return (results_mu, results_cov)



def main():
    pass


if __name__ == '__main__':
    main()

