#!/usr/bin/env python
# encoding: utf-8
"""
dae_theano.py

Guillaume Alain.
"""

import sys
import os
import pdb
import numpy

from theano import *
import theano.tensor as T

class DAE(object):
    """
    A DAE with sigmoid input units and sigmoid
    hidden units.
    """
    def __init__(self, 
                 n_hiddens=24,
                 W=None,
                 c=None,
                 b=None,
                 learning_rate=0.001,
                 jacobi_penalty=0.1,
                 batch_size=10,
                 epochs=100,
                 prob_large_noise=0.2,
                 large_noise_sigma=0.5,
                 want_logging = True):
        """
        Initialize a DAE.
        
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
        self.prob_large_noise = prob_large_noise
        self.large_noise_sigma = large_noise_sigma
        self.want_logging = want_logging

        # read-only
        self.output_scaling_factor = 2.0

        # logging
        
        if self.want_logging:
            self.logging = {}
            for k in ['noisy', 'noiseless']:
                self.logging[k] = {}
                self.logging[k]['mean_abs_loss'] = []
                self.logging[k]['var_abs_loss'] = []

                self.logging[k]['mean_abs_act'] = []
                self.logging[k]['var_abs_act'] = []

                self.logging[k]['mean_abs_ract'] = []
                self.logging[k]['var_abs_ract'] = []

                self.logging[k]['mean_abs_grad_W'] = []
                self.logging[k]['var_abs_grad_W'] = []


        # then setup the theano functions once
        self.theano_setup()
    
    def theano_setup(self):
    
        W = T.dmatrix('W')
        b = T.dvector('b')
        c = T.dvector('c')
        x = T.dmatrix('x')
    
        s = T.dot(x, W) + c
        # h = 1 / (1 + T.exp(-s))
        # h = T.nnet.sigmoid(s)
        h = T.tanh(s)
        # r = T.dot(h,W.T) + b
        # r = theano.printing.Print("r=")(2*T.tanh(T.dot(h,W.T) + b))
        ract = T.dot(h,W.T) + b
        r = self.output_scaling_factor * T.tanh(ract)
    
        #g  = function([W,b,c,x], h)
        #f  = function([W,b,c,h], r)
        #fg = function([W,b,c,x], r)
    
        # Another variable to be able to call a function
        # with a noisy x and compare it to a reference x.
        y = T.dmatrix('y')

        all_losses = ((r - y)**2)
        loss = T.sum(all_losses)
        #loss = ((r - y)**2).sum()
        
        self.theano_encode_decode = function([W,b,c,x], r)
        self.theano_all_losses = function([W,b,c,x,y], [all_losses, T.abs_(s), T.abs_(ract)])
        self.theano_gradients = function([W,b,c,x,y], [T.grad(loss, W), T.grad(loss, b), T.grad(loss, c)])

    def encode_decode(self, x):
        return self.theano_encode_decode(self.W, self.b, self.c, x)

    def model_loss(self, x, useNoise = True):
        """
        Computes the error of the model with respect
        to the total cost.
        
        -------
        x: array-like, shape (n_examples, n_inputs)
        
        Returns
        -------
        all_losses: array-like, shape (n_examples,)
        's' the prenonlinear unit activations (whatever)
        """

        if useNoise:
            return self.theano_all_losses(self.W, self.b, self.c, x + numpy.random.normal(scale=self.jacobi_penalty, size=x.shape), x.copy())
        else:
            return self.theano_all_losses(self.W, self.b, self.c, x.copy(), x.copy())

    def one_step_grad_descent(self, x, perform_update = True, jacobi_penalty_override = None):
        """
        Perform one step of gradient descent on the
        DAE objective using the examples {\bf x}.
        
        Parameters
        ----------
        x: array-like, shape (n_examples, n_inputs)
        """
        
        jacobi_penalty = self.jacobi_penalty

        if numpy.random.uniform() < self.prob_large_noise:
            jacobi_penalty = self.large_noise_sigma
        if not (jacobi_penalty_override == None):
            jacobi_penalty = jacobi_penalty_override

        if jacobi_penalty > 0:
            perturbed_x = x + numpy.random.normal(scale = jacobi_penalty, size=x.shape)
        else:
            perturbed_x = x.copy()

        # total_perturbation = numpy.abs(x - perturbed_x).sum()
        # if total_perturbation < 1e-4:
        #    print "Bear in mind that your gradient will probably be 0 if you don't have enough noise."
        #    print "Right now the total perturbation is %f." % total_perturbation

        #y = self.encode_decode(perturbed_x)

        grad_W, grad_b, grad_c = self.theano_gradients(self.W, self.b, self.c, perturbed_x, x.copy())

        if perform_update:
            self.W = self.W - self.learning_rate * grad_W
            self.b = self.b - self.learning_rate * grad_b
            self.c = self.c - self.learning_rate * grad_c

        return (grad_W, grad_b, grad_c)


    def reset_params(self, d = 2):
        #if self.W == None:
        self.W = numpy.random.uniform( low = -0.1, high = 0.1, size=(d, self.n_hiddens) )
        #self.W = numpy.random.uniform(
        #    low = - 4.0 * numpy.sqrt(6./(d + self.n_hiddens)),
        #    high = 4.1 * numpy.sqrt(6./(d + self.n_hiddens)),
        #    size=(d, self.n_hiddens))
        self.c = numpy.zeros(self.n_hiddens)
        self.b = numpy.zeros(d)

    def fit(self, X, verbose=False):
        """
        Fit the model to the data X.
        
        Parameters
        ----------
        X: array-like, shape (n_examples, n_inputs)
            Training data, where n_examples in the number of examples
            and n_inputs is the number of features.
        """
        
        self.reset_params(X.shape[1])

        # We'll be using double indirection to shuffle
        # around the minibatches. We will keep shuffling
        # the indices in 'inds' and the chunks will be
        # described by 'inds_ranges' which will be a collection
        # of ranges.
        #
        # ex :
        #    self.batch_size is 3
        #    inds is [10,3,2,5,9,8,6,0,7,4,1]
        #    inds_ranges is [(0,3), (3,6), (6,9), (9,10)]
        #
        # Results in batches being
        #    [10,3,2], [5,9,8], [0,7,4], [1]

        inds = range(X.shape[0])
        
        n_batches = len(inds) / self.batch_size
        inds_ranges = []
        for k in range(0, n_batches):

            start = k * self.batch_size
            if start >= X.shape[0]:
                break

            end = (k+1) * self.batch_size
            end = min(end, X.shape[0])

            # Keep in mind that the lower bound is inclusive
            # and the upper bound is exclusive.
            # This is why 'end' terminates with X.shape[0]
            # while that value would be illegal for 'start'.

            inds_ranges.append( (start, end) )
            
        if verbose:
            print "The ranges used for the minibatches are "
            print inds_ranges


        for epoch in range(self.epochs):

            # Shuffle the 'inds', because we don't modify
            # the 'inds_ranges'. Only one of them has to change.
            numpy.random.shuffle(inds)
            
            for (start, end) in inds_ranges:
                X_minibatch = X[inds[start:end]]
                self.one_step_grad_descent(X_minibatch)

            if self.want_logging:
                if verbose and (epoch % 100 == 0):
                    sys.stdout.flush()
                    print "Epoch %d" % epoch
                    self.perform_logging(verbose = True, X = X)
                else:
                    self.perform_logging(verbose = False, X = X)

    def perform_logging(self, verbose = False, X = None):

        # The 'X' parameter is used to log the gradients.
        # We are recomputing them and wasting computation here, but
        # whenever we train a model we shouldn't be doing all the
        # extensive logging that we do for debugging purposes.
        # The 'X' is generally data from a minibatch.

        # Two blocks of code where the only word that changes is
        # 'noisy' to 'noiseless'.

        # 'noisy'
        noisy_all_losses, noisy_all_abs_act, noisy_all_abs_ract = self.model_loss(X, useNoise=True)

        self.logging['noisy']['mean_abs_loss'].append( numpy.abs(noisy_all_losses).mean() )
        self.logging['noisy']['var_abs_loss'].append( numpy.abs(noisy_all_losses).var() )

        self.logging['noisy']['mean_abs_act'].append( noisy_all_abs_act.mean() )
        self.logging['noisy']['var_abs_act'].append( noisy_all_abs_act.var() )

        self.logging['noisy']['mean_abs_ract'].append( noisy_all_abs_ract.mean() )
        self.logging['noisy']['var_abs_ract'].append( noisy_all_abs_ract.var() )

        if not (X == None):
            grad_W, grad_b, grad_c = self.one_step_grad_descent(X, perform_update = False)
            self.logging['noisy']['mean_abs_grad_W'].append( numpy.abs(grad_W).mean() )
            self.logging['noisy']['var_abs_grad_W'].append( numpy.abs(grad_W).var() )


        # 'noiseless'
        noiseless_all_losses, noiseless_all_abs_act, noiseless_all_abs_ract = self.model_loss(X, useNoise=False)

        self.logging['noiseless']['mean_abs_loss'].append( numpy.abs(noiseless_all_losses).mean() )
        self.logging['noiseless']['var_abs_loss'].append( numpy.abs(noiseless_all_losses).var() )

        self.logging['noiseless']['mean_abs_act'].append( noiseless_all_abs_act.mean() )
        self.logging['noiseless']['var_abs_act'].append(  noiseless_all_abs_act.var() )

        self.logging['noiseless']['mean_abs_ract'].append( noiseless_all_abs_ract.mean() )
        self.logging['noiseless']['var_abs_ract'].append( noiseless_all_abs_ract.var() )

        if not (X == None):
            grad_W, grad_b, grad_c = self.one_step_grad_descent(X, perform_update = False, jacobi_penalty_override = 0.0)
            self.logging['noiseless']['mean_abs_grad_W'].append( numpy.abs(grad_W).mean() )
            self.logging['noiseless']['var_abs_grad_W'].append( numpy.abs(grad_W).var() )

        if verbose:
            print "  -- Exact --"
            print "    Loss : %0.6f" % self.logging['noiseless']['mean_abs_loss'][-1]
            #print "    Activations Mean Abs. Hidden = %0.6f, Reconstructed = %0.6f" % (abs_act, abs_ract)
            print "  -- Noise --"
            print "    Loss : %0.6f" % self.logging['noisy']['mean_abs_loss'][-1]
            #print "    Activations Mean Abs. Hidden = %0.6f, Reconstructed = %0.6f" % (noise_abs_act, noise_abs_ract)
            #print "  Gradient W Mean Abs = %f" % numpy.abs(self.grad_W).mean()
            print "\n"



def main():
    pass


if __name__ == '__main__':
    main()


# import dae_theano
# # dae_theano = reload(dae_theano)
# mydae = dae_theano.DAE()

# X = numpy.random.random((1000, 2))
# X[:,1] = 13*X[:,0]
# mydae.fit(X)

