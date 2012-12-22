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
                 large_noise_sigma=0.5):
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

        # read-only
        self.output_scaling_factor = 2.0

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

    def one_step_grad_descent(self, x):
        """
        Perform one step of gradient descent on the
        DAE objective using the examples {\bf x}.
        
        Parameters
        ----------
        x: array-like, shape (n_examples, n_inputs)
        """
        
        if numpy.random.uniform() < self.prob_large_noise:
            jacobi_penalty = self.large_noise_sigma
        perturbed_x = x + numpy.random.normal(scale = self.jacobi_penalty, size=x.shape)
        
        total_perturbation = numpy.abs(x - perturbed_x).sum()
        if total_perturbation < 1e-4:
            print "Bear in mind that your gradient will probably be 0 if you don't have enough noise."
            print "Right now the total perturbation is %f." % total_perturbation

        y = self.encode_decode(perturbed_x)

        grad_W, grad_b, grad_c = self.theano_gradients(self.W, self.b, self.c, x.copy(), y)

        if False:
            print "numpy.abs(self.W).mean() = %f" % numpy.abs(self.W).mean()
            # print "numpy.abs(self.b).mean() = %f" % numpy.abs(self.b).mean()
            #print "numpy.abs(self.c).mean() = %f" % numpy.abs(self.c).mean()

            print "numpy.abs(grad_W).mean() = %f" % numpy.abs(grad_W).mean()
            #print "numpy.abs(b_g).mean() = %f" % numpy.abs(b_g).mean()
            #print "numpy.abs(c_g).mean() = %f" % numpy.abs(c_g).mean()

        self.W -= self.learning_rate * grad_W
        self.b -= self.learning_rate * grad_b
        self.c -= self.learning_rate * grad_c

        # for logging, not for anything else
        self.grad_W = grad_W
        self.grad_b = grad_b
        self.grad_c = grad_c

    def reset_params(self, d = 2):
        #if self.W == None:
        self.W = numpy.random.uniform(
            low = - 0.1 * numpy.sqrt(6./(d + self.n_hiddens)),
            high = 0.1 * numpy.sqrt(6./(d + self.n_hiddens)),
            size=(d, self.n_hiddens))
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

        inds = range(X.shape[0])
        
        numpy.random.shuffle(inds)
        
        n_batches = len(inds) / self.batch_size
        
        print "len(inds) = %d" % len(inds)
        print "n_batches = %d" % n_batches
        print "self.batch_size = %d" % self.batch_size
        print "self.epochs = %d" % self.epochs

        for epoch in range(self.epochs):
            for minibatch in range(n_batches):
                self.one_step_grad_descent(X[inds[minibatch::n_batches]])
            
            if verbose and (epoch % 10 == 0):

                noise_all_losses, noise_all_abs_act, noise_all_abs_ract = self.model_loss(X, useNoise=True)
                noise_abs_loss = numpy.abs(noise_all_losses).mean()
                noise_abs_act = noise_all_abs_act.mean()
                noise_abs_ract = noise_all_abs_ract.mean()

                all_losses, all_abs_act, all_abs_ract = self.model_loss(X, useNoise=False)
                abs_loss = numpy.abs(all_losses).mean()
                abs_act = all_abs_act.mean()
                abs_ract = all_abs_ract.mean()

                sys.stdout.flush()
                print "Epoch %d" % epoch
                print "  -- Exact --"
                print "    Loss : %0.6f" % abs_loss
                print "    Activations Mean Abs. Hidden = %0.6f, Reconstructed = %0.6f" % (abs_act, abs_ract)
                print "  -- Noise --"
                print "    Loss : %0.6f" % noise_abs_loss
                print "    Activations Mean Abs. Hidden = %0.6f, Reconstructed = %0.6f" % (noise_abs_act, noise_abs_ract)
                print "  Gradient W Mean Abs = %f" % numpy.abs(self.grad_W).mean()
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

