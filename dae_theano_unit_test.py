#!/usr/bin/env python

import numpy
import unittest
import dae_theano

def reference_encode_decode(W, b, c, x):
    s = numpy.dot(x, W) + c
    h = numpy.tanh(s)
    ract = numpy.dot(h,W.T) + b
    r = numpy.tanh(ract)
    return (s,h,ract,r)

def reference_gradients(W, b, c, x, y, epsilon=1.0e-6):
    # Proceed with numerical differentiation
    # by adding the epsilon to each of the components
    # and doing it the long way (with some tolerance
    # value given).

    grad_W = numpy.zeros(W.shape)
    grad_b = numpy.zeros(b.shape)
    grad_c = numpy.zeros(c.shape)

    (_, _, _, ground_r) = reference_encode_decode(W, b, c, x)
    ground_loss = ((y - ground_r)**2).sum()

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_mod = W.copy()
            W_mod[i,j] = W_mod[i,j] + epsilon
            (_, _, _, r) = reference_encode_decode(W_mod, b, c, x)
            # and now the elementary numerical estimate of the derivative
            grad_W[i,j] = (((y - r)**2).sum() - ground_loss) / epsilon

    for k in range(b.shape[0]):
        b_mod = b.copy()
        b_mod[k] = b_mod[k] + epsilon
        (_, _, _, r) = reference_encode_decode(W, b_mod, c, x)
        grad_b[k] = (((y - r)**2).sum() - ground_loss) / epsilon

    for k in range(c.shape[0]):
        c_mod = c.copy()
        c_mod[k] = c_mod[k] + epsilon
        (_, _, _, r) = reference_encode_decode(W, b, c_mod, x)
        grad_c[k] = (((y - r)**2).sum() - ground_loss) / epsilon

    return (grad_W, grad_b, grad_c)

def assert_total_difference(testCaseInstance, A, B, abs_tol=1.0e-6):
    testCaseInstance.assertTrue(numpy.abs(A - B).sum() < abs_tol)


class DAE_test_container():

    def __init__(self, test, n_visibles = 1, n_hiddens = 1, params_scaling = 0.0):
        # 'test' contains an instance of a class like
        # TestGradientsWithV1H1(unittest.TestCase)
        # so that we can call it's "assertTrue" method.
        self.test = test

        mydae = dae_theano.DAE(n_hiddens = n_hiddens)
        if params_scaling > 0.0:
            mydae.c = numpy.random.normal(size=(n_hiddens,), scale = params_scaling)
            mydae.b = numpy.random.normal(size=(n_visibles,), scale = params_scaling)
            mydae.W = numpy.random.normal(size=(n_hiddens, n_visibles), scale = params_scaling)
        else:
            mydae.c = numpy.zeros((n_hiddens,))
            mydae.b = numpy.zeros((n_visibles,))
            mydae.W = numpy.zeros((n_hiddens, n_visibles))

        self.n_visibles = n_visibles
        self.n_hiddens = n_hiddens
        self.mydae = mydae

    def compare_gradients(self, input_noise_scale = 1.0, use_spiral = False):

        if use_spiral:
            if not (self.mydae.n_visibles == 2):
                print "It doesn't make sense to ask to use the spiral if you don't have 2 input units."
            assert(self.mydae.n_visibles == 2)
            import debian_spiral
            N = 10
            (X0,Y0) = debian_spiral.sample(N, 0.0)
            X = numpy.vstack((X0,Y0)).T
        else:
            N = 10
            X = numpy.zeros((N, self.n_visibles))

        if input_noise_scale > 0:
            noise = numpy.random.normal(scale = input_noise_scale, size = X.shape)
        else:
            noise = numpy.zeros(X.shape)
        noisy_X = X + noise

        dae_grad_W, dae_grad_b, dae_grad_c = self.mydae.theano_gradients(self.mydae.W,
                                                                         self.mydae.b,
                                                                         self.mydae.c,
                                                                         noisy_X, X)
        ref_grad_W, ref_grad_b, ref_grad_c = reference_gradients(self.mydae.W,
                                                                 self.mydae.b,
                                                                 self.mydae.c,
                                                                 noisy_X, X)

        print "--- ref ---"
        print ref_grad_W
        print ref_grad_b
        print ref_grad_c
        print "--- dae ---"
        print dae_grad_W
        print dae_grad_b
        print dae_grad_c

        # We scaled the output of the tanh for the DAE.
        # That results in all the gradients being multiplied
        # by 4 so we have to correct for this.
        alpha = self.mydae.output_scaling_factor ** 2

        assert_total_difference(self.test, dae_grad_W, alpha * ref_grad_W, 0.01)
        assert_total_difference(self.test, dae_grad_b, alpha * ref_grad_b, 0.01)
        assert_total_difference(self.test, dae_grad_c, alpha * ref_grad_c, 0.01)


class TestGradientsWithV1H1(unittest.TestCase):

    def test_gradient_v1h1(self):
        for input_noise_scale in [0.0, 1.0, 10.0]:
            for params_scaling in [0.0, 1.0, 10.0]:
                self.daetc = DAE_test_container(self,
                                                n_visibles = 1,
                                                n_hiddens = 1,
                                                params_scaling = params_scaling)
                self.daetc.compare_gradients(input_noise_scale = input_noise_scale)


class TestGradientsWithManyUnits(unittest.TestCase):

    def test_gradient_many_units(self):
        n_visibles = numpy.random.random() * 100
        n_hiddens = numpy.random.random() * 100

        for input_noise_scale in [0.0, 1.0, 10.0]:
            for params_scaling in [0.0, 1.0, 10.0]:
                self.daetc = DAE_test_container(self,
                                                n_visibles = 1, 
                                                n_hiddens = 1,
                                                params_scaling = params_scaling)
                self.daetc.compare_gradients(input_noise_scale = input_noise_scale)



if __name__ == '__main__':
    unittest.main()


