
import hamiltonian_monte_carlo as hmc
import unittest
import numpy as np

class TestBasicSimulation(unittest.TestCase):

    def test_sample_from_1D_gaussian(self):

        energy_norm_cst = 0.5*np.log(2*3.141592)

        U = lambda q : energy_norm_cst + (q**2).sum()/2
        grad_U = lambda q : q.sum()

        # This won't work as well with epsilon = 1.e-3
        # or with L = 10. It'll get something close, but
        # the histogram plot will look difform.
        epsilon = 0.1
        L = 10
        N = 100000

        N_accepted_counter = 0
        X = np.zeros((N,1))
        # starts at the origin with X[0,:]
        for n in np.arange(1,N):
            X[n,:] = hmc.one_step_update(U, grad_U, epsilon, L, X[n-1,:])
            if X[n,:] != X[n-1,:]:
                N_accepted_counter = N_accepted_counter + 1

        sample_mean = X.mean()
        sample_var = X.var()

        self.assertTrue(abs(sample_mean) < 0.01)
        self.assertTrue(abs(sample_var - 1.0) < 0.01)

        if False:

            print "sample mean : %0.6f" % sample_mean
            print "sample var  : %0.6f" % sample_var
            print "acceptance ration : %0.6f" % (N_accepted_counter * 1.0 / N,)

            import matplotlib
            matplotlib.use('Agg')
            import pylab
            import os
            pylab.hist(X[:,0], 50)
            pylab.draw()
            pylab.savefig(os.path.join('/u/alaingui/umontreal/cae.py/plots/', 'debug_HMC.png'), dpi=300)
            pylab.close()



if __name__ == '__main__':
    unittest.main()
