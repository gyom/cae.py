#!/usr/bin/env python

import dae_theano
#dae_theano = reload(dae_theano)
mydae = dae_theano.DAE(n_hiddens=80,
                       epochs=100,
                       learning_rate=0.001,
                       prob_large_noise=0.0,
                       jacobi_penalty=0.1,
                       batch_size=1000)

import debian_spiral
import numpy

N = 10000
(X,Y) = debian_spiral.sample(N, 0.0)

data = numpy.vstack((X,Y)).T
mydae.fit(data, verbose=True)

print "mydae.W is "
print mydae.W
print "mydae.b is "
print mydae.b
print "mydae.c is "
print mydae.c

#print data


plotgrid_N_buckets = 20
window_width = 1.0
(plotgrid_X, plotgrid_Y) = numpy.meshgrid(numpy.arange(- window_width,
                                                       window_width,
                                                       2 * window_width / plotgrid_N_buckets),
                                          numpy.arange(- window_width,
                                                       window_width,
                                                       2 * window_width / plotgrid_N_buckets))
plotgrid = numpy.vstack([numpy.hstack(plotgrid_X), numpy.hstack(plotgrid_Y)]).T
D = numpy.sqrt(plotgrid[:,0]**2 + plotgrid[:,1]**2)
plotgrid = plotgrid[D<0.7]

print plotgrid_X.shape
print plotgrid_Y.shape

print "Will keep only %d points on the plotting grid after starting from %d." % (plotgrid.shape[0], plotgrid_X.shape[0])

print "Making predictions for the grid."

grid_pred = mydae.encode_decode(plotgrid)
#grid_pred = predict(plotgrid, W, grid, (lambda X, xi: kernel(X,xi,sigma)))
grid_error = numpy.sqrt(((grid_pred - plotgrid)**2).sum(axis=1)).mean()

print "Generating plot."

if True:
    import matplotlib
    matplotlib.use('Agg')
    import pylab
    import os
    # print only one point in 100
    #pylab.scatter(data[0:-1:100,0], data[0:-1:100,1], c='#f9a21d')
    pylab.scatter(data[:,0], data[:,1], c='#f9a21d')
    pylab.hold(True)
    arrows_scaling = 1.0
    pylab.quiver(plotgrid[:,0],
                 plotgrid[:,1],
                 arrows_scaling * (grid_pred[:,0] - plotgrid[:,0]),
                 arrows_scaling * (grid_pred[:,1] - plotgrid[:,1]))
    pylab.draw()
    # pylab.axis([-0.6, 0.6, -0.6, 0.6])
    pylab.axis([-window_width*1.5, window_width*1.5, -window_width*1.5, window_width*1.5])
    pylab.savefig(os.path.join('/u/alaingui/umontreal/cae.py/plots', 'spiral_reconstruction_grid_01.png'))
    pylab.close()