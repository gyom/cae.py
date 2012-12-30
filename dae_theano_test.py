#!/usr/bin/env python

import dae_theano
#dae_theano = reload(dae_theano)
mydae = dae_theano.DAE(n_hiddens=56,
                       epochs=1000,
                       learning_rate=0.01,
                       prob_large_noise=0.000,
                       large_noise_sigma=1.0,
                       jacobi_penalty=0.01,
                       batch_size=25)

import debian_spiral
import numpy

n_spiral_samples = 2000
spiral_samples_noise = 0.01
(X,Y) = debian_spiral.sample(n_spiral_samples, spiral_samples_noise)

data = numpy.vstack((X,Y)).T
mydae.fit(data, verbose=True)

mydae.set_params_to_best_noisy()

# print "mydae.W is "
# print mydae.W
# print "mydae.b is "
# print mydae.b
# print "mydae.c is "
# print mydae.c
# print data


import os

# create a new directory to host the result files of this experiment
output_directory = '/u/alaingui/umontreal/cae.py/plots/experiment_%0.6d' % int(numpy.random.random() * 1.0e6)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


import matplotlib
matplotlib.use('Agg')
import pylab


pylab.hold(True)

p1, = pylab.plot(mydae.logging['noisy']['mean_abs_loss'], label='noisy', c='#f9761d', linewidth = 2)
for s in [-1.0, 1.0]:
    pylab.plot(mydae.logging['noisy']['mean_abs_loss']
               + s * numpy.sqrt(mydae.logging['noisy']['var_abs_loss']),
               c='#f9a21d', linestyle='dashed')

p2, = pylab.plot(mydae.logging['noiseless']['mean_abs_loss'], label='noiseless', c='#9418cd', linewidth = 2)
for s in [-1.0, 1.0]:
    pylab.plot(mydae.logging['noiseless']['mean_abs_loss']
               + s * numpy.sqrt(mydae.logging['noiseless']['var_abs_loss']),
               c='#d91986', linestyle='dashed')

pylab.title('Absolute Losses')
pylab.legend([p1,p2], ["noisy", "noiseless"])
pylab.draw()
pylab.savefig(os.path.join(output_directory, 'absolute_losses.png'), dpi=300)
pylab.close()




################################
##                            ##
## spiral reconstruction grid ##
##                            ##
################################

plotgrid_N_buckets = 30
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
print "grid_error = %0.6f" % grid_error


print "Generating plot."

# print only one point in 100
# pylab.scatter(data[0:-1:100,0], data[0:-1:100,1], c='#f9a21d')
pylab.scatter(data[:,0], data[:,1], c='#f9a21d')
pylab.hold(True)
arrows_scaling = 1.0
pylab.quiver(plotgrid[:,0],
             plotgrid[:,1],
             arrows_scaling * (grid_pred[:,0] - plotgrid[:,0]),
             arrows_scaling * (grid_pred[:,1] - plotgrid[:,1]))
pylab.draw()
#pylab.axis([-0.6, 0.6, -0.6, 0.6])
pylab.axis([-0.7, 0.7, -0.7, 0.7])
# pylab.axis([-window_width*1.5, window_width*1.5, -window_width*1.5, window_width*1.5])
pylab.savefig(os.path.join(output_directory, 'spiral_reconstruction_grid.png'), dpi=300)
pylab.close()



###################################
##                               ##
## html file for showing results ##
##                               ##
###################################


html_file_path = os.path.join(output_directory, 'results.html')
f = open(html_file_path, "w")

hyperparams_contents = """
<p>nbr visible units : %d</p>
<p>nbr hidden  units : %d</p>

<p>batch size : %d</p>
<p>epochs : %d</p>

<p>learning rate  : %0.6f</p>
<p>training noise : %0.6f</p>
""" % (mydae.W.shape[0],
       mydae.n_hiddens,
       mydae.batch_size,
       mydae.epochs,
       mydae.learning_rate,
       mydae.jacobi_penalty)

#n_spiral_samples = 30
#spiral_samples_noise = 0.0


params_contents = ""


graphs_contents = """
<img src='%s' width='600px'/>
<img src='%s' width='600px'/>
""" % ('absolute_losses.png', 'spiral_reconstruction_grid.png')


contents = """
<html>
    <head>
        <style>
            div.listing {
                margin-left: 25px;
                margin-top: 5px;
                margin-botton: 5px;
            } 
        </style>
    </head>
<body>
<h3>Hyperparameters</h3>
<div class='listing'>%s</div>
<h3>Parameters</h3>
<div class='listing'>%s</div>
<h3>Graphs</h3>
<div class='listing'>%s</div>
</body>
</html>""" % (hyperparams_contents,
              params_contents,
              graphs_contents)

f.write(contents)
f.close()


