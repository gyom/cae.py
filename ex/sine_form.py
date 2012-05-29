import numpy
import cae

N = 100
noise = 0.01
domain = 0.2 + 0.6*numpy.array(sorted(numpy.random.random((N,))))
image = 0.5 + 0.3*numpy.sin((domain-0.2)*2.6*numpy.pi) + numpy.random.normal(0, noise, domain.shape)
curve_data = numpy.vstack((domain, image)).T

my_cae = cae.CAE(n_hiddens=256,
    W=None,
    c=None,
    b=None,
    learning_rate=0.001,
    jacobi_penalty=0.1,
    batch_size=10,
    epochs=10000)

my_cae.fit(curve_data)


import pylab


################  quiver + scatter ####################

grid_points = numpy.array([(x,y) for x in numpy.arange(0, 1.00001, 0.1) for y in numpy.arange(0, 1.00001, 0.1)])
reconstructed_grid_points = my_cae.reconstruct(grid_points)

X = grid_points[:,0]
Y = grid_points[:,1]
U = reconstructed_grid_points[:,0] -  grid_points[:,0]
V = reconstructed_grid_points[:,1] -  grid_points[:,1]

if False:
    p = pylab.subplot(111)
    pylab.hold(True)
    pylab.quiver(X, Y, U, V, scale=1.0)
    pylab.axis([0., 1., 0., 1.])
    pylab.scatter(curve_data[:,0], curve_data[:,1])
    pylab.show()
    pylab.hold(False)


############  reconstruction scatter  ############

reconstructed_curve_data = my_cae.reconstruct(curve_data)

if False:
    p = pylab.subplot(111)
    pylab.hold(True)
    pylab.scatter(curve_data[:,0], curve_data[:,1])
    pylab.axis([0., 1., 0., 1.])
    pylab.scatter(reconstructed_curve_data[:,0], reconstructed_curve_data[:,1], c='r')
    pylab.show()
    pylab.hold(False)

###################################################

p = pylab.subplot(111)
pylab.hold(True)
pylab.axis([0., 1., 0., 1.])
pylab.quiver(X, Y, U, V, scale=1.0, color='#f59a43', linewidths=(1,), edgecolors=('#6e451e'))
pylab.scatter(curve_data[:,0], curve_data[:,1])
pylab.scatter(reconstructed_curve_data[:,0], reconstructed_curve_data[:,1], c='r')
pylab.show()
pylab.hold(False)










