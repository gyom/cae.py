import numpy
import cae

N = 100
# the slope has to be between [0.0, 1.0] because the outputs are sigmoids
slope = 1.0
noise = 0.1
domain = 0.2 + 0.6*numpy.array(sorted(numpy.random.random((N,))))
image = numpy.array([slope*t for t in domain]) + noise*numpy.random.random((N,))
line_data = numpy.vstack((domain, image)).T

my_cae = cae.CAE(n_hiddens=256,
    W=None,
    c=None,
    b=None,
    learning_rate=0.001,
    jacobi_penalty=0.1,
    batch_size=10,
    epochs=10000,
    use_cae_jacobian=False)

my_cae.fit(line_data, verbose=True)


############ quiver plot for mu ##############

import pylab

grid_points = numpy.array([(x,y) for x in numpy.arange(0, 1.00001, 0.2) for y in numpy.arange(0, 1.00001, 0.2)])
#grid_points = numpy.array([(x,y) for x in numpy.arange(0.4, 0.61, 0.01) for y in numpy.arange(0.4, 1.00001, 0.01)])
(mu, C) = my_cae.mu_C_for_quadratic_loss(grid_points)

reconstructed_grid_points = my_cae.reconstruct(grid_points)
reconstructed_line_data = my_cae.reconstruct(line_data)

pylab.subplot(121)
pylab.axis([0., 1., 0., 1.])
pylab.scatter(line_data[:,0], line_data[:,1], c='#1d5a8e')
pylab.subplot(121)
pylab.scatter(reconstructed_line_data[:,0], reconstructed_line_data[:,1], c='#329cf4')

pylab.subplot(122)
#pylab.axis([0., 1., 0., 1.])
pylab.scatter(grid_points[:,0], grid_points[:,1], c='#6e451e')
pylab.subplot(122)
#pylab.scatter(reconstructed_grid_points[:,0], reconstructed_grid_points[:,1], c='#f210a5')
pylab.scatter(mu[:,0], mu[:,1], c='#f210a5')

for i in numpy.arange(0,grid_points.shape[0]):
    pylab.subplot(122)
    pylab.plot((grid_points[i,0],mu[i,0]), (grid_points[i,1],mu[i,1]), c='#000000')

# around every value of mu, we print the covariance ellipse
for i in numpy.arange(0,grid_points.shape[0]):
    (eigvals, eigvecs) = numpy.linalg.eig(C[i,:,:])
    #(eigvals, eigvecs) = numpy.linalg.eig(0.5*C[i,:,:] + 0.5*C[i,:,:].T)
    #assert((eigvecs.dot(eigvecs.T)==numpy.eye(2)).all())
    print "valeurs propres : (%f, %f)" % (eigvals[0], eigvals[1])
    for k in (0,1):
        v = eigvecs[:,k]
        s = 0.1*numpy.sqrt(eigvals[k])
        pylab.subplot(122)
        pylab.axis('equal')
        if k==0:
            pylab.plot((mu[i,0],mu[i,0]+s*v[0]), (mu[i,1], mu[i,1]+s*v[1]), c='#0ac263')
        else:
            pylab.plot((mu[i,0],mu[i,0]+s*v[0]), (mu[i,1], mu[i,1]+s*v[1]), c='#0a7bc2')
    
#X = grid_points[:,0]
#Y = grid_points[:,1]
#U = mu[:,0] -  grid_points[:,0]
#V = mu[:,1] -  grid_points[:,1]
#pylab.quiver(X, Y, U, V, units='x', scale=1.0)

pylab.show()




#reconstruction_data = my_cae.reconstruct(line_data)

import pylab
#pylab.plot(line_data[:,0], line_data[:,1])
#pylab.show()

#pylab.plot(reconstruction_data[:,0], reconstruction_data[:,1])
#pylab.show()




################  quiver plot  ###############
grid_points = numpy.array([(x,y) for x in numpy.arange(0, 1.00001, 0.05) for y in numpy.arange(0, 1.00001, 0.05)])
reconstructed_grid_points = my_cae.reconstruct(grid_points)

X = grid_points[:,0]
Y = grid_points[:,1]
U = reconstructed_grid_points[:,0] -  grid_points[:,0]
V = reconstructed_grid_points[:,1] -  grid_points[:,1]
pylab.quiver(X, Y, U, V)
pylab.axis([0., 1., 0., 1.])
pylab.hold(True)
pylab.plot(line_data[:,0], line_data[:,1])
pylab.show()


################  quiver + scatter ####################

grid_points = numpy.array([(x,y) for x in numpy.arange(0, 1.00001, 0.05) for y in numpy.arange(0, 1.00001, 0.05)])
reconstructed_grid_points = my_cae.reconstruct(grid_points)

X = grid_points[:,0]
Y = grid_points[:,1]
U = reconstructed_grid_points[:,0] -  grid_points[:,0]
V = reconstructed_grid_points[:,1] -  grid_points[:,1]

reconstructed_line_data = my_cae.reconstruct(line_data)

p = pylab.subplot(111)
pylab.hold(True)
pylab.axis([0., 1., 0., 1.])
#pylab.quiver(X, Y, U, V, scale=1.0, color='#f59a43', linewidths=(1,), edgecolors=('#6e451e'))
pylab.scatter(line_data[:,0], line_data[:,1])
pylab.scatter(reconstructed_line_data[:,0], reconstructed_line_data[:,1], c='r')
pylab.show()
pylab.hold(False)


#################  scatter + mu  ####################





