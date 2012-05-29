#pylab.subplot(111)
#pylab.scatter(grid_points[:,0], grid_points[:,1], c='#6e451e')
pylab.subplot(111)
pylab.scatter(mu[:,0], mu[:,1], c='#f210a5')

# around every value of mu, we print the covariance ellipse
for i in numpy.arange(0,grid_points.shape[0]):
    (eigvals, eigvecs) = numpy.linalg.eig(C[i,:,:])
    #(eigvals, eigvecs) = numpy.linalg.eig(0.5*C[i,:,:] + 0.5*C[i,:,:].T)
    for k in (0,1):
        v = eigvecs[:,k]
        s = 0.1*numpy.sqrt(eigvals[k])
        pylab.subplot(111)
        pylab.axis('equal')
        pylab.plot((mu[i,0],mu[i,0]+s*v[0]), (mu[i,1], mu[i,1]+s*v[1]), c='#0ac263')

pylab.show()




def dev_mu_C_for_quadratic_loss(X):
    (n,d) = X.shape
    results_mu = numpy.zeros((n,d))
    results_cov = numpy.zeros((n,d,d))
    for i in numpy.arange(0,n):
        r0 = my_cae.reconstruct(X[i,:])
        h0 = my_cae.encode(X[i,:])
        A = numpy.diag(r0*(1-r0))
        B = numpy.diag(h0*(1-h0))
        Z = numpy.linalg.inv(numpy.eye(len(r0)) - numpy.dot(numpy.dot(numpy.dot(my_cae.W,B),my_cae.W.T),A))
        # mu = X[i,:] + numpy.dot(Z,r0-X[i,:])
		mu = X[i,:] + numpy.dot(r0-X[i,:],Z)
        # C = numpy.dot(numpy.dot(numpy.dot(numpy.dot(numpy.linalg.pinv(my_cae.W.T), B), my_cae.W.T), numpy.linalg.inv(A)), Z)
        C = numpy.dot(Z, numpy.dot(numpy.dot(numpy.dot(numpy.linalg.pinv(my_cae.W.T), B), my_cae.W.T), numpy.linalg.inv(A)))
        results_mu[i,:] = mu
        results_cov[i,:,:] = C
    return (results_mu, results_cov)

grid_points = numpy.array([(x,y) for x in numpy.arange(0, 1.00001, 0.2) for y in numpy.arange(0, 1.00001, 0.2)])
(mu, C) = dev_mu_C_for_quadratic_loss(grid_points)
reconstructed_grid_points = my_cae.reconstruct(grid_points)

pylab.subplot(111)
pylab.scatter(grid_points[:,0], grid_points[:,1], c='#6e451e')
pylab.subplot(111)
pylab.scatter(reconstructed_grid_points[:,0], reconstructed_grid_points[:,1], c='#db893c')
pylab.subplot(111)
pylab.scatter(mu[:,0], mu[:,1], c='#f210a5')
pylab.axis('equal')
pylab.show()


pylab.subplot(111)
pylab.axis('equal')
for i in numpy.arange(0,grid_points.shape[0]):
    (eigvals, eigvecs) = numpy.linalg.eig(C[i,:,:])
    #(eigvals, eigvecs) = numpy.linalg.eig(0.5*C[i,:,:] + 0.5*C[i,:,:].T)
    for k in (0,1):
        v = eigvecs[:,k]
        s = numpy.sqrt(eigvals[k])
        pylab.plot((mu[i,0],mu[i,0]+s*v[0]), (mu[i,1], mu[i,1]+s*v[1]), c='#0ac263')

pylab.show()





def mu_C_for_quadratic_loss2(self, X, return_rJ=False):
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
        #
        if return_rJ:
            results_mu[i,:] = r0
            results_cov[i,:,:] = J
        else:
            Z = numpy.linalg.inv(numpy.eye(len(r0)) - J)
            mu = X[i,:] + numpy.dot(Z,r0-X[i,:])
            C = numpy.dot(J, Z)
            results_mu[i,:] = mu
            results_cov[i,:,:] = C
    return (results_mu, results_cov)

cae.DAE.mu_C_for_quadratic_loss = mu_C_for_quadratic_loss2






pylab.subplot(111)
#pylab.axis([0., 1., 0., 1.])
pylab.scatter(grid_points[:,0], grid_points[:,1], c='#6e451e')
pylab.subplot(111)
#pylab.scatter(reconstructed_grid_points[:,0], reconstructed_grid_points[:,1], c='#f210a5')
pylab.scatter(mu[:,0], mu[:,1], c='#f210a5')

# around every value of mu, we print the covariance ellipse
for i in numpy.arange(0,grid_points.shape[0]):
    (eigvals, eigvecs) = numpy.linalg.eig(C[i,:,:])
    #(eigvals, eigvecs) = numpy.linalg.eig(0.5*C[i,:,:] + 0.5*C[i,:,:].T)
    #assert((eigvecs.dot(eigvecs.T)==numpy.eye(2)).all())
    print "valeurs propres : (%f, %f)" % (eigvals[0], eigvals[1])
    for k in (0,1):
        v = eigvecs[:,k]
        s = 0.1*numpy.sqrt(eigvals[k])
        pylab.subplot(111)
        pylab.axis('equal')
        if k==0:
            pylab.plot((mu[i,0],mu[i,0]+s*v[0]), (mu[i,1], mu[i,1]+s*v[1]), c='#0ac263')
        else:
            pylab.plot((mu[i,0],mu[i,0]+s*v[0]), (mu[i,1], mu[i,1]+s*v[1]), c='#0a7bc2')