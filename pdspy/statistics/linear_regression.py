import numpy

def linear_regression(_x, _y, sigma_y=None):

    x = numpy.mat(_x).T
    y = numpy.mat(_y).T

    A = numpy.concatenate((x, numpy.ones(x.shape)), axis=1)

    if type(sigma_y) != type(None):
        C = numpy.diag(sigma_y**2)
    else:
        C = numpy.diag(numpy.ones(_y.shape))

    Xa = numpy.linalg.inv(A.T * numpy.linalg.inv(C) * A) * \
            (A.T * numpy.linalg.inv(C) * y)

    Cov = numpy.linalg.inv(A.T * numpy.linalg.inv(C) * A)

    return Xa[0,0], Xa[1,0], Cov[0,0]**0.5, Cov[1,1]**0.5
