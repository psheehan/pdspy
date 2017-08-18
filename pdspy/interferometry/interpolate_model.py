import numpy
import scipy.interpolate
from .libinterferometry import Visibilities

def interpolate_model(u, v, freq, model):

    nu = int(model.u.size**0.5)
    model_u = model.u[0:nu]
    model_v = model.v[0:-1:nu]

    real = []
    imag = []

    for i in range(len(model.freq)):
        f = scipy.interpolate.RectBivariateSpline(model_u, model_v, \
                model.real[:,i].reshape((nu,nu)), kx=5, ky=5)
        real.append(f.ev(u*freq[i]/freq.mean(), v*freq[i]/freq.mean()))
        f = scipy.interpolate.RectBivariateSpline(model_u, model_v, \
                model.imag[:,i].reshape((nu,nu)), kx=5, ky=5)
        imag.append(f.ev(u*freq[i]/freq.mean(), v*freq[i]/freq.mean()))

    real = numpy.transpose(numpy.array(real))
    imag = numpy.transpose(numpy.array(imag))

    return Visibilities(u, v, freq, real, imag, numpy.ones(real.shape))
