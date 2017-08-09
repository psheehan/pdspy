import numpy
import scipy.interpolate
from .libinterferometry import Visibilities

def interpolate_model(u, v, freq, model):

    uuvv = numpy.transpose(numpy.vstack((model.u, model.v)))

    real = []
    imag = []

    for i in range(len(model.freq)):
        real.append(scipy.interpolate.griddata(uuvv, model.real[:,i], \
                (u*freq[i]/freq.mean(), v*freq[i]/freq.mean()), method='cubic'))
        imag.append(scipy.interpolate.griddata(uuvv, model.imag[:,i], \
                (u*freq[i]/freq.mean(), v*freq[i]/freq.mean()), method='cubic'))

    real = numpy.transpose(numpy.array(real))
    imag = numpy.transpose(numpy.array(imag))

    return Visibilities(u, v, freq, real, imag, numpy.ones(real.shape))
