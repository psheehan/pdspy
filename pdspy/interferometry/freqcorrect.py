import numpy
from .libinterferometry import Visibilities
from ..constants.physics import c

def freqcorrect(data, new_freq=None):

    if new_freq != None:
        new_freq = numpy.array([new_freq])
    else:
        new_freq = numpy.array([data.freq.mean()])

    new_u = numpy.array([])
    new_v = numpy.array([])
    new_real = numpy.array([])
    new_imag = numpy.array([])
    new_weights = numpy.array([])
    new_baseline = numpy.array([])

    for i in range(data.freq.size):
        new_u = numpy.concatenate((new_u, data.u * data.freq[i]/new_freq[0]))
        new_v = numpy.concatenate((new_v, data.v * data.freq[i]/new_freq[0]))
        new_real = numpy.concatenate((new_real, data.real[:,i].copy()))
        new_imag = numpy.concatenate((new_imag, data.imag[:,i].copy()))
        new_weights = numpy.concatenate((new_weights, data.weights[:,i].copy()))
        new_baseline = numpy.concatenate((new_baseline, data.baseline.copy()))

    new_u = new_u.reshape((new_u.size,))
    new_v = new_v.reshape((new_v.size,))
    new_real = new_real.reshape((new_real.size,1))
    new_imag = new_imag.reshape((new_imag.size,1))
    new_weights = new_weights.reshape((new_weights.size,1))
    new_baseline = new_baseline.reshape((new_baseline.size,))

    good = new_weights[:,0] > 0.0
    new_u = new_u[good]
    new_v = new_v[good]
    new_real = new_real[good,:]
    new_imag = new_imag[good,:]
    new_weights = new_weights[good,:]
    new_baseline = new_baseline[good]

    return Visibilities(new_u, new_v, new_freq, new_real, new_imag, \
            new_weights, baseline=new_baseline)
