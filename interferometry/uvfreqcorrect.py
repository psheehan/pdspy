from numpy import array, concatenate
from .interferometry import Visibilities
from ..constants.physics import c

def uvfreqcorrect(data):

    new_freq = array([data.freq.mean()])

    new_u = array([])
    new_v = array([])
    new_real = array([])
    new_imag = array([])
    new_weights = array([])
    new_baseline = array([])

    for i in range(data.freq.size):
        new_u = concatenate((new_u,data.u*data.freq[i]/new_freq[0]))
        new_v = concatenate((new_v,data.v*data.freq[i]/new_freq[0]))
        new_real = concatenate((new_real,data.real[:,i]))
        new_imag = concatenate((new_imag,data.imag[:,i]))
        new_weights = concatenate((new_weights,data.weights[:,i]))
        new_baseline = concatenate((new_baseline,data.baseline))

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
    new_baseline = new_baseline[good,:]

    return Visibilities(new_u,new_v,new_freq,new_real,new_imag,new_weights, \
            baseline=new_baseline)
