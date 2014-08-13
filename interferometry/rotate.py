from .libinterferometry import Visibilities
import numpy

def rotate(data, pa=0):

    newu =  data.u * numpy.cos(pa) + data.v * numpy.sin(pa)
    newv = -data.u * numpy.sin(pa) + data.v * numpy.cos(pa)

    return Visibilities(newu, newv, data.freq.copy(), data.real.copy(), \
            data.imag.copy(), data.weights.copy())
