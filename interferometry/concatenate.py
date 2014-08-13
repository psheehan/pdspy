from .libinterferometry import Visibilities
import numpy

def concatenate(visibilities):

    for i, vis in enumerate(visibilities):
        if i == 0:
            u = vis.u.copy()
            v = vis.v.copy()
            real = vis.real.copy()
            imag = vis.imag.copy()
            amp = vis.amp.copy()
            weights = vis.amp.copy()
            freq = vis.freq.copy()
            baseline = vis.baseline.copy()
        else:
            u = numpy.concatenate((u, vis.u.copy()))
            v = numpy.concatenate((v, vis.v.copy()))
            real = numpy.concatenate((real, vis.real.copy()))
            imag = numpy.concatenate((imag, vis.imag.copy()))
            amp = numpy.concatenate((amp, vis.amp.copy()))
            weights = numpy.concatenate((weights, vis.weights.copy()))
            baseline = numpy.concatenate((baseline, vis.baseline.copy()))

    return Visibilities(u, v, freq, real, imag, weights, baseline=baseline)
