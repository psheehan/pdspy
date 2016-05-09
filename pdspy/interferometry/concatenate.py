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
            weights = vis.weights.copy()
            freq = vis.freq.copy()
            if type(vis.baseline) != type(None):
                baseline = vis.baseline.copy()
                incl_baselines = True
            else:
                incl_baselines = False
        else:
            u = numpy.concatenate((u, vis.u.copy()))
            v = numpy.concatenate((v, vis.v.copy()))
            real = numpy.concatenate((real, vis.real.copy()))
            imag = numpy.concatenate((imag, vis.imag.copy()))
            amp = numpy.concatenate((amp, vis.amp.copy()))
            weights = numpy.concatenate((weights, vis.weights.copy()))
            if incl_baselines:
                if type(vis.baseline) != type(None):
                    baseline = numpy.concatenate((baseline,vis.baseline.copy()))
                else:
                    incl_baselines = False

    if incl_baselines:
        return Visibilities(u, v, freq, real, imag, weights, baseline=baseline)
    else:
        return Visibilities(u, v, freq, real, imag, weights)
