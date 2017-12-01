import numpy
import scipy.interpolate
from ..constants.astronomy import arcsec
from .libinterferometry import Visibilities
from galario import double

def interpolate_model(u, v, freq, model, nthreads=1, dRA=0., dDec=0.):

    double.threads(nthreads)

    real = []
    imag = []

    dxy = (model.x[1] - model.x[0])*arcsec

    for i in range(len(model.freq)):
        vis = double.sampleImage(model.image[:,:,i,0], dxy, u, v, \
                dRA=dRA*arcsec, dDec=dDec*arcsec)

        real.append(vis.real)
        imag.append(vis.imag)

    real = numpy.concatenate(real, axis=1)
    imag = numpy.concatenate(imag, axis=1)

    return Visibilities(u, v, freq, real, imag, numpy.ones(real.shape))
