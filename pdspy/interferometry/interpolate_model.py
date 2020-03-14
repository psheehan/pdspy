import numpy
import scipy.interpolate
from ..constants.astronomy import arcsec
from .libinterferometry import Visibilities
from galario import double
try:
    import trift
except:
    pass

def interpolate_model(u, v, freq, model, nthreads=1, dRA=0., dDec=0., \
        code="galario"):

    real = []
    imag = []

    if code == "galario":
        double.threads(nthreads)

        dxy = (model.x[1] - model.x[0])*arcsec

        for i in range(len(model.freq)):
            vis = double.sampleImage(model.image[:,:,i,0].copy(order='C'), \
                    dxy, u, v, dRA=dRA*arcsec, dDec=dDec*arcsec)

            real.append(vis.real.reshape((u.size,1)))
            imag.append(vis.imag.reshape((u.size,1)))
    elif code == "trift":
        vis = trift.trift_c(model.x*arcsec, model.y*arcsec, \
                model.image[:,0], u, v)

        real.append(vis.real.reshape((u.size,1)))
        imag.append(vis.imag.reshape((u.size,1)))

    real = numpy.concatenate(real, axis=1)
    imag = numpy.concatenate(imag, axis=1)

    return Visibilities(u, v, freq, real, imag, numpy.ones(real.shape))
