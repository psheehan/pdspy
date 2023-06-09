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
        code="galario", nxy=1024, dxy=0.01):

    if code == "galario":
        real = []
        imag = []

        double.threads(nthreads)

        dxy = (model.x[1] - model.x[0])*arcsec

        for i in range(len(model.freq)):
            vis = double.sampleImage(model.image[::-1,:,i,0].copy(order='C'), \
                    dxy, u, v, dRA=dRA*arcsec, dDec=dDec*arcsec)

            real.append(vis.real.reshape((u.size,1)))
            imag.append(-vis.imag.reshape((u.size,1)))

        real = numpy.concatenate(real, axis=1)
        imag = numpy.concatenate(imag, axis=1)

    elif code == "galario-unstructured":
        real = []
        imag = []

        double.threads(nthreads)

        vol = None
        for i in range(len(model.freq)):
            vis = double.sampleUnstructuredImage(model.x*arcsec, \
                    -model.y*arcsec, model.image[:,i].copy(order='C'), nxy, \
                    dxy*arcsec, u, v, dRA=dRA*arcsec, dDec=dDec*arcsec)

            real.append(vis.real.reshape((u.size,1)))
            imag.append(-vis.imag.reshape((u.size,1)))

        real = numpy.concatenate(real, axis=1)
        imag = numpy.concatenate(imag, axis=1)

    elif code == "trift":
        vis = trift.cpu.trift(model.x*arcsec, model.y*arcsec, \
                model.image, u, v, dRA*arcsec, dDec*arcsec, \
                nthreads=nthreads, mode="extended")

        real, imag = vis.real, vis.imag

    return Visibilities(u, v, freq, real, imag, numpy.ones(real.shape))
