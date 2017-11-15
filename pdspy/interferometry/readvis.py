from .readuvfits import readuvfits
from .libinterferometry import Visibilities
from glob import glob
import numpy

def readvis(filename, fmt="casa"):

    filenames = numpy.array(glob(filename+"/*.uv.fits"))

    for i in range(filenames.size):
        if i == 0:
            vis = readuvfits(filenames[i],fmt=fmt)

            vis.u /= vis.freq.mean()
            vis.v /= vis.freq.mean()
        else:
            new = readuvfits(filenames[i],fmt=fmt)

            vis.freq = numpy.concatenate((vis.freq,new.freq))
            vis.real = numpy.concatenate((vis.real,new.real),axis=1)
            vis.imag = numpy.concatenate((vis.imag,new.imag),axis=1)
            vis.amp = numpy.concatenate((vis.amp,new.amp),axis=1)
            vis.weights = numpy.concatenate((vis.weights,new.weights),axis=1)

    vis.real = vis.real[:,numpy.argsort(vis.freq)][:,::-1]
    vis.imag = vis.imag[:,numpy.argsort(vis.freq)][:,::-1]
    vis.amp = vis.amp[:,numpy.argsort(vis.freq)][:,::-1]
    vis.weights = vis.weights[:,numpy.argsort(vis.freq)][:,::-1]
    vis.freq = vis.freq[numpy.argsort(vis.freq)][::-1]

    vis.u *= vis.freq.mean()
    vis.v *= vis.freq.mean()

    return Visibilities(vis.u, vis.v, vis.freq, vis.real, vis.imag, \
            vis.weights, baseline=vis.baseline)
