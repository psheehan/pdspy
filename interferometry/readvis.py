from .readuvfits import readuvfits
from .interferometry import Visibilities
from glob import glob
from numpy import array, concatenate, argsort

def readvis(filename):

    filenames = array(glob(filename+"/*.uv.fits"))

    for i in range(filenames.size):
        if i == 0:
            vis = readuvfits(filenames[i],fmt="casa")

            vis.u /= vis.freq.mean()
            vis.v /= vis.freq.mean()
        else:
            new = readuvfits(filenames[i],fmt="casa")

            vis.freq = concatenate((vis.freq,new.freq))
            vis.real = concatenate((vis.real,new.real),axis=1)
            vis.imag = concatenate((vis.imag,new.imag),axis=1)
            vis.amp = concatenate((vis.amp,new.amp),axis=1)
            vis.weights = concatenate((vis.weights,new.weights),axis=1)

    vis.real = vis.real[:,argsort(vis.freq)][:,::-1]
    vis.imag = vis.imag[:,argsort(vis.freq)][:,::-1]
    vis.amp = vis.amp[:,argsort(vis.freq)][:,::-1]
    vis.weights = vis.weights[:,argsort(vis.freq)][:,::-1]
    vis.freq = vis.freq[argsort(vis.freq)][::-1]

    vis.u *= vis.freq.mean()
    vis.v *= vis.freq.mean()

    return Visibilities(vis.u, vis.v, vis.freq, vis.real, vis.imag, \
            vis.weights, baseline=vis.baseline)
