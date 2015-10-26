import numpy
from ..interferometry import Visibilities
from ..constants.astronomy import pc, arcsec
from scipy.fftpack import fft2, fftshift, fftfreq, ifftshift

def imtovis(image):

    ##### Some natural constants
    
    real = numpy.empty((image.x.size*image.y.size,image.freq.size))
    imag = numpy.empty((image.x.size*image.y.size,image.freq.size))
    weights = numpy.ones(real.shape)
    for i in range(image.freq.size):
        vis = fftshift(fft2(ifftshift(image.image[:,:,i,0])))
        real[:,i] = vis.real.reshape((image.x.size*image.y.size,))
        imag[:,i] = vis.imag.reshape((image.x.size*image.y.size,))

    uu = fftshift(fftfreq(image.x.size, (image.x[1] - image.x[0]) * arcsec))
    vv = fftshift(fftfreq(image.y.size, (image.y[1] - image.y[0]) * arcsec))
    
    u, v = numpy.meshgrid(uu, vv)
    u = u.reshape((image.x.size*image.y.size,))
    v = v.reshape((image.y.size*image.y.size,))

    freq = image.freq

    return Visibilities(u, v, freq, real, imag, weights)
