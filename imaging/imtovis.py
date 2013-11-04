import numpy
from ..interferometry.interferometry import Visibilities
from ..constants.astronomy import pc, arcsec
from scipy.fftpack import fft2,fftshift

def imtovis(image, dpc=140):

    ##### Some natural constants
    
    if dpc == None:
        r = 140*pc
    else:
        r = dpc*pc
    
    real = numpy.empty((image.x.size*image.y.size,image.freq.size))
    imag = numpy.empty((image.x.size*image.y.size,image.freq.size))
    weights = numpy.ones(real.shape)
    for i in range(image.freq.size):
        vis = fftshift(fft2(fftshift(image.image[:,:,i])))
        real[:,i] = vis.real.reshape((image.x.size*image.y.size,))
        imag[:,i] = vis.imag.reshape((image.x.size*image.y.size,))
    
    max_x = 1.0 / ( (image.x[1] - image.x[0])/r *arcsec )
    max_y = 1.0 / ( (image.y[1] - image.y[0])/r *arcsec )
    
    u, v = numpy.meshgrid( numpy.linspace(-max_x, max_x, image.x.size), \
            numpy.linspace(-max_y, max_y, image.y.size), indexing='ij')
    u = u.reshape((image.x.size*image.y.size,))
    v = v.reshape((image.y.size*image.y.size,))

    freq = image.freq

    return Visibilities(u,v,freq,real,imag,weights)
