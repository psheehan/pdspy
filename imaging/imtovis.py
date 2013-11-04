import numpy
from ..interferometry.interferometry import Visibilities
from ..constants.astronomy import pc, arcsec
from scipy.fftpack import fft2,fftshift

def imtovis(image,dpc=140,ext=0):

    ##### Some natural constants
    
    im = image.image[:,:,ext]
    
    if dpc == None:
        r = 140*pc
    else:
        r = dpc*pc
    
    vis = fftshift(fft2(fftshift(im)))
    
    max_x = 1.0 / ( (image.x[1] - image.x[0])/r *arcsec )
    max_y = 1.0 / ( (image.y[1] - image.y[0])/r *arcsec )
    
    u, v = numpy.meshgrid( numpy.linspace(-max_x, max_x, image.x.size), \
            numpy.linspace(-max_y, max_y, image.y.size), indexing='ij')
    u = u.reshape((image.x.size**2,))
    v = v.reshape((image.y.size**2,))

    freq = numpy.array([image.freq[ext]])
    real = (vis.real).reshape((image.x.size**2,1))
    imag = (vis.imag).reshape((image.x.size**2,1))
    weights = numpy.ones(real.shape)
    
    return Visibilities(u,v,freq,real,imag,weights)
