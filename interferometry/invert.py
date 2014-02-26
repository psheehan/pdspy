import numpy
from .average import average
from .grid import grid
from .clean import clean
from ..imaging import Image
from ..constants.physics import c
from ..constants.math import pi
from ..constants.astronomy import arcsec
from scipy.fftpack import ifft2, fftshift, ifftshift, fftfreq
from scipy.special import jn

def invert(data, imsize=256, pixel_size=0.25, convolution="pillbox"):
    
    binsize = 1.0 / (pixel_size * imsize * arcsec)
    print(binsize)
    gridded_data = grid(data, gridsize=imsize, binsize=binsize, \
            convolution=convolution)
    print(gridded_data.uvdist.max())
            
    real = gridded_data.real.reshape((imsize, imsize))
    imag = gridded_data.imag.reshape((imsize, imsize))
    weights = gridded_data.weights.reshape((imsize, imsize)).T

    comp = real + 1j*imag
    
    x = fftshift(fftfreq(imsize, binsize)) / arcsec
    y = fftshift(fftfreq(imsize, binsize)) / arcsec
    xx, yy = numpy.meshgrid(x, y)
    r = numpy.sqrt(xx**2 + yy**2)
    
    im = fftshift(ifft2(ifftshift(comp))).real

    image = im.reshape((imsize, imsize, 1))

    return Image(image, x=x, y=y)

def pillbox(u,v,delta_u,delta_v):
    
    arr = zeros(u.shape)
    arr[u.shape[0]/2.,u.shape[1]/2.] = 1.0
    
    return arr

def exp_sinc(u,v,delta_u,delta_v):
    
    alpha1 = 1.55
    alpha2 = 2.52
    
    return sinc(u/(alpha1*delta_u))*exp(-1*(u/(alpha2*delta_u))**2)* \
           sinc(v/(alpha1*delta_u))*exp(-1*(v/(alpha2*delta_v))**2)

def spheroidal(u,v,delta_u,delta_v):
    
    alpha = 0.0
    m = 6.0
    
    return abs(1-(2*u/(m*delta_u))**2)**alpha
