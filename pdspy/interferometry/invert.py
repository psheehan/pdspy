import numpy
from .libinterferometry import grid
from .clean import clean
from ..imaging import Image
from ..constants.astronomy import arcsec
from scipy.fftpack import ifft2, fftshift, ifftshift, fftfreq

def invert(data, imsize=256, pixel_size=0.25, convolution="pillbox", mfs=False):
    
    binsize = 1.0 / (pixel_size * imsize * arcsec)
    gridded_data = grid(data, gridsize=imsize, binsize=binsize, \
            convolution=convolution, mfs=mfs)
            
    u = gridded_data.u.reshape((imsize, imsize))
    v = gridded_data.v.reshape((imsize, imsize))
    freq = gridded_data.freq.copy()
    real = gridded_data.real.reshape((imsize, imsize))
    imag = gridded_data.imag.reshape((imsize, imsize))
    weights = gridded_data.weights.reshape((imsize, imsize))

    comp = real + 1j*imag
    
    x = fftshift(fftfreq(imsize, binsize)) / arcsec
    y = fftshift(fftfreq(imsize, binsize)) / arcsec
    xx, yy = numpy.meshgrid(x, y)
    r = numpy.sqrt(xx**2 + yy**2)

    if convolution == "pillbox":
        conv_func = pillbox
    elif convolution == "expsinc":
        conv_func = exp_sinc

    im = fftshift(ifft2(ifftshift(comp))).real
    convolve = fftshift(ifft2(ifftshift(conv_func(u, v, binsize, \
           binsize)))).real

    image = (im/convolve).reshape((imsize, imsize, 1))

    return Image(image, x=x, y=y, freq=freq)

def pillbox(u, v, delta_u, delta_v):

    m = 1
    
    arr = numpy.ones(u.shape, dtype=float)

    arr[numpy.abs(u) >= m * delta_u / 2] = 0
    arr[numpy.abs(v) >= m * delta_v / 2] = 0

    return arr

def exp_sinc(u, v, delta_u, delta_v):
    
    alpha1 = 1.55
    alpha2 = 2.52
    m = 6
    
    arr = numpy.sinc(u / (alpha1 * delta_u)) * \
            numpy.exp(-1 * (u / (alpha2 * delta_u))**2)* \
            numpy.sinc(v / (alpha1 * delta_v)) * \
            numpy.exp(-1 * (v / (alpha2 * delta_v))**2)

    arr[numpy.abs(u) >= m * delta_u / 2] = 0
    arr[numpy.abs(v) >= m * delta_v / 2] = 0

    return arr
