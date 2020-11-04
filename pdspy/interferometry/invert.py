import numpy
from . import center
from .libinterferometry import grid
from ..imaging import Image
from ..constants.astronomy import arcsec
from scipy.fftpack import ifft2, fftshift, ifftshift, fftfreq

def invert(data, imsize=256, pixel_size=0.25, convolution="pillbox", mfs=False,\
        weighting="natural", robust=2, npixels=0, centering=None, \
        mode='continuum', beam=False, uvtaper=None):

    # If we are calculating the beam, set all of the real values to 1 and the
    # imaginary data to 0.

    if beam:
        real = data.real.copy()
        imag = data.imag.copy()

        data.real[:,:] = 1.
        data.imag[:,:] = 0.

    # Apply a uv-taper to the data, if requested.

    if type(uvtaper) != type(None):
        taper = numpy.exp(-0.5*data.uvdist**2/(uvtaper*1e3)**2)

        weights = data.weights.copy()
        for i in range(data.freq.size):
            data.weights[:,i] *= taper

    # Grid the data before imaging.
    
    binsize = 1.0 / (pixel_size * imsize * arcsec)
    gridded_data = grid(data, gridsize=imsize, binsize=binsize, \
            convolution=convolution, mfs=mfs, imaging=True, \
            weighting=weighting, robust=robust, npixels=npixels, mode=mode)

    # If making an image of the beam, restore the data to what it was before.

    if beam:
        data.real = real
        data.imag = imag

    # If using uvtaper, restore the weights.

    if type(uvtaper) != type(None):
        data.weights = weights

    # Center the data, if requested.

    if type(centering) != type(None):
        gridded_data = center(gridded_data, centering)

    # Set up information for the final image.
            
    x = fftshift(fftfreq(imsize, binsize)) / arcsec
    y = fftshift(fftfreq(imsize, binsize)) / arcsec
    xx, yy = numpy.meshgrid(x, y)
    r = numpy.sqrt(xx**2 + yy**2)

    image = numpy.zeros((imsize,imsize,gridded_data.real.shape[1],1))

    # Loop through frequency space and do the Fourier transform.

    for i in range(gridded_data.real.shape[1]):
        u = gridded_data.u.reshape((imsize, imsize))
        v = gridded_data.v.reshape((imsize, imsize))
        real = gridded_data.real[:,i].reshape((imsize, imsize))
        imag = gridded_data.imag[:,i].reshape((imsize, imsize))
        weights = gridded_data.weights[:,i].reshape((imsize, imsize))

        comp = real + 1j*imag
        
        if convolution == "pillbox":
            conv_func = pillbox
        elif convolution == "expsinc":
            conv_func = exp_sinc

        im = fftshift(ifft2(ifftshift(comp))).real * imsize**2

        convolve = fftshift(ifft2(ifftshift(conv_func(u, v, binsize, \
               binsize)))).real

        image[:,:,i,0] = (im/convolve)[:,::-1]

        # Make sure the beam peaks at exactly 1. Should be just a small 
        # correction.

        if beam:
            image[:,:,i,0] /= image[:,:,i,0].max()

    return Image(image, x=x, y=y, freq=gridded_data.freq)

def pillbox(u, v, delta_u, delta_v):

    m = 1
    
    arr = numpy.ones(u.shape, dtype=float)*u.size

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

    arr = arr/arr.sum() * arr.size

    return arr
