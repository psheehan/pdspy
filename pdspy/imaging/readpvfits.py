from ..constants.astronomy import arcsec
from ..constants.physics import c
from .libimaging import Image
from astropy.utils.exceptions import AstropyWarning
import astropy.io.fits as fits
import astropy.wcs as wcs
import warnings
import numpy

def readpvfits(filename):
    
    # Open the fits file.

    data = fits.open(filename)

    # Figure out the dimensions of each axis and create an array to put the data
    # into, and put the data into that array.

    npol, nv, nx = data[0].data.shape
    
    image = numpy.zeros((1,nx,nv,npol))

    for i in range(npol):
        for j in range(nv):
            image[0,:,j,i] = data[0].data[i,j,:]

    # Read in the x and y coordinate information, including the WCS info if it
    # is available.

    header = data[0].header

    #x, y = numpy.meshgrid(numpy.linspace(0,nx-1,nx), numpy.linspace(0,ny-1,ny))
    x, y = None, None

    x0 = data[0].header["CRVAL1"]
    dx = data[0].header["CDELT1"]
    nx0 = data[0].header["CRPIX1"]

    x = (numpy.arange(nx) - (nx0-1))*dx + x0

    nu0 = data[0].header["CRVAL2"]
    dnu = data[0].header["CDELT2"]
    n0 = data[0].header["CRPIX2"]
    freq = (numpy.arange(nv)-(n0-1))*dnu + nu0
    restfreq = data[0].header["RESTFRQ"]

    velocity = c * (restfreq - freq) / restfreq

    data.close()
 
    return Image(image, x=x, y=y, header=header, wcs=None, velocity=velocity, \
            freq=freq)
