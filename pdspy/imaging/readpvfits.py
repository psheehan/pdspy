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

    npol, nfreq, nx = data[0].data.shape
    
    image = numpy.zeros((nfreq,nx,1,npol))

    for i in range(npol):
        image[:,:,0,i] = data[0].data[i,:,:]

    # Read in the x and y coordinate information, including the WCS info if it
    # is available.

    header = data[0].header

    x0 = data[0].header["CRVAL1"]
    dx = data[0].header["CDELT1"]
    n0 = data[0].header["CRPIX1"]
    x = (numpy.arange(nx)-(n0-1))*dx

    nu0 = data[0].header["CRVAL2"]
    dnu = data[0].header["CDELT2"]
    n0 = data[0].header["CRPIX2"]
    freq = (numpy.arange(nfreq)-(n0-1))*dnu + nu0

    velocity = c * (data[0].header["RESTFRQ"] - freq) / \
            data[0].header["RESTFRQ"]

    data.close()
 
    return Image(image, x=x, y=None, header=header, wcs=None, \
            velocity=velocity, freq=None)
