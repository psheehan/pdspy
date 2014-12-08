from .libimaging import Image
import astropy.io.fits as fits
import astropy.wcs as wcs
import numpy

def readimfits(filename):
    
    # Open the fits file.

    data = fits.open(filename)

    # Figure out the dimensions of each axis and create an array to put the data
    # into, and put the data into that array.

    npol, nfreq, ny, nx = data[0].data.shape
    
    image = numpy.zeros((ny,nx,nfreq,npol))

    for i in range(npol):
        for j in range(nfreq):
            image[:,:,j,i] = data[0].data[i,j,:,:].reshape(ny,nx)

    # Read in the x and y coordinate information, including the WCS info if it
    # is available.

    header = data[0].header

    w = wcs.WCS(header)

    #x, y = numpy.meshgrid(numpy.linspace(0,nx-1,nx), numpy.linspace(0,ny-1,ny))
    x, y = None, None

    if header["CTYPE3"] == "VELOCITY":
        v0 = data[0].header["CRVAL3"]
        dv = data[0].header["CDELT3"]
        n0 = data[0].header["CRPIX3"]
        velocity = (numpy.arange(nfreq)-(n0-1))*dv/1000.+v0/1000.

        freq = None
    elif header["CTYPE3"] == "FREQ":
        nu0 = data[0].header["CRVAL3"]
        dnu = data[0].header["CDELT3"]
        n0 = data[0].header["CRPIX3"]
        freq = (numpy.arange(nfreq)-(n0-1))*dnu + nu0

        velocity = None

    data.close()
 
    return Image(image, x=x, y=y, header=header, wcs=w, velocity=velocity, \
            freq=freq)
