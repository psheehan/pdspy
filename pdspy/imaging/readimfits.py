from ..constants.physics import c
from ..constants.astronomy import arcsec
from .libimaging import Image
from astropy.utils.exceptions import AstropyWarning
import astropy.io.fits as fits
import astropy.wcs as wcs
import warnings
import numpy

def readimfits(filename):
    
    # Open the fits file.

    data = fits.open(filename)

    # Figure out the dimensions of each axis and create an array to put the data
    # into, and put the data into that array.

    if len(data[0].data.shape) == 4:
        npol, nfreq, ny, nx = data[0].data.shape
    elif len(data[0].data.shape) == 3:
        nfreq, ny, nx = data[0].data.shape
        npol = 1
    elif len(data[0].data.shape) == 2:
        ny, nx = data[0].data.shape
        npol, nfreq = 1, 1
    
    image = numpy.zeros((ny,nx,nfreq,npol))

    for i in range(npol):
        for j in range(nfreq):
            if len(data[0].data.shape) == 4:
                image[:,:,j,i] = data[0].data[i,j,:,:].reshape(ny,nx)
            if len(data[0].data.shape) == 3:
                image[:,:,j,i] = data[0].data[j,:,:].reshape(ny,nx)
            if len(data[0].data.shape) == 2:
                image[:,:,j,i] = data[0].data[:,:].reshape(ny,nx)

    # Read in the x and y coordinate information, including the WCS info if it
    # is available.

    header = data[0].header

    # Check whether there is a CASA beam table.

    if len(data) > 1:
        if data[1].columns[0].name == 'BMAJ':
            header["BMAJ"] = data[1].data["BMAJ"].mean()*arcsec * 180./numpy.pi
            header["BMIN"] = data[1].data["BMIN"].mean()*arcsec * 180./numpy.pi
            header["BPA"] = data[1].data["BPA"].mean()

    # Turn off the WCS warnings that come from the PCX_Y values because they
    # are annoying...
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=AstropyWarning)

        w = wcs.WCS(header)
        try:
            w = w.dropaxis(2)
        except:
            pass
        try:
            w = w.dropaxis(3)
        except:
            pass

    #x, y = numpy.meshgrid(numpy.linspace(0,nx-1,nx), numpy.linspace(0,ny-1,ny))
    x, y = None, None

    if len(data[0].data.shape) >= 3:
        if header["CTYPE3"] in ["VELOCITY","VRAD"]:
            v0 = data[0].header["CRVAL3"]
            dv = data[0].header["CDELT3"]
            n0 = data[0].header["CRPIX3"]
            velocity = (numpy.arange(nfreq)-(n0-1))*dv/1000.+v0/1000.

            nu0 = data[0].header["RESTFREQ"]
            freq = nu0 - velocity*1e5 * nu0 / c
        elif header["CTYPE3"] == "FREQ":
            nu0 = data[0].header["CRVAL3"]
            dnu = data[0].header["CDELT3"]
            n0 = data[0].header["CRPIX3"]
            freq = (numpy.arange(nfreq)-(n0-1))*dnu + nu0

            velocity = None
    else:
        velocity, freq = None, None

    data.close()
 
    return Image(image, x=x, y=y, header=header, wcs=w, velocity=velocity, \
            freq=freq)
