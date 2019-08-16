from ..constants.astronomy import arcsec
from ..constants.physics import c
from . import Image
try:
    import pvextractor
except:
    pass
import numpy

def extract_pv_diagram(image, xy=(0.,0.), pa=0., length=100, width=1):

    ny, nx, nfreq, npol = image.image.shape

    # Create a data cube that is appropriately shaped.

    cube = numpy.empty((nfreq, ny, nx))
    for i in range(nfreq):
        cube[i,:,:] = image.image[:,:,i,0]

    # Create the Path object.

    x0, y0 = xy

    line = [(x0-length/2*numpy.sin(pa),y0-length/2*numpy.cos(pa)), \
            (x0+length/2*numpy.sin(pa), y0+length/2*numpy.cos(pa))]

    path = pvextractor.Path(line, width=width)

    # Extract the PV diagram along the Path

    pv = pvextractor.extract_pv_slice(cube, path)

    # Convert back to my Image class.

    pvdiagram = numpy.empty((1, pv.data.shape[1], pv.data.shape[0], 1))
    for i in range(pv.data.shape[0]):
        pvdiagram[0,:,i,0] = pv.data[i,:]

    # Get the velocity.

    velocity = c * (image.header["RESTFRQ"] - image.freq) / \
            image.header["RESTFRQ"]

    # Get the x coordinates.

    x0 =  0.
    dx =  image.header["CDELT2"] * numpy.pi/180 / arcsec
    nx0 = int(pv.data.shape[1] / 2) + 1

    x = (numpy.arange(pv.data.shape[1]) - (nx0-1))*dx + x0

    return Image(pvdiagram, x=x, velocity=velocity, freq=image.freq)
