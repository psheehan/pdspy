from scipy.signal import fftconvolve
from scipy.optimize import leastsq
from ..imaging import Image
from .invert import invert
import numpy

def clean(data, imsize=256, pixel_size=0.25, convolution="pillbox", mfs=False,\
        weighting="natural", robust=2, npixels=0, centering=None, \
        mode='continuum', gain=0.1, maxiter=1000, threshold=0.001, \
        uvtaper=None):

    # First make the image.

    image = invert(data, imsize=imsize, pixel_size=pixel_size, \
            convolution=convolution, mfs=mfs, weighting=weighting, \
            robust=robust, npixels=npixels, centering=centering, mode=mode, \
            uvtaper=uvtaper)

    # Now, also make an image of the beam.

    beam = invert(data, imsize=2*imsize, pixel_size=pixel_size, \
            convolution=convolution, mfs=mfs, weighting=weighting, \
            robust=robust, npixels=npixels, centering=centering, mode=mode, \
            beam=True)

    # Now start the clean-ing by defining some variables.

    dirty = image.image[:,:,:,0]
    dirty_beam = beam.image[:,:,:,0]
    wherezero = dirty == 0
    nonzero = dirty != 0
    
    model = numpy.zeros(dirty.shape)

    # Calculate the size of the beam.

    clean_beam = numpy.zeros(dirty_beam.shape)

    ny, nx, nfreq = dirty_beam.shape
    x, y = numpy.meshgrid(numpy.arange(nx) - nx/2 + 1, numpy.arange(ny) - ny/2)

    for i in range(nfreq):
        fitfunc = lambda p, x, y: numpy.exp(-(x * numpy.cos(p[2]) - \
                y * numpy.sin(p[2]))**2 / (2*p[0]**2) - (x * numpy.sin(p[2]) + \
                y * numpy.cos(p[2]))**2 / (2*p[1]**2))
        errfunc = lambda p, x, y, z, w: numpy.ravel((fitfunc(p, x, y) - z) * w)
        p0 = [0.5,0.5,0.]

        weights = numpy.abs(dirty_beam[:,:,i])*(dirty_beam[:,:,i] > 0.4)

        p, success = leastsq(errfunc, p0, args=(x,y,dirty_beam[:,:,i],weights))

        clean_beam[:,:,i] = fitfunc(p, x, y)
    
    # Generate a mask.

    mask = numpy.ones(dirty.shape)
    """
    TODO: Add in auto-masking.
    """

    # Now loop through and subtract off the beam.

    n = 0
    stop = False
    while n < maxiter and not stop:
        # Determine the location of the maximum value inside the mask.

        maxval = dirty*mask == (dirty*mask).max()

        # Add that value to the model (with some gain).
        
        model[maxval] = model[maxval] + dirty[maxval]*gain

        # Also subtract off that value from the image.
        
        subtract = numpy.zeros(dirty.shape)
        subtract[maxval] = dirty[maxval]*gain
        
        for i in range(nfreq):
            dirty[:,:,i] = dirty[:,:,i] - fftconvolve(subtract[:,:,i], \
                    dirty_beam[:,:,i], mode='same')
        dirty[wherezero] = 0.
        
        # Do we stop here?

        stop = (dirty*mask).max() < threshold

        n = n + 1

    # Generate a clean beam and convolve with the model to make a cleaned image.

    clean_image = numpy.zeros(dirty.shape)
    for i in range(nfreq):
        clean_image[:,:,i] = fftconvolve(model[:,:,i],clean_beam[:,:,i], \
                mode='same')+dirty[:,:,i]
    
    model = Image(model.reshape((model.shape[0],model.shape[1],\
            model.shape[2],1)), freq=data.freq)
    residuals = Image(dirty.reshape((dirty.shape[0],dirty.shape[1],\
            dirty.shape[2],1)), freq=data.freq)
    clean_beam = Image(clean_beam.reshape((dirty_beam.shape[0],\
            dirty_beam.shape[1],dirty_beam.shape[2],1)), freq=data.freq)
    clean_image = Image(clean_image.reshape(model.image.shape), \
            freq=data.freq)
    
    return clean_image, residuals, beam, model
