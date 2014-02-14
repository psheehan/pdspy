from numpy import zeros, sqrt, array, mat, ones, zeros, arange, exp, cos, sin, \
        ravel
from scipy.signal import fftconvolve
from scipy.optimize import leastsq
from ..imaging import Image

def clean(image, beam, gain=0.1, maxiter=1000, threshold=0.001, box=None):
    
    dirty = image.image[:,:,0]
    dirtybeam = beam.image[:,:,0]
    nx = dirty.shape[1]
    ny = dirty.shape[0]
    wherezero = dirty == 0
    nonzero = dirty != 0
    
    model = numpy.zeros(dirty.shape)

    mask = numpy.zeros(dirty.shape)
    if mask != None:
        mask[box[2]+nx/2:box[3]+nx/2,box[0]+ny/2:box[1]+ny/2]=1.
    else:
        mask += 1.
    
    n = 0
    cont = True
    while (n < maxiter) and cont:
        maxval = dirty*mask == (dirty*mask).max()
        
        model[maxval] = model[maxval] + dirty[maxval]*gain
        
        subtract = numpy.zeros(dirty.shape)
        subtract[maxval] = dirty[maxval]*gain
        
        dirty = dirty - fftconvolve(subtract, dirtybeam, mode='same')
        dirty[wherezero] = 0.
        
        rms = dirty[nonzero].std()
        if rms < threshold:
            cont=False
        
        n = n + 1
    
    fitfunc = lambda p, x, y: numpy.exp(-(x * numpy.cos(p[2]) - \
            y * numpy.sin(p[2]))**2 / (2*p[0]**2) - (x * numpy.sin(p[2]) + \
            y * numpy.cos(p[2]))**2 / (2*p[1]**2))
    errfunc = lambda p, x, y, z: numpy.ravel(fitfunc(p, x, y) - z)
    p0 = [3.,3.,0.]

    x = array(mat(ones(ny)).T*(arange(nx)-nx/2))
    y = array(mat(arange(ny)-ny/2).T*ones(nx))

    p, success = leastsq(errfunc, p0, args=(x,y,dirtybeam))
    # clean_beam = gauss2dfit(beam,coeff,/tilt)

    clean_beam = fitfunc(p, x, y)
    
    clean_image = fftconvolve(model,clean_beam,mode='same')+dirty
    
    model = Image(model.reshape((model.shape[0],model.shape[1],1)))
    residuals = Image(dirty.reshape((dirty.shape[0],dirty.shape[1],1)))
    clean_beam = Image(clean_beam.reshape(model.image.shape))
    clean_image = Image(clean_image.reshape(model.image.shape))
    
    return clean_image
