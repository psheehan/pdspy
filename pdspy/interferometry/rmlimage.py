from scipy.optimize import minimize
from ..imaging import Image
from . import interpolate_model
import numpy

def rmlimage(data, imsize=512, pixelsize=0.01):

    def neg_ln_like(p):
        if numpy.any(p <= 0):
            return numpy.inf
        else:
            model = Image(numpy.array(p).reshape((imsize,imsize,1,1)), \
                    x=numpy.arange(imsize)*pixelsize, y=numpy.arange(imsize)*\
                    pixelsize, freq=data.freq)

            model_vis = interpolate_model(data.u, data.v, data.freq, model)

            return ((data.real - model_vis.real)**2 * data.weights + \
                    (data.imag - model_vis.imag)**2 * data.weights).sum() + \
                    (model.image * numpy.log(model.image)).sum()

    result = minimize(neg_ln_like, numpy.ones(imsize**2))

    return Image(numpy.array(result.x).reshape((imsize,imsize,1,1)), \
            x=numpy.arange(imsize)*pixelsize, y=numpy.arange(imsize)*\
            pixelsize, freq=data.freq)
