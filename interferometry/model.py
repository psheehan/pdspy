import numpy
from ..constants.astronomy import arcsec
from ..constants.physics import c
from ..constants.math import py
from .interferometry import Visibilities

def model(u, v, params, return_type="complex", funct=["gauss"], \
            nparams=None, freq=array([226999690470.0])):
    
    params = numpy.array(params)
    funct = numpy.array(funct)
    
    if nparams == None:
        nparams = numpy.zeros(funct.size)
    
    for i in range(funct.size):
        if (funct[i] == "point") or (funct[i] == "gauss") or \
                (funct[i] == "circle"):
            if funct[i] == "point":
                nparams[i] = 3
            elif funct[i] == "gauss":
                nparams[i] = 6
            elif funct[i] == "circle":
                nparams[i] = 6
    
    model = numpy.zeros(u.size)+1j*numpy.zeros(u.size)
    for i in range(funct.size):
        index = 0
        for j in range(i):
            index = index+nparams[j]
        
        par = params.copy()
        par[index+0] *= arcsec
        par[index+1] *= arcsec
        if (funct[i] == "gauss") ^ (funct[i] == "circle"):
            par[index+2] *= arcsec
            if funct[i] == "gauss":
                par[index+3] *= arcsec
        
        if funct[i] == "point":
            model += point_model(u, v, par[index+0], par[index+1], par[index+2])
        elif funct[i] == "gauss":
            model += gaussian_model(u, v, par[index+0], par[index+1], \
                par[index+2], par[index+3], par[index+4], par[index+5])
        elif funct[i] == "circle":
            model += circle_model(u, v, par[index+0], par[index+1], \
                par[index+2], par[index+3], par[index+4], par[index+5])
    
    real = model.real
    imag = model.imag
    
    if return_type == "real":
        return real
    elif return_type == "imag":
        return imag
    elif return_type == "complex":
        return real + 1j*imag
    elif return_type == "amp":
        return numpy.sqrt(real**2 + imag**2)
    elif return_type == "data":
        return Visibilities(u, v, numpy.array([freq[0]]), \
                real.reshape((real.size,1)), imag.reshape((imag.size,1)), \
                numpy.ones((real.size,1)))
    elif return_type == "append":
        return numpy.concatenate((real,imag))

def point_model(u, v, xcenter, ycenter, flux):
    
    return flux * numpy.exp(-2*3.14159 * (0 + 1j*(u * xcenter +v * ycenter)))

def circle_model(u, v, xcenter, ycenter, radius, incline, theta, flux):
    
    urot = u * numpy.cos(theta) - v * numpy.sin(theta)
    vrot = u * numpy.sin(theta) + v * numpy.cos(theta)
    
    return flux * BeselJ(2*pi * radius* numpy.sqrt(urot**2 + vrot**2 * \
            numpy.cos(incline)**2),1) / (pi * radius * numpy.sqrt(urot**2 + \
            vrot**2 * numpy.cos(incline)**2)) * numpy.exp(-2*pi * (0 + \
            1j*(u*xcenter+v*ycenter)))

def gaussian_model(u, v, xcenter, ycenter, usigma, vsigma, theta, flux):
    
    urot = u * numpy.cos(theta) - v * numpy.sin(theta)
    vrot = u * numpy.sin(theta) + v * numpy.cos(theta)
    
    return flux * numpy.exp(-2*pi**2 * (usigma**2 * urot**2 + \
            vsigma**2 * vrot**2)) * numpy.exp(-2*pi * 1j*(u*xcenter+v*ycenter))
