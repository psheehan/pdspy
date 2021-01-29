import numpy
import scipy.special
from ..constants.astronomy import arcsec
from ..constants.physics import c
from .libinterferometry import Visibilities

def model(u, v, params, return_type="complex", funct="gauss", freq=230., \
        primary_beam=None):
    
    if type(params) == list:
        params = numpy.array(params)
    elif type(params) == numpy.ndarray:
        pass

    if type(funct) == str:
        funct = numpy.array([funct])
    elif type(funct) == list:
        funct = numpy.array(funct)
    elif type(funct) == numpy.ndarray:
        pass

    nparams = numpy.zeros(funct.size)
    for i in range(funct.size):
        if funct[i] == "point":
            nparams[i] = 3
        elif funct[i] == "gauss":
            nparams[i] = 6
        elif funct[i] == "circle":
            nparams[i] = 6
        elif funct[i] == "ring":
            nparams[i] = 7
    
    model = 1j*numpy.zeros(u.size)
    for i in range(funct.size):
        index = 0
        for j in range(i):
            index += nparams[j]
        index = int(index)
        
        # Make a copy of the parameters so we don't mess it up.
        par = params.copy()

        # If we were suplied with a primary beam shape, then calculate the flux
        # correction factor.
        if primary_beam != None:
            r0 = (par[index+0]**2 + par[index+1]**2)**0.5

            if primary_beam == "ALMA":
                fwhm_pb = 1.13 * (c*1.0e-2 / freq) / 12 / arcsec
            elif primary_beam == "VLA":
                fwhm_pb = 60. * 45 / (freq / 1.0e9)
            else:
                fwhm_pb = 0.

            sigma_pb = fwhm_pb / (2. * numpy.sqrt(2 * numpy.log(2.)))
            pb = numpy.exp(-r0**2/(2*sigma_pb**2))

            par[int(index + nparams[i] - 1)] *= pb

        # Convert sizes into arcseconds.
        par[index+0] *= arcsec
        par[index+1] *= arcsec
        if (funct[i] == "gauss") or (funct[i] == "circle") or \
                (funct[i] == "ring"):
            par[index+2] *= arcsec
            #if (funct[i] == "gauss"):
            if (funct[i] == "gauss") or (funct[i] == "ring"):
                par[index+3] *= arcsec
        
        # Generate the model.
        if funct[i] == "point":
            model += point_model(u, v, par[index+0], par[index+1], par[index+2])
        elif funct[i] == "gauss":
            model += gaussian_model(u, v, par[index+0], par[index+1], \
                par[index+2], par[index+3], par[index+4], par[index+5])
        elif funct[i] == "circle":
            model += circle_model(u, v, par[index+0], par[index+1], \
                par[index+2], par[index+3], par[index+4], par[index+5])
        elif funct[i] == "ring":
            model += ring_model(u, v, par[index+0], par[index+1], \
                par[index+2], par[index+3], par[index+4], par[index+5], \
                par[index+6])

    real = model.real
    imag = model.imag
    
    if return_type == "real":
        return real
    elif return_type == "imag":
        return imag
    elif return_type == "complex":
        return (real + 1j*imag).reshape((real.size,1))
    elif return_type == "amp":
        return numpy.sqrt(real**2 + imag**2)
    elif return_type == "data":
        return Visibilities(u, v, numpy.array([freq]), \
                real.reshape((real.size,1)), imag.reshape((imag.size,1)), \
                numpy.ones((real.size,1)))
    elif return_type == "append":
        return numpy.concatenate((real,imag))

def point_model(u, v, xcenter, ycenter, flux):
    
    return flux * numpy.exp(-2*3.14159 * (0 + 1j*(u * xcenter +v * ycenter)))

def circle_model(u, v, xcenter, ycenter, radius, incline, theta, flux):
    
    urot = u * numpy.cos(theta) - v * numpy.sin(theta)
    vrot = u * numpy.sin(theta) + v * numpy.cos(theta)

    numpy.seterr(invalid="ignore")

    vis = flux * scipy.special.j1(2*numpy.pi * radius * \
            numpy.sqrt(urot**2+vrot**2*numpy.cos(incline)**2)) / \
            (numpy.pi * radius * numpy.sqrt(urot**2+vrot**2*\
            numpy.cos(incline)**2)) * numpy.exp(-2*numpy.pi * (0 + \
            1j*(u*xcenter+v*ycenter)))

    numpy.seterr(invalid="warn")

    vis[urot**2+vrot**2*numpy.cos(incline)**2 == 0] = flux

    return vis

def ring_model(u, v, xcenter, ycenter, inradius, outradius, incline, theta, \
        flux):

    if inradius > outradius:
        return numpy.repeat(numpy.inf, u.size)

    A = (inradius/outradius)**2

    return flux * (circle_model(u, v, xcenter, ycenter, outradius, incline, \
            theta, 1) - A*circle_model(u, v, xcenter, ycenter, inradius, \
            incline, theta, 1)) / (1 - A)

def gaussian_model(u, v, xcenter, ycenter, usigma, vsigma, theta, flux):

    #if vsigma > usigma:
    #    return numpy.repeat(numpy.inf, u.size)
    
    urot = u * numpy.cos(theta) - v * numpy.sin(theta)
    vrot = u * numpy.sin(theta) + v * numpy.cos(theta)
    
    return flux * numpy.exp(-2*numpy.pi**2 * (usigma**2 * urot**2 + \
            vsigma**2 * vrot**2)) * numpy.exp(-2*numpy.pi * \
            1j*(u*xcenter+v*ycenter))
