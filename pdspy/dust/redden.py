from . import __file__
from scipy.interpolate import interp1d
import numpy
import os

def redden(wave, flux, Av, law="steenman"):

    # Read in the extinction coefficients.

    if law == 'steenman':
        data = numpy.loadtxt(os.path.dirname(__file__)+ \
                "/reddening/steenman_the.dat", usecols=[1,2])
    elif law == 'mcclure':
        if Av <= 1.:
            data = numpy.loadtxt(os.path.dirname(__file__)+ \
                    "/reddening/mcclure.dat", usecols=[0,1])
        else:
            data = numpy.loadtxt(os.path.dirname(__file__)+ \
                    "/reddening/mcclure.dat", usecols=[0,2])

    waves = data[:,0]
    coeffs = data[:,1]

    # Interpolate the data to the wavelength range supplied.

    coeffs = interp1d(numpy.log10(waves), numpy.log10(coeffs), \
            fill_value="extrapolate")

    coeff = 10.**coeffs(numpy.log10(wave))

    # Now adjust the provided flux.

    return flux / 10.**(Av * coeff / 2.5)
