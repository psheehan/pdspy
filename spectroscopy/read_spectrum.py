from numpy import loadtxt
from .spectroscopy import Spectrum

def read_spectrum(file):
    
    spectrum = loadtxt(file)
    wave = spectrum[:,0]
    flux = spectrum[:,1]
    unc = spectrum[:,2]
    
    data = Spectrum(wave,flux,unc)
    
    return data
