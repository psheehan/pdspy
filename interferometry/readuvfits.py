from pyfits import open
from numpy import concatenate, arange, mod, repeat
from ..constants.physics import c
from .interferometry import Visibilities

def readuvfits(filename,fmt="miriad",fast=False):
    
    data = open(filename)
    
    header = data[0].header
    u = data[0].data.field("c1")
    v = data[0].data.field("c2")
    if fmt == "casa":
        real = (data[0].data.field("data"))[:,0,0,0,:,0,0]
        imag = (data[0].data.field("data"))[:,0,0,0,:,0,1]
        weights = (data[0].data.field("data"))[:,0,0,0,:,0,2]
        baselines = data[0].data.field("c6")
    elif fmt == "miriad":
        real = (data[0].data.field("data"))[:,0,0,:,0,0]
        imag = (data[0].data.field("data"))[:,0,0,:,0,1]
        weights = (data[0].data.field("data"))[:,0,0,:,0,2]
        baselines = data[0].data.field("c4")
    ant2 = mod(baselines,256)
    ant1 = (baselines-ant2)/256
    baseline = repeat("  6.1-6.1",u.size)
    baseline[((ant1 < 7) & (ant2 >= 7)) ^ ((ant1 >= 7) & (ant2 < 7))] = \
            " 6.1-10.4"
    baseline[(ant1 < 7) & (ant2 < 7)] = "10.4-10.4"
    
    u = concatenate((u,-u))
    v = concatenate((v,-v))
    real = concatenate((real,real))
    imag = concatenate((imag,-imag))
    weights = concatenate((weights,weights))
    baseline = concatenate((baseline,baseline))
    
    freq0 = header["CRVAL4"]
    delta_freq = header["CDELT4"]
    pix0 = header["CRPIX4"]
    nfreq = header["NAXIS4"]
    freq = freq0 + (arange(nfreq)-pix0)*delta_freq
    
    u *= freq.mean()
    v *= freq.mean()
    
    data.close()
    
    uvdata = Visibilities(u,v,freq,real,imag,weights,baseline=baseline)
    
    uvdata.set_header(header)
    
    return uvdata
