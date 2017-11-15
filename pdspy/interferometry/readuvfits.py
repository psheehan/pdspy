from astropy.io.fits import open
from ..constants.physics import c
from .libinterferometry import Visibilities
import numpy

def readuvfits(filename, fmt="miriad", fast=False):
    
    data = open(filename)
    
    header = data[0].header
    u = data[0].data.field(0).astype(numpy.float64)
    v = data[0].data.field(1).astype(numpy.float64)
    if fmt == "casa":
        for i in range(data[0].data.field("data").shape[5]):
            if i == 0:
                real = (data[0].data.field("data"))[:,0,0,0,:,0,0].\
                        astype(numpy.float64)
                imag = (data[0].data.field("data"))[:,0,0,0,:,0,1].\
                        astype(numpy.float64)
                weights = (data[0].data.field("data"))[:,0,0,0,:,0,2].\
                        astype(numpy.float64)
                baselines = data[0].data.field(5)
            else:
                u = numpy.concatenate((u, data[0].data.field(0).\
                        astype(numpy.float64)))
                v = numpy.concatenate((v, data[0].data.field(1).\
                        astype(numpy.float64)))
                real = numpy.concatenate((real, (data[0].data.\
                        field("data"))[:,0,0,0,:,1,0].astype(numpy.float64)))
                imag = numpy.concatenate((imag, (data[0].data.\
                        field("data"))[:,0,0,0,:,1,1].astype(numpy.float64)))
                weights = numpy.concatenate((weights, (data[0].\
                        data.field("data"))[:,0,0,0,:,1,2].\
                        astype(numpy.float64)))
                baselines = numpy.concatenate((baselines,data[0].data.field(5)))
    if fmt == "evla":
        for i in range(data[0].data.field("data").shape[5]):
            if i == 0:
                real = (data[0].data.field("data"))[:,0,0,:,0,0,0].\
                        astype(numpy.float64)
                imag = (data[0].data.field("data"))[:,0,0,:,0,0,1].\
                        astype(numpy.float64)
                weights = (data[0].data.field("data"))[:,0,0,:,0,0,2].\
                        astype(numpy.float64)
                baselines = data[0].data.field(5)
            else:
                u = numpy.concatenate((u, data[0].data.field(0).\
                        astype(numpy.float64)))
                v = numpy.concatenate((v, data[0].data.field(1).\
                        astype(numpy.float64)))
                real = numpy.concatenate((real, (data[0].data.\
                        field("data"))[:,0,0,:,0,1,0].astype(numpy.float64)))
                imag = numpy.concatenate((imag, (data[0].data.\
                        field("data"))[:,0,0,:,0,1,1].astype(numpy.float64)))
                weights = numpy.concatenate((weights, (data[0].\
                        data.field("data"))[:,0,0,:,0,1,2].\
                        astype(numpy.float64)))
                baselines = numpy.concatenate((baselines,data[0].data.field(5)))
    elif fmt == "miriad":
        real = (data[0].data.field("data"))[:,0,0,:,0,0].astype(numpy.float64)
        imag = (data[0].data.field("data"))[:,0,0,:,0,1].astype(numpy.float64)
        weights = (data[0].data.field("data"))[:,0,0,:,0,2].\
                astype(numpy.float64)
        baselines = data[0].data.field(3)
    ant2 = numpy.mod(baselines,256)
    ant1 = (baselines-ant2)/256
    baseline = numpy.repeat("  6.1-6.1",u.size)
    baseline[((ant1 < 7) & (ant2 >= 7)) ^ ((ant1 >= 7) & (ant2 < 7))] = \
            " 6.1-10.4"
    baseline[(ant1 < 7) & (ant2 < 7)] = "10.4-10.4"
    
    u = numpy.concatenate((u, -u))
    v = numpy.concatenate((v, -v))
    real = numpy.concatenate((real, real))
    imag = numpy.concatenate((imag, -imag))
    weights = numpy.concatenate((weights, weights))
    baseline = numpy.concatenate((baseline, baseline))
    
    if fmt == "evla":
        freq = data[0].header["CRVAL4"] + data[1].data.field('if freq')
        nfreq = data[1].data.field("if freq").size
        freq = freq.reshape((nfreq,))
    else:
        freq0 = header["CRVAL4"]
        delta_freq = header["CDELT4"]
        pix0 = header["CRPIX4"]
        nfreq = header["NAXIS4"]
        freq = freq0 + (numpy.arange(nfreq)-(pix0-1))*delta_freq
    
    u *= freq.mean()
    v *= freq.mean()

    weights *= nfreq
    
    data.close()
    
    uvdata = Visibilities(u, v, freq, real, imag, weights, baseline=baseline)
    
    uvdata.set_header(header)
    
    return uvdata
