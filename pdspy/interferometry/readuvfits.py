from astropy.io.fits import open
from ..constants.physics import c
from .libinterferometry import Visibilities
import numpy

def readuvfits(filename, fmt="casa", fast=False):
    
    data = open(filename)
    
    header = data[0].header
    u = data[0].data.field(0).astype(numpy.float64)
    v = data[0].data.field(1).astype(numpy.float64)

    order = numpy.argsort(u)

    u = u[order]
    v = v[order]

    if fmt in ["casa","noema"]:
        arr = data[0].data.field("data").astype(numpy.float)

        for i in range(min(2,data[0].data.field("data").shape[5])):
            if i == 0:
                real = [arr[:,0,0,j,:,0,0] for j in range(arr.shape[3])]
                imag = [arr[:,0,0,j,:,0,1] for j in range(arr.shape[3])]
                weights = [arr[:,0,0,j,:,0,2] for j in range(arr.shape[3])]
                baselines = data[0].data.field(5)

                real = numpy.concatenate(real, axis=1)
                imag = numpy.concatenate(imag, axis=1)
                weights = numpy.concatenate(weights, axis=1)

                real = real[order,:]
                imag = imag[order,:]
                weights = weights[order,:]
                baselines = baselines[order]
            else:
                new_real = [arr[:,0,0,j,:,1,0] for j in range(arr.shape[3])]
                new_imag = [arr[:,0,0,j,:,1,1] for j in range(arr.shape[3])]
                new_weights = [arr[:,0,0,j,:,1,2] for j in range(arr.shape[3])]

                new_real = numpy.concatenate(new_real, axis=1)
                new_imag = numpy.concatenate(new_imag, axis=1)
                new_weights = numpy.concatenate(new_weights, axis=1)

                new_real = new_real[order,:]
                new_imag = new_imag[order,:]
                new_weights = new_weights[order,:]

                real = real*weights + new_real*new_weights
                imag = imag*weights + new_imag*new_weights
                weights += new_weights
                real[weights != 0] /= weights[weights != 0]
                imag[weights != 0] /= weights[weights != 0]
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
    
    if fmt == "casa":
        IF = data[1].data.field('if freq')[0]
        delta_freq = data[1].data.field('ch width')[0]
    else:
        IF = 0.
        delta_freq = header["CDELT4"]
    freq0 = header["CRVAL4"]
    pix0 = header["CRPIX4"]
    nfreq = header["NAXIS4"]

    if isinstance(IF, float):
        IF = numpy.array([IF])
        delta_freq = numpy.array([delta_freq])

    freq = [freq0 + IF[i] + (numpy.arange(nfreq)-(pix0-1))*delta_freq[i] \
            for i in range(IF.size)]

    freq = numpy.concatenate(freq)
    nfreq = header["NAXIS4"]*IF.size
    
    u *= freq.mean()
    v *= freq.mean()

    weights *= nfreq
    
    data.close()

    uvdata = Visibilities(u, v, freq, real, -imag, weights, baseline=baseline)
    
    uvdata.set_header(header)
    
    return uvdata
