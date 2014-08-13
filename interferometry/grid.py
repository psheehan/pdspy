import numpy
from .libinterferometry import Visibilities
from .freqcorrect import freqcorrect

def grid(data, gridsize=256, binsize=2000.0, convolution="pillbox", \
        mfs=False, channel=None):
    
    if mfs:
        vis = freqcorrect(data)
        u = vis.u.copy()
        v = vis.v.copy()
        freq = vis.freq.copy()
        real = vis.real.copy()
        imag = vis.imag.copy()
        weights = vis.weights.copy()
    else:
        u = data.u.copy()
        v = data.v.copy()
        if channel != None:
            freq = numpy.array([data.freq[channel]])
            real = data.real[:,channel].copy().reshape((data.real.shape[0],1))
            imag = data.imag[:,channel].copy().reshape((data.real.shape[0],1))
            weights = data.weights[:,channel].copy(). \
                    reshape((data.real.shape[0],1))
        else:
            freq = numpy.array([data.freq.mean()])
            real = data.real.copy()
            imag = data.imag.copy()
            weights = data.weights.copy()
    
    # Set the weights equal to 0 when the point is flagged (i.e. weight < 0)
    weights = numpy.where(weights < 0,0.0,weights)
    # Set the weights equal to 0 when the real and imaginary parts are both 0
    weights[(real==0) & (imag==0)] = 0.0
    
    weights /= weights.sum()
    
    # Average over the U-V plane by creating bins to average over.
    
    if gridsize%2 == 0:
        uu = numpy.linspace(-gridsize*binsize/2, (gridsize/2-1)*binsize, \
                gridsize)
        vv = numpy.linspace(-gridsize*binsize/2, (gridsize/2-1)*binsize, \
                gridsize)
    else:
        uu = numpy.linspace(-(gridsize-1)*binsize/2, (gridsize-1)*binsize/2, \
                gridsize)
        vv = numpy.linspace(-(gridsize-1)*binsize/2, (gridsize-1)*binsize/2, \
                gridsize)

    new_u, new_v = numpy.meshgrid(uu, vv)
    new_real = numpy.zeros((gridsize,gridsize))
    new_imag = numpy.zeros((gridsize,gridsize))
    new_weights = numpy.zeros((gridsize,gridsize))

    if gridsize%2 == 0:
        i = numpy.round(u/binsize+gridsize/2.)
        j = numpy.round(v/binsize+gridsize/2.)
    else:
        i = numpy.round(u/binsize+(gridsize-1)/2.)
        j = numpy.round(v/binsize+(gridsize-1)/2.)
    
    if convolution == "pillbox":
        convolve_func = ones_arr
        ninclude = 3
    elif convolution == "expsinc":
        convolve_func = exp_sinc
        ninclude = 9
    
    inc_range = numpy.linspace(-(ninclude-1)/2, (ninclude-1)/2, ninclude)
    for k in range(u.size):
        for l in inc_range+j[k]:
            for m in inc_range+i[k]:
                convolve = convolve_func(u[k]-new_u[l,m],v[k] - new_v[l,m], \
                        binsize, binsize)
                new_real[l,m] += (real[k,:]*weights[k,:]).sum()*convolve
                new_imag[l,m] += (imag[k,:]*weights[k,:]).sum()*convolve
                new_weights[l,m] += weights[k,:].sum()*convolve
    
    new_u = new_u.reshape(gridsize**2)
    new_v = new_v.reshape(gridsize**2)
    new_real = new_real.reshape((gridsize**2,1))
    new_imag = new_imag.reshape((gridsize**2,1))
    new_weights = new_weights.reshape((gridsize**2,1))

    return Visibilities(new_u, new_v, freq, new_real, new_imag, new_weights)

def exp_sinc(u, v, delta_u, delta_v):
    
    alpha1 = 1.55
    alpha2 = 2.52
    m = 6
    
    arr = numpy.sinc(u / (alpha1 * delta_u)) * \
            numpy.exp(-1 * (u / (alpha2 * delta_u))**2)* \
            numpy.sinc(v / (alpha1 * delta_v))* \
            numpy.exp(-1 * (v / (alpha2 * delta_v))**2)

    if (abs(u) >= m * delta_u / 2) or (abs(v) >= m * delta_v / 2):
        arr = 0.

    return arr

def ones_arr(u,v,delta_u,delta_v):
    
    m = 1

    arr = 1.0

    if (abs(u) >= m * delta_u / 2) or (abs(v) >= m * delta_v / 2):
        arr = 0.

    return arr
