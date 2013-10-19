from numpy import where,zeros,sqrt,cos,sin,arange,array,exp,abs,mat,ones, \
                  round,sinc
from .interferometry import Visibilities
from .uvfreqcorrect import uvfreqcorrect
from ..constants.math import pi

def uvgrid(data,gridsize=256,binsize=2.0,channels=False,convolution="pillbox", \
           mfs=False,channel=None):
    
    if mfs:
        vis = uvfreqcorrect(data)
        u = vis.u
        v = vis.v
        real = vis.real
        imag = vis.imag
        weights = vis.weights
    else:
        u = data.u
        v = data.v
        if channel != None:
            real = data.real[:,channel].reshape((data.real.shape[0],1))
            imag = data.imag[:,channel].reshape((data.real.shape[0],1))
            weights = data.weights[:,channel].reshape((data.real.shape[0],1))
        else:
            real = data.real
            imag = data.imag
            weights = data.weights
    
    # Set the weights equal to 0 when the point is flagged (i.e. weight < 0)
    weights = where(weights < 0,0.0,weights)
    # Set the weights equal to 0 when the real and imaginary parts are both 0
    weights[(real==0) & (imag==0)] = 0.0
    
    weights /= weights.sum()
    
    # Average over the U-V plane by creating bins to average over.
    
    if channels == False:
        nchannels = 1
    else:
        nchannels = real[0,:].size
    
    new_u = array(mat(ones(gridsize)).T*(arange(gridsize)-gridsize/2.)*binsize)
    new_v = array(mat(arange(gridsize)-gridsize/2.).T*binsize*ones(gridsize))
    new_real = zeros((gridsize,gridsize))
    new_imag = zeros((gridsize,gridsize))
    new_weights = zeros((gridsize,gridsize))

    i = round(u/binsize+gridsize/2.)
    j = round(v/binsize+gridsize/2.)
    
    if convolution == "pillbox":
        convolve_func = ones_arr
        ninclude = 1
    elif convolution == "expsinc":
        convolve_func = exp_sinc
        ninclude = 7
    
    inc_range = arange(ninclude)-(ninclude-1)/2
    for k in range(u.size):
        for l in inc_range+j[k]:
            for m in inc_range+i[k]:
                convolve = convolve_func(u[k]-new_u[l,m],v[k]- \
                    new_v[l,m],binsize,binsize)
                new_real[l,m] += (real[k,:]*weights[k,:]).sum()*convolve
                new_imag[l,m] += (imag[k,:]*weights[k,:]).sum()*convolve
                new_weights[l,m] += weights[k,:].sum()*convolve
    
    new_u = new_u.reshape(gridsize**2)
    new_v = new_v.reshape(gridsize**2)
    new_real = new_real.reshape((gridsize**2,nchannels))
    new_imag = new_imag.reshape((gridsize**2,nchannels))
    new_weights = new_weights.reshape((gridsize**2,nchannels))
    
    if channels == False:
        freq = array([data.freq.mean()])
    
    return Visibilities(new_u,new_v,freq,new_real,new_imag,new_weights)

def exp_sinc(u,v,delta_u,delta_v):
    
    alpha1 = 1.55
    alpha2 = 2.52
    
    return sinc(u/(alpha1*delta_u))*exp(-1*(u/(alpha2*delta_u))**2)* \
           sinc(v/(alpha1*delta_u))*exp(-1*(v/(alpha2*delta_v))**2)

def ones_arr(u,v,delta_u,delta_v):
    
    return 1.0
