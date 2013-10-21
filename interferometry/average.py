from numpy import where,zeros,sqrt,cos,sin,arange,array,abs,mat,ones,round, \
        linspace, meshgrid
from .interferometry import Visibilities

def average(data,gridsize=256,binsize=None,radial=False):
    
    u = data.u.copy()
    v = data.v.copy()
    uvdist = data.uvdist.copy()
    real = data.real.copy()
    imag = data.imag.copy()
    weights = data.weights.copy()
    
    # Set the weights equal to 0 when the point is flagged (i.e. weight < 0)
    weights = where(weights < 0,0.0,weights)
    # Set the weights equal to 0 when the real and imaginary parts are both 0
    weights[(real == 0) & (imag == 0)] = 0.0
    
    # Average over the U-V plane by creating bins to average over.
    
    if radial:
        new_u = linspace(binsize/2,(gridsize-0.5)*binsize,gridsize)
        new_u = new_u.reshape((1,gridsize))
        new_v = zeros((1,gridsize))
        new_real = zeros((1,gridsize))
        new_imag = zeros((1,gridsize))
        new_weights = zeros((1,gridsize))

        i = round(uvdist/binsize)
        j = zeros(uvdist.size)
    else:
        uu = linspace(-(gridsize-1)*binsize/2,(gridsize-1)*binsize/2,gridsize)
        vv = linspace(-(gridsize-1)*binsize/2,(gridsize-1)*binsize/2,gridsize)

        new_u, new_v = meshgrid(uu, vv)
        new_real = zeros((gridsize,gridsize))
        new_imag = zeros((gridsize,gridsize))
        new_weights = zeros((gridsize,gridsize))

        i = round(u/binsize+(gridsize-1)/2.)
        j = round(v/binsize+(gridsize-1)/2.)
    
    for k in range(u.size):
        new_real[j[k],i[k]] += (real[k,:]*weights[k,:]).sum()
        new_imag[j[k],i[k]] += (imag[k,:]*weights[k,:]).sum()
        new_weights[j[k],i[k]] += weights[k,:].sum()
    
    good_data = new_weights != 0.0
    new_u = new_u[good_data]
    new_v = new_v[good_data]
    new_real = new_real[good_data] / new_weights[good_data]
    new_imag = new_imag[good_data] / new_weights[good_data]
    new_weights = new_weights[good_data]
    
    freq = array([data.freq.sum()/data.freq.size])
    
    return Visibilities(new_u,new_v,freq,new_real,new_imag,new_weights)
