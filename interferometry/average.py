from numpy import where,zeros,sqrt,cos,sin,arange,array,abs,mat,ones,round, \
        linspace
from .interferometry import Visibilities

def uvaverage(data,only_freq=False,nbins=17,binsize=None,radial=False, \
        vaxis=False,channels=False,grid=False):
    
    u = data.u
    v = data.v
    real = data.real
    imag = data.imag
    weights = data.weights
    uvdist = data.uvdist
    
    # Set the weights equal to 0 when the point is flagged (i.e. weight < 0)
    weights = where(weights < 0,0.0,weights)
    # Set the weights equal to 0 when the real and imaginary parts are both 0
    weights[(real == 0) & (imag == 0)] = 0.0
    
    # If nbins = 0 then only average over the channels, and then stop.
    
    if only_freq == True:
        new_real = zeros(real[:,0].size).reshape(real[:,0].size,1)
        new_imag = zeros(real[:,0].size).reshape(real[:,0].size,1)
        new_weights = zeros(real[:,0].size).reshape(real[:,0].size,1)
        
        new_weights[:,0] = weights.sum(axis=1)
        include = new_weights[:,0] != 0.0
        new_real[include,0] = (real[include,:]*weights[include,:]).sum(axis=1)/\
            weights[include,:].sum(axis=1)
        new_imag[include,0] = (imag[include,:]*weights[include,:]).sum(axis=1)/\
            weights[include,:].sum(axis=1)
        
        good = new_weights[:,0] > 0
        new_real = new_real[good,:]
        new_imag = new_imag[good,:]
        new_weights = new_weights[good,:]
        new_u = data.u[good]
        new_v = data.v[good]
        new_baseline = data.baseline[good]
        
        freq = array([data.freq.mean()])
        
        return Visibilities(new_u,new_v,freq,new_real,new_imag,new_weights, \
                baseline=new_baseline)
    
    # If radial = True, then average into radial bins.
    
    if radial == True:
        bin_size = uvdist.max()*1.05/nbins
        uvdist_1d = (arange(nbins)+1./2)*bin_size
        new_real = zeros((nbins,1))
        new_imag = zeros((nbins,1))
        new_weights = zeros((nbins,1))
        new_u = uvdist_1d
        new_v = zeros(nbins)
        
        for i in arange(nbins):
            include = abs(uvdist-uvdist_1d[i]) < bin_size/2
    
            if weights[include,:].sum() > 0.0:
                new_real[i,0] = (real[include,:]*weights[include,:]).sum()/ \
                    weights[include,:].sum()
                new_imag[i,0] = (imag[include,:]*weights[include,:]).sum()/ \
                    weights[include,:].sum()
                new_weights[i] = weights[include,:].sum()
            else:
                new_weights[i] = -1.0
        
        good_data= new_weights[:,0] != -1.0
        new_real = new_real[good_data,:]
        new_imag = new_imag[good_data,:]
        new_weights = new_weights[good_data,:]
        new_u = new_u[good_data]
        new_v = new_v[good_data]
        
        freq = array([data.freq.sum()/data.freq.size])
        
        return Visibilities(new_u,new_v,freq,new_real,new_imag,new_weights)

    # If vaxis = True, average into bins along the v-axis.
    
    if vaxis == True:
        bins = linspace(u.min(),u.max(),num=nbins+1)
        new_real = zeros((nbins,1))
        new_imag = zeros((nbins,1))
        new_weights = zeros((nbins,1))
        new_u = zeros(nbins)
        new_v = 0.5*(bins[0:nbins]+bins[1:nbins+1])
        
        for i in arange(nbins):
            include = (v >= bins[i]) & (v <= bins[i+1]) & (u <= 50.) & \
                    (u >= -50.)
    
            if weights[include,:].sum() > 0.0:
                new_real[i,0] = (real[include,:]*weights[include,:]).sum()/ \
                    weights[include,:].sum()
                new_imag[i,0] = (imag[include,:]*weights[include,:]).sum()/ \
                    weights[include,:].sum()
                new_weights[i] = weights[include,:].sum()
            else:
                new_weights[i] = -1.0
        
        good_data= new_weights[:,0] != -1.0
        new_real = new_real[good_data,:]
        new_imag = new_imag[good_data,:]
        new_weights = new_weights[good_data,:]
        new_u = new_u[good_data]
        new_v = new_v[good_data]
        
        freq = array([data.freq.sum()/data.freq.size])
        
        return Visibilities(new_u,new_v,freq,new_real,new_imag,new_weights)

    # Average over the U-V plane by creating bins to average over.
    
    if channels == False:
        nchannels = 1
    else:
        nchannels = real[0,:].size
    
    if binsize == None:
        binsize = 2*uvdist.max()/(nbins-1)
    
    new_u = zeros((nbins,nbins))
    new_v = zeros((nbins,nbins))
    new_real = zeros((nbins,nbins))
    new_imag = zeros((nbins,nbins))
    new_weights = zeros((nbins,nbins))

    i = round(u/binsize+nbins/2.)#-1)
    j = round(v/binsize+nbins/2.)
    
    for k in range(u.size):
        new_real[j[k],i[k]] += (real[k,:]*weights[k,:]).sum()
        new_imag[j[k],i[k]] += (imag[k,:]*weights[k,:]).sum()
        new_weights[j[k],i[k]] += weights[k,:].sum()
        new_u[j[k],i[k]] += u[k]*weights[k,:].sum()
        new_v[j[k],i[k]] += v[k]*weights[k,:].sum()
    
    good_data = new_weights != 0.0
    if grid == False:
        new_real = new_real[good_data]
        new_imag = new_imag[good_data]
        new_weights = new_weights[good_data]
        new_u = new_u[good_data]
        new_v = new_v[good_data]
    else:
        new_u = array(mat(ones(nbins)).T*(arange(nbins)-nbins/2.)*binsize)
        new_v = array(mat(arange(nbins)-nbins/2.).T*binsize*ones(nbins))
    
    good_data = new_weights != 0.0
    new_real[good_data] /= new_weights[good_data]
    new_imag[good_data] /= new_weights[good_data]
    if grid == False:
        new_u[good_data] /= new_weights[good_data]
        new_v[good_data] /= new_weights[good_data]
    
    new_u = new_u.reshape(new_u.size)
    new_v = new_v.reshape(new_v.size)
    new_real = new_real.reshape((new_real.size,nchannels))
    new_imag = new_imag.reshape((new_imag.size,nchannels))
    new_weights = new_weights.reshape((new_weights.size,nchannels))
    
    if channels == False:
        freq = array([data.freq.sum()/data.freq.size])
    
    return Visibilities(new_u,new_v,freq,new_real,new_imag,new_weights)
