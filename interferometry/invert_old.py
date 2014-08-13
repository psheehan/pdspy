import numpy
from .average import average
from .grid import grid
from .clean import clean
from ..imaging import Image
from ..constants.physics import c
from ..constants.math import pi
from ..constants.astronomy import arcsec
from scipy.fftpack import ifft2, fftshift, ifftshift
from scipy.special import jn

def invert(data, imsize=256, pixel_size=0.25, weights="natural", \
        convolution="pillbox", mfs=False, spectral=False, restfreq=None, \
        start_velocity=None, beam=False, nchannels=1, clean=False, \
        maxiter=1000, threshold=0.001, box=None):
    
    kms = 1.0e5
    
    if (data.freq.size > 1) and (mfs == False) and (spectral == False):
        averaged_data = average(data,only_freq=True)
    else:
        averaged_data = data.copy()

    if beam:
        averaged_data.real[:,:] = 1.
        averaged_data.imag[:,:] = 0.
    
    maxD = 0.

    if spectral:
        velocity = c * (restfreq - averaged_data.freq) / restfreq
        startchannel = numpy.where(abs(velocity - start_velocity*kms) == \
                abs(velocity-start_velocity*kms).min())[0][0]
    else:
        startchannel = 0

    for i in range(startchannel, startchannel + nchannels):
        baselines = numpy.unique(averaged_data.baseline)

        im = numpy.zeros((imsize, imsize))
        response = numpy.zeros((imsize, imsize))

        for j in baselines:
            baseline_data = averaged_data.get_baselines(j)
            if beam:
                weight = baseline_data.weights[baseline_data.real != 0.].sum()
            else:
                weight = baseline_data.weights[(baseline_data.real != 0.) \
                        & (baseline_data.imag != 0.)].sum()

            binsize = 1.0 / (pixel_size * imsize * arcsec)
            gridded_data = grid(baseline_data, gridsize=imsize, \
                    binsize=binsize, convolution=convolution, mfs=mfs, \
                    channel=i)
            
            real = gridded_data.real.reshape((imsize, imsize))
            imag = gridded_data.imag.reshape((imsize, imsize))
    
            if weights == "natural":
                weights = gridded_data.weights.reshape((imsize, imsize)).T
            elif weights == "uniform":
                weights = gridded_data.weights.reshape((imsize, imsize)).T
                weights[weights > 0] = 1.0

            comp = real + 1j*imag
    
            x = linspace(-imsize/2, imsize/2-1, imsize)*pixel_size
            y = linspace(-imsize/2, imsize/2-1, imsize)*pixel_size
            xarray, yarray = meshgrid(x, y)
            rarray = sqrt(xarray**2+yarray**2)
    
            if convolution == "pillbox":
                convolve_func = pillbox
            elif convolution == "expsinc":
                convolve_func = exp_sinc
    
            u = gridded_data.u.reshape((imsize,imsize))
            v = gridded_data.v.reshape((imsize,imsize))
            
            convolve = convolve_func(u,v,binsize,binsize)
            convolve = fftshift(fft2(fftshift(convolve))).real

            D1 = float(j.split("-")[0])/(c/data.freq[i])*arcsec
            D2 = float(j.split("-")[1])/(c/data.freq[i])*arcsec

            if D1 > maxD:
                maxD = D1
            elif D2 > maxD:
                maxD = D2

            nonzero = rarray != 0.
            responsivity = numpy.zeros(xarray.shape)
            responsivity[nonzero] = 4*jn(1,pi*rarray[nonzero]*D1)* \
                    jn(1,pi*rarray[nonzero]*D2)/((pi*rarray[nonzero])**2*D1*D2)
            responsivity[rarray == 0.] = 1.

            if clean:
                image = fftshift(fft2(fftshift(comp))).real/convolve
                beam = fftshift(fft2(fftshift(weights))).real/convolve

                cleanim = clean(Image(image.reshape((imsize,imsize,1))), \
                        Image(beam.reshape((imsize,imsize,1))), \
                        maxiter=maxiter, threshold=threshold, \
                        box=box).image[:,:,0]

                im += responsivity * cleanim
            else:
                im += responsivity * fftshift(fft2(fftshift(comp))).real/ \
                        convolve*weight

            response += responsivity*weight

        im[rarray > 1.0/maxD] = 0.
        im = im/response

        if i == startchannel:
            image = im.reshape((imsize,imsize,1))
        else:
            image = concatenate((image,im),axis=2)

    if spectral:
        return Image(image, \
                velocity=velocity[startchannel:startchannel+nchannels])
    else:
        return Image(image)

def pillbox(u,v,delta_u,delta_v):
    
    arr = zeros(u.shape)
    arr[u.shape[0]/2.,u.shape[1]/2.] = 1.0
    
    return arr

def exp_sinc(u,v,delta_u,delta_v):
    
    alpha1 = 1.55
    alpha2 = 2.52
    
    return sinc(u/(alpha1*delta_u))*exp(-1*(u/(alpha2*delta_u))**2)* \
           sinc(v/(alpha1*delta_u))*exp(-1*(v/(alpha2*delta_v))**2)

def spheroidal(u,v,delta_u,delta_v):
    
    alpha = 0.0
    m = 6.0
    
    return abs(1-(2*u/(m*delta_u))**2)**alpha
