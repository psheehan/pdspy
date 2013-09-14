from numpy import zeros,ones,arange,mat,array
from ..interferometry.interferometry import Visibilities
from ..constants.astronomy import pc, arcsec
from scipy.fftpack import fft2,fftshift

def imtovis(image,dpc=140,ext=0):

    ##### Some natural constants
    
    um  = 1.0e-4         # micron                  [cm]
    
    unit_scale = image.wave[ext]*um*arcsec     # m arcsec rad^-1
    
    im = image.image[:,:,ext]
    
    if dpc == None:
        r = 140*pc
    else:
        r = dpc*pc
    
    x = image.x / r * arcsec / unit_scale
    y = image.y / r * arcsec / unit_scale
    
    nx = x.size
    ny = y.size
    
    imsize = nx
    vis = fftshift(fft2(fftshift(im)))
    
    pixel_scale = unit_scale/((image.x[1]-image.x[0])/r * arcsec * nx)
    
    uarray = mat(ones(nx)).T * (arange(nx)-nx/2)*pixel_scale
    varray = mat(arange(nx)-nx/2).T*pixel_scale * ones(nx)
    
    u = zeros(uarray.size)
    v = zeros(uarray.size)
    freq = array([3.0e10/1.3e-1])
    real = zeros(uarray.size).reshape(uarray.size,1)
    imag = zeros(uarray.size).reshape(uarray.size,1)
    weights = ones(uarray.size).reshape(uarray.size,1)
    
    for i in arange(imsize):
        for j in arange(imsize):
            u[i*imsize+j] = uarray[i,j]
            v[i*imsize+j] = varray[i,j]
            real[i*imsize+j,0] = vis[i,j].real
            imag[i*imsize+j,0] = vis[i,j].imag
    
    return Visibilities(u,v,freq,real,imag,weights)
