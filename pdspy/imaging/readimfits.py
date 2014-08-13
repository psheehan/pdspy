from astropy.io.fits import open
from numpy import array,arange,zeros,concatenate,mat,ones
from .imaging import Image

def readimfits(filename):
    
    f = open(filename)
    data = array(f)

    if array(data[0].data.shape).size == 4:
        nx = data[0].data.shape[3]
        ny = data[0].data.shape[2]
        nspec = data[0].data.shape[1]
        nz = 1 #data.size
    else:
        nx = data[0].data.shape[1]
        ny = data[0].data.shape[0]
        nspec = 1
        nz = data.size
    
    if nspec != 1:
        image = zeros(nx*ny*nspec*nz).reshape(ny,nx,nspec)
    else:
        image = zeros(nx*ny*nspec*nz).reshape(ny,nx,nz)

    header = []

    freq = zeros(data.size)

    for i in arange(max(nspec,nz)):
        if nspec != 1:
            image[:,:,i] = data[0].data[0,i,:,:].reshape(ny,nx)
        else:
            image[:,:,i] = data[i].data.reshape(ny,nx)

        #freq[i] = data[i].header["RESTFREQ"]

        if (nspec != 1) & (i == 0):
            header.append(data[0].header)
        elif (nspec != 1) & (i > 0):
            header = header
        else:
            header.append(data[i].header)

    x = array(mat(ones(ny)).T * arange(nx))
    y = array(mat(arange(ny)).T * ones(nx))

    if nspec != 1:
        v0 = data[0].header["CRVAL3"]
        dv = data[0].header["CDELT3"]
        n0 = data[0].header["CRPIX3"]

        velocity = (arange(nspec)-(n0-1))*dv/1000.+v0/1000.
    else:
        velocity = None

    f.close()
 
    return Image(image,x=x,y=y,header=header,velocity=velocity)#,RA=RA,Dec=Dec)
