#==========================================================================
#==========================================================================
#==========================================================================
#              PYTHON PACKAGE TO READ RESULTS FROM RADMC-3D
#==========================================================================
#==========================================================================
#==========================================================================

from os.path import exists
from numpy import array, empty, linspace

#==========================================================================
#                        ROUTINES FOR IMAGES
#==========================================================================


#-----------------------------------------------------------------
#              READ THE RECTANGULAR TELESCOPE IMAGE
#-----------------------------------------------------------------
def image(filename=None,ext=None):
    
    nx=0
    ny=0
    nf=0
    sizepix_x=0.e0
    sizepix_y=0.e0
    
    if (filename == None):
        if (ext == None):
            filename = "image.out"
        else:
            filename = "image_"+str(ext)+".out"

    if (exists(filename) == False):
        print("Sorry, cannot find {0:s}. Presumably radmc2d exited without success. See above for possible error messages of radmc3d!".format(filename))
        return
    else:
        f = open(filename, "r")

    # Read the image.

    iformat = int(f.readline())

    if (iformat < 1) or (iformat > 4):
        print("ERROR: File format of {0:s} not recognized.".format(filename))
        return

    if (iformat == 1) or (iformat == 3):
        radian = (1 == 0)
    else:
        radian = (1 == 1)

    if (iformat == 1) or (iformat == 2):
        stokes = (1 == 0)
    else:
        stokes = (1 == 1)

    nx, ny = tuple(array(f.readline().split(),dtype=int))
    nf = int(f.readline())
    sizepix_x, sizepix_y = tuple(array(f.readline().split(),dtype=float))

    lam = empty(nf)
    for i in range(nf):
        lam[i] = float(f.readline())
    
    f.readline()

    if stokes:
        image = empty((ny,nx,nf,4))
    else:
        image = empty((ny,nx,nf,1))

    for i in range(nf):
        for j in range(ny):
            for k in range(nx):
                if stokes:
                    image[j,k,i,:] = array(f.readline().split(), dtype=float)
                else:
                    image[j,k,i,0] = float(f.readline())

                if (j == ny-1) and (k == nx-1):
                    f.readline()

    f.close()

    # Compute the flux in this image as seen at 1 pc.

    if stokes:
        flux = image[:,:,:,0].sum(axis=0).sum(axis=0)
    else:
        flux = image.sum(axis=0).sum(axis=0)

    flux *= sizepix_x*sizepix_y

    if not radian:
        pc = 3.0857200e18
        flux /= pc**2

    # Compute the x and y coordinates

    x = linspace(-(nx-1)/2.,(nx-1)/2.,nx)*sizepix_x
    y = linspace(-(ny-1)/2.,(ny-1)/2.,ny)*sizepix_y
    
    return image, x, y, lam

#==========================================================================
#                        ROUTINES FOR SPECTRA
#==========================================================================

#-----------------------------------------------------------------
#                       READ A SPECTRUM
#-----------------------------------------------------------------

def spectrum(filename=None,ext=None):

    if (filename == None):
        if (ext == None):
            filename = "spectrum.out"
        else:
            filename = "spectrum_"+str(ext)+".out"

    if exists(filename):
        f = open(filename, "r")
    else:
        print("Sorry, cannot find {0:s}. Presumably radmc2d exited without success. See above for possible error messages of radmc3d!".format(filename))
        return

    # Read the spectrum.

    iformat = int(f.readline())

    if (iformat != 1):
        print("ERROR: File format of {0:s} not recognized.".format(filename))
        return

    nf = int(f.readline())

    f.readline()

    lam = empty(nf)
    spectrum = empty(nf)

    for i in range(nf):
        lam[i], spectrum[i] = tuple(array(f.readline().split(),dtype=float))

    cc = 2.9979245800000e10

    freq = 1e4*cc/lam

    f.close()

    return spectrum, lam

#==========================================================================
#               ROUTINES FOR READING DENSITY AND TEMPERATURE
#==========================================================================

#--------------------------------------------------------------------------
#                    READ THE AMR GRID INFORMATION
#--------------------------------------------------------------------------

def dust_temperature(filename=None, ext=None):

    f = open("dust_temperature.dat","r")

    f.readline()

    ncells = int(f.readline())
    nspecies = int(f.readline())

    temperature = []
    for i in range(nspecies):
        temp = empty((ncells,))

        for j in range(ncells):
            temp[j] = float(f.readline())

        temperature.append(temp)

    f.close()

    return temperature
