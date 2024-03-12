#==========================================================================
#==========================================================================
#==========================================================================
#              PYTHON PACKAGE TO READ RESULTS FROM RADMC-3D
#==========================================================================
#==========================================================================
#==========================================================================

cimport cython
cimport numpy

from os.path import exists
from numpy import array, empty, linspace, fromfile, loadtxt, intc, hstack
import sys

#==========================================================================
#                        ROUTINES FOR IMAGES
#==========================================================================


#-----------------------------------------------------------------
#              READ THE RECTANGULAR TELESCOPE IMAGE
#-----------------------------------------------------------------
@cython.boundscheck(False)
def image(filename=None,ext=None,binary=False):
    
    cdef unsigned int nx=0
    cdef unsigned int ny=0
    cdef unsigned int nf=0
    cdef unsigned int index
    cdef numpy.ndarray[ndim=1, dtype=double] data
    sizepix_x=0.e0
    sizepix_y=0.e0
    
    if (filename == None):
        if (ext == None):
            if binary:
                filename = "image.bout"
            else:
                filename = "image.out"
        else:
            if binary:
                filename = "image_"+str(ext)+".bout"
            else:
                filename = "image_"+str(ext)+".out"

    if (exists(filename) == False):
        f = open("radmc3d.out","r")
        lines = f.readlines()
        f.close()
        for line in lines:
            sys.stdout.write(line)
        print("Sorry, cannot find {0:s}. Presumably radmc2d exited without success. See above for possible error messages of radmc3d!".format(filename))
        return
    else:
        if binary:
            f = open(filename, "rb")
            data = fromfile(filename)
        else:
            f = open(filename, "r")

    # Read the image.

    if binary:
        iformat = int.from_bytes(f.read(8), byteorder="little")
    else:
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

    if binary:
        nx = int.from_bytes(f.read(8), byteorder="little")
        ny = int.from_bytes(f.read(8), byteorder="little")
        nf = int.from_bytes(f.read(8), byteorder="little")
        sizepix_x, sizepix_y = tuple(data[4:6])
    else:
        nx, ny = tuple(array(f.readline().split(),dtype=int))
        nf = int(f.readline())
        sizepix_x, sizepix_y = tuple(array(f.readline().split(),dtype=float))

    if binary:
        lam = data[6:6+nf]
    else:
        lam = empty(nf)
        for i in range(nf):
            lam[i] = float(f.readline())
    
        f.readline()

    cdef numpy.ndarray[ndim=4, dtype=double] image
    if stokes:
        image = empty((ny,nx,nf,4))
    else:
        image = empty((ny,nx,nf,1))

    if binary:
        index = 6+nf
        for i in range(nf):
            for j in range(ny):
                for k in range(nx):
                    if stokes:
                        image[j,k,i,:] = data[index:index+4]
                        index += 4
                    else:
                        image[j,k,i,0] = data[index]
                        index += 1
    else:
        for i in range(nf):
            for j in range(ny):
                for k in range(nx):
                    if stokes:
                        image[j,k,i,:] = array(f.readline().split(), \
                                dtype=float)
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

    x = linspace(-(<int>nx-1)/2.,(<int>nx-1)/2.,<int>nx)*sizepix_x
    y = linspace(-(<int>ny-1)/2.,(<int>ny-1)/2.,<int>ny)*sizepix_y
    
    return image, x, y, lam

#-----------------------------------------------------------------
#              READ AN UNSTRUCTURED IMAGE
#-----------------------------------------------------------------

def unstructured_image(filename=None,ext=None,binary=False):
    
    data = loadtxt(filename)

    x = data[:,0]
    y = data[:,1]
    
    image = data[:,4:]

    return image, x, y

#-----------------------------------------------------------------
#              READ THE CIRCULAR TELESCOPE IMAGE
#-----------------------------------------------------------------
@cython.boundscheck(False)
def circimage(filename=None,ext=None,binary=False):
    
    cdef unsigned int nx=0
    cdef unsigned int ny=0
    cdef unsigned int nf=0
    cdef unsigned int index
    cdef numpy.ndarray[ndim=1, dtype=double] data
    sizepix_x=0.e0
    sizepix_y=0.e0
    
    if (filename == None):
        if (ext == None):
            if binary:
                filename = "circimage.bout"
            else:
                filename = "circimage.out"
        else:
            if binary:
                filename = "circimage_"+str(ext)+".bout"
            else:
                filename = "circimage_"+str(ext)+".out"

    if (exists(filename) == False):
        f = open("radmc3d.out","r")
        lines = f.readlines()
        f.close()
        for line in lines:
            sys.stdout.write(line)
        print("Sorry, cannot find {0:s}. Presumably radmc2d exited without success. See above for possible error messages of radmc3d!".format(filename))
        return
    else:
        if binary:
            f = open(filename, "rb")
            data = fromfile(filename)
        else:
            f = open(filename, "r")

    # Read the image.

    if binary:
        iformat = fromfile(filename, count=1, dtype=intc).astype(int)[0]
    else:
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

    # Read in the number of pixels and frequencies.
    if binary:
        _, nr, nphi, nf = list(fromfile(filename, count=4, dtype=intc).\
                astype(int))
    else:
        nr, nphi = tuple(array(f.readline().split(),dtype=int))
        nf = int(f.readline())

    # Read in the radius information.

    if binary:
        index = 2
        re = hstack(([0.0], data[index:index+nr+1]))
        index += nr+1
        rc = hstack(([0.0], data[index:index+nr]))
        index += nr
    else:
        f.readline()
        re = empty(nr+2)
        for i in range(nr+2):
            re[i] = float(f.readline())

        f.readline()
        rc = empty(nr+1)
        for i in range(nr+1):
            rc[i] = float(f.readline())

    # Read in the phi information.

    if binary:
        phie = data[index:index+nphi+1]
        index += nphi+1
        phic = data[index:index+nphi]
        index += nphi
    else:
        f.readline()
        phie = empty(nphi+1)
        for i in range(nphi+1):
            phie[i] = float(f.readline())

        f.readline()
        phic = empty(nphi)
        for i in range(nphi):
            phic[i] = float(f.readline())

    # Read in the wavelength information.

    if binary:
        lam = data[index:index+nf]
        index += nf
    else:
        f.readline()
        lam = empty(nf)
        for i in range(nf):
            lam[i] = float(f.readline())
    
        f.readline()

    cdef numpy.ndarray[ndim=4, dtype=double] image
    if stokes:
        image = empty((nphi,nr+1,nf,4))
    else:
        image = empty((nphi,nr+1,nf,1))

    if binary:
        for i in range(nf):
            for j in range(nphi):
                for k in range(nr+1):
                    if stokes:
                        image[j,k,i,:] = data[index:index+4]
                        index += 4
                    else:
                        image[j,k,i,0] = data[index]
                        index += 1
    else:
        for i in range(nf):
            for j in range(nphi):
                for k in range(nr+1):
                    if stokes:
                        image[j,k,i,:] = array(f.readline().split(), \
                                dtype=float)
                    else:
                        image[j,k,i,0] = float(f.readline())

                    if (j == nphi-1) and (k == nr+1-1):
                        f.readline()

    f.close()

    # Compute the flux in this image as seen at 1 pc.

    if stokes:
        flux = image[:,:,:,0].sum(axis=0).sum(axis=0)
    else:
        flux = image.sum(axis=0).sum(axis=0)

    return image, rc, phic, lam

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

# Read in the dust temperature.

def amr_grid(filename=None, ext=None):

    f = open("amr_grid.inp","r")

    f.readline()
    f.readline()

    coordsystem = int(f.readline())
    gridinfo = int(f.readline())
    incl_x, incl_y, incl_z = tuple(f.readline().replace("\t","").\
            replace("\n","").split())
    nx, ny, nz = tuple(array(f.readline().replace("\t"," ").\
            replace("\n","").split(), dtype=int))

    x = array(f.readline().replace("\n","").split(), dtype=float)
    y = array(f.readline().replace("\n","").split(), dtype=float)
    z = array(f.readline().replace("\n","").split(), dtype=float)

    return coordsystem, gridinfo, incl_x, incl_y, incl_z, nx, ny, nz, x, y, z

# Read in the dust density.

def dust_density(filename=None, ext=None):

    f = open("dust_density.inp","r")

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

# Read in the dust temperature.

def dust_temperature(filename=None, ext=None, binary=False):

    if (filename == None):
        if (ext == None):
            if binary:
                filename = "dust_temperature.bdat"
            else:
                filename = "dust_temperature.dat"
        else:
            if binary:
                filename = "dust_temperature_"+str(ext)+".bdat"
            else:
                filename = "dust_temperature_"+str(ext)+".dat"

    if binary:
        f = open(filename, "rb")
        data = fromfile(filename)
    else:
        f = open(filename,"r")

    if binary:
        int.from_bytes(f.read(8), byteorder="little")
        int.from_bytes(f.read(8), byteorder="little")
        ncells = int.from_bytes(f.read(8), byteorder="little")
        nspecies = int.from_bytes(f.read(8), byteorder="little")
    else:
        f.readline()
        ncells = int(f.readline())
        nspecies = int(f.readline())

    temperature = []
    index = 3
    for i in range(nspecies):
        temp = empty((ncells,))

        for j in range(ncells):
            if binary:
                temp[j] = data[index+j]
            else:
                temp[j] = float(f.readline())

        temperature.append(temp)

    f.close()

    return temperature

# Read in the scattering phase function.

def scattering_phase(filename=None, binary=False):

    if filename == None:
        if binary:
            filename = "scattering_phase.bout"
        else:
            filename = "scattering_phase.out"

    if binary:
        f = open(filename, "rb")
        data = fromfile(filename)
    else:
        f = open(filename, "r")

    f.readline()
    ncells = int(f.readline())
    nfreq = int(f.readline())

    freq = array(f.readline().split(), dtype=float)

    scattering_phase = []
    for i in range(nfreq):
        temp = empty((ncells,))

        for j in range(ncells):
            temp[j] = float(f.readline())

        scattering_phase.append(temp)

    f.close()

    return freq, scattering_phase
