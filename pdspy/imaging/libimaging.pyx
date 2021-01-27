import numpy
cimport numpy
import astropy
import h5py
from ..constants.physics import c

cdef class ImageObject:
    cdef public numpy.ndarray image, x, y, unc, velocity, freq, wave
    cdef public header, wcs

    def __init__(self, numpy.ndarray[double, ndim=4] image=None, \
            numpy.ndarray[double, ndim=1] x=None, \
            numpy.ndarray[double, ndim=1] y=None, \
            numpy.ndarray[double, ndim=1] wave=None, \
            numpy.ndarray[double, ndim=1] freq=None, \
            numpy.ndarray[double, ndim=4] unc=None, \
            numpy.ndarray[double, ndim=1] velocity=None, \
            header=None, wcs = None):

        self.image = image

        self.x = x
        self.y = y

        self.header = header
        self.wcs = wcs

        self.unc = unc
        self.velocity = velocity

        if (type(wave) == type(None)) and (type(freq) != type(None)):
            self.freq = freq
            self.wave = c / freq
        elif (type(wave) != type(None)) and (type(freq) == type(None)):
            self.wave = wave
            self.freq = c / wave
        elif (type(wave) != type(None)) and (type(freq) != type(None)):
            self.wave = wave
            self.freq = freq

    def __reduce__(self):
        return (rebuild, (self.image, self.x, self.y, self.wave, self.freq, \
                self.unc, self.velocity, self.header, self.wcs))

def rebuild(image, x, y, wave, freq, unc, velocity, header, wcs):
    return ImageObject(image, x, y, wave, freq, unc, velocity, header, wcs)

class Image(ImageObject):

    def asFITS(self, channel=None):
        hdulist = astropy.io.fits.HDUList([])

        if channel != None:
            hdu = astropy.io.fits.PrimaryHDU(self.image[:,:,channel,0].astype(\
                    numpy.float32))

            if type(self.header) != type(None):
                hdu.header = self.header[channel]

            hdulist.append(hdu)

        else:
            hdu = astropy.io.fits.PrimaryHDU(numpy.transpose(\
                    self.image[:,:,:,0].astype(numpy.float32), axes=[2,1,0]))

            if type(self.header) != type(None):
                hdu.header = self.header

            hdulist.append(hdu)

        return hdulist

    def set_uncertainty_from_image(self, image, box=128):
        cdef unsigned int ny, nx, nfreq, npol
        cdef unsigned int xmin, xmax, ymin, ymax
        cdef unsigned int i, j
        cdef unsigned int halfbox
        cdef numpy.ndarray[double, ndim=2] subimage
        cdef numpy.ndarray[double, ndim=4] unc

        ny, nx, nfreq, npol = self.image.shape

        halfbox = box / 2

        unc = numpy.empty(self.image.shape)
        for i in range(ny):
            for j in range(nx):
                xmin = max(0,<int>j-64)
                xmax = min(<int>j+64,nx)
                ymin = max(0,<int>i-64)
                ymax = min(<int>i+64,ny)

                subimage = image.image[ymin:ymax,xmin:xmax,0,0]

                if (numpy.isnan(subimage).prod() == True):
                    unc[i,j,0,0] = subimage[0,0]
                else:
                    unc[i,j,0,0] = numpy.nanstd(subimage)

        self.unc = unc

    def read(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "r")
        else:
            f = usefile

        if ('x' in f):
            x = f['x'][...].astype(numpy.double)
            y = f['y'][...].astype(numpy.double)
        else:
            x = None
            y = None

        if ('freq' in f):
            freq = f['freq'][...].astype(numpy.double)
        else:
            freq = None

        image = f['image'][...].astype(numpy.double)

        if ('unc' in f):
            unc = f['unc'][...].astype(numpy.double)
        else:
            unc = None

        if ('wcs' in f.attrs):
            wcs = astropy.wcs.WCS(f.attrs['wcs'])
        else:
            wcs = None

        if ('header' in f.attrs):
            header = astropy.io.fits.Header().fromstring(f.attrs['header'])
        else:
            header = None

        self.__init__(image, x=x, y=y, unc=unc, freq=freq, wcs=wcs, \
                header=header)

        if (usefile == None):
            f.close()

    def write(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "w")
        else:
            f = usefile

        #if hasattr(self, "x"):
        if (type(self.x) != type(None)):
            x_dset = f.create_dataset("x", self.x.shape, dtype='f')
            x_dset[...] = self.x
            y_dset = f.create_dataset("y", self.y.shape, dtype='f')
            y_dset[...] = self.y

        #if hasattr(self, "freq"):
        if (type(self.freq) != type(None)):
            freq_dset = f.create_dataset("freq", self.freq.shape, dtype='f')
            freq_dset[...] = self.freq

        image_dset = f.create_dataset("image", self.image.shape, dtype='f')
        image_dset[...] = self.image

        #if hasattr(self, "unc"):
        if (type(self.unc) != type(None)):
            unc_dset = f.create_dataset("unc", self.unc.shape, \
                    dtype='f')
            unc_dset[...] = self.unc

        if hasattr(self, "wcs") and ((type(self.wcs) != type(None))):
            f.attrs['wcs'] = self.wcs.to_header_string()

        if hasattr(self, "header") and (type(self.header) != type(None)):
            f.attrs['header'] = self.header.tostring()

        if (usefile == None):
            f.close()

################################################################################
#
# Unstructured images.
#
################################################################################


cdef class UnstructuredImageObject:
    cdef public numpy.ndarray image, x, y, unc, velocity, freq, wave

    def __init__(self, numpy.ndarray[double, ndim=2] image=None, \
            numpy.ndarray[double, ndim=1] x=None, \
            numpy.ndarray[double, ndim=1] y=None, \
            numpy.ndarray[double, ndim=1] wave=None, \
            numpy.ndarray[double, ndim=1] freq=None, \
            numpy.ndarray[double, ndim=2] unc=None, \
            numpy.ndarray[double, ndim=1] velocity=None):

        self.image = image

        self.x = x
        self.y = y

        self.unc = unc
        self.velocity = velocity

        if (type(wave) == type(None)) and (type(freq) != type(None)):
            self.freq = freq
            self.wave = c / freq
        elif (type(wave) != type(None)) and (type(freq) == type(None)):
            self.wave = wave
            self.freq = c / wave
        elif (type(wave) != type(None)) and (type(freq) != type(None)):
            self.wave = wave
            self.freq = freq

    def __reduce__(self):
        return (rebuild_unstructured, (self.image, self.x, self.y, self.wave, \
                self.freq, self.unc, self.velocity))

def rebuild_unstructured(image, x, y, wave, freq, unc, velocity):
    return UnstructuredImageObject(image, x, y, wave, freq, unc, velocity)

class UnstructuredImage(UnstructuredImageObject):

    def read(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "r")
        else:
            f = usefile

        if ('x' in f):
            x = f['x'][...].astype(numpy.double)
            y = f['y'][...].astype(numpy.double)
        else:
            x = None
            y = None

        if ('freq' in f):
            freq = f['freq'][...].astype(numpy.double)
        else:
            freq = None

        image = f['image'][...].astype(numpy.double)

        if ('unc' in f):
            unc = f['unc'][...].astype(numpy.double)
        else:
            unc = None

        self.__init__(image, x=x, y=y, unc=unc, freq=freq)

        if (usefile == None):
            f.close()

    def write(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "w")
        else:
            f = usefile

        if (type(self.x) != type(None)):
            x_dset = f.create_dataset("x", self.x.shape, dtype='f')
            x_dset[...] = self.x
            y_dset = f.create_dataset("y", self.y.shape, dtype='f')
            y_dset[...] = self.y

        if (type(self.freq) != type(None)):
            freq_dset = f.create_dataset("freq", self.freq.shape, dtype='f')
            freq_dset[...] = self.freq

        image_dset = f.create_dataset("image", self.image.shape, dtype='f')
        image_dset[...] = self.image

        if (type(self.unc) != type(None)):
            unc_dset = f.create_dataset("unc", self.unc.shape, \
                    dtype='f')
            unc_dset[...] = self.unc

        if (usefile == None):
            f.close()
