import numpy
cimport numpy
import astropy
import h5py
from ..constants.physics import c

cdef class ImageObject:
    cdef public numpy.ndarray image, x, y, unc, velocity, freq, wave

    def __init__(self, numpy.ndarray[double, ndim=4] image=None, \
            numpy.ndarray[double, ndim=1] x=None, \
            numpy.ndarray[double, ndim=1] y=None, \
            numpy.ndarray[double, ndim=1] wave=None, \
            numpy.ndarray[double, ndim=1] freq=None, \
            numpy.ndarray[double, ndim=4] unc=None, \
            numpy.ndarray[double, ndim=1] velocity=None, \
            header=None, wcs = None):

        if (type(image) != type(None)):
            self.image = image

        if (type(x) != type(None)):
            self.x = x
            self.y = y
        if (type(header) != type(None)):
            self.header = header
        if (type(wcs) != type(None)):
            self.wcs = wcs
        if (type(unc) != type(None)):
            self.unc = unc
        if (type(velocity) != type(None)):
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

class Image(ImageObject):

    def asFITS(self):
        hdulist = astropy.io.fits.HDUList([])
        for i in range(self.image[0,0,:,0].size):
            hdu = astropy.io.fits.PrimaryHDU(self.image[:,:,i,0].astype(\
                    numpy.float32))

            if hasattr(self, "header"):
                hdu.header = self.header[i]

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
            x = f['x'].value.astype(numpy.double)
            y = f['y'].value.astype(numpy.double)
        else:
            x = None
            y = None

        if ('freq' in f):
            freq = f['freq'].value.astype(numpy.double)
        else:
            freq = None

        image = f['image'].value.astype(numpy.double)

        if ('unc' in f):
            unc = f['unc'].value.astype(numpy.double)
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
