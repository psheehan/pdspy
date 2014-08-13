import numpy
import astropy
import h5py
from ..constants.physics import c

class Image:

    def __init__(self,image=None,x=None,y=None,header=None,wave=None, \
            freq=None,unc=None,velocity=None):

        if (image != None):
            self.image = image

        if (x != None):
            self.x = x
            self.y = y
        if (header != None):
            self.header = header
        if (unc != None):
            self.unc = unc
        if (velocity != None):
            self.velocity = velocity

        if (wave == None) and (freq != None):
            self.freq = freq
            self.wave = c / freq
        elif (wave != None) and (freq == None):
            self.wave = wave
            self.freq = c / wave
        elif (wave != None) and (freq != None):
            self.wave = wave
            self.freq = freq

    def asFITS(self):
        hdulist = astropy.io.fits.HDUList([])
        for i in range(self.image[0,0,:].size):
            hdu = astropy.io.fits.PrimaryHDU(self.image[:,:,i])

            if hasattr(self, "header"):
                hdu.header = self.header[i]

            hdulist.append(hdu)

        return hdulist

    def read(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "r")
        else:
            f = usefile

        if ('x' in f):
            x = f['x'].value
            y = f['y'].value
        else:
            x = None
            y = None

        if ('freq' in f):
            freq = f['freq'].value
        else:
            freq = None

        image = f['image'].value

        if ('unc' in f):
            unc = f['unc'].value
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

        if hasattr(self, "x"):
            x_dset = f.create_dataset("x", (self.x.size,), dtype='f')
            x_dset[...] = self.x
            y_dset = f.create_dataset("y", (self.y.size,), dtype='f')
            y_dset[...] = self.y

        if hasattr(self, "freq"):
            freq_dset = f.create_dataset("freq", (self.freq.size,), dtype='f')
            freq_dset[...] = self.freq

        image_dset = f.create_dataset("image", self.image.shape, dtype='f')
        image_dset[...] = self.image

        if hasattr(self, "unc"):
            unc_dset = f.create_dataset("uncertainty", self.unc.shape, \
                    dtype='f')
            unc_dset[...] = self.unc

        if (usefile == None):
            f.close()
