import numpy
import astropy
import h5py
from ..constants.physics import c

class Spectrum:

    def __init__(self, wave=None, flux=None, unc=None):
        if (type(wave) != type(None)):
            self.wave = wave
            self.freq = c / wave / 1.0e-4
            self.flux = flux
            if type(unc) == type(None):
                unc = numpy.zeros(wave.size)
                self.unc = unc
            else:
                self.unc = unc

    def asFITS(self):
        hdulist = astropy.io.fits.HDUList([])

        hdu = pyfits.PrimaryHDU(numpy.concatenate(( \
                self.wave.reshape((1,self.wave.size)), \
                self.flux.reshape((1,self.wave.size)), \
                self.unc.reshape((1,self.wave.size)))))
        hdulist.append(hdu)

        return hdulist

    def read(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "r")
        else:
            f = usefile

        wave = f['wave'][...]
        flux = f['flux'][...]
        unc = f['unc'][...]

        self.__init__(wave, flux, unc)

        if (usefile == None):
            f.close()

    def write(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "w")
        else:
            f = usefile

        if hasattr(self, "wave"):
            wave_dset = f.create_dataset("wave", (self.wave.size,), dtype='f')
            wave_dset[...] = self.wave
            flux_dset = f.create_dataset("flux", (self.flux.size,), dtype='f')
            flux_dset[...] = self.flux
            unc_dset = f.create_dataset("unc", (self.unc.size,), dtype='f')
            unc_dset[...] = self.unc

        if (usefile == None):
            f.close()
