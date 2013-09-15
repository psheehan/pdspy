import ctypes
import numpy
import os
import astropy

lib = ctypes.cdll.LoadLibrary(os.path.dirname(__file__)+'/libspectroscopy.so')
lib.new_Spectrum.restype = ctypes.c_void_p

class Spectrum:

    def __init__(self, wave, flux, unc):
        self.wave = wave
        self.flux = flux
        self.unc = unc

        self.obj = lib.new_Spectrum( \
                wave.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                flux.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                unc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                ctypes.c_int(wave.size))

    def __del__(self):
        lib.delete_Spectrum(ctypes.c_void_p(self.obj))

    def asFITS(self):
        hdulist = astropy.io.fits.HDUList([])

        hdu = pyfits.PrimaryHDU(numpy.concatenate(( \
                self.wave.reshape((1,self.wave.size)), \
                self.flux.reshape((1,self.wave.size)), \
                self.unc.reshape((1,self.wave.size)))))
        hdulist.append(hdu)

        return hdulist
