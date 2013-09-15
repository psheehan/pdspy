import ctypes
import numpy
import os
import astropy

lib = ctypes.cdll.LoadLibrary(os.path.dirname(__file__)+'/libinterferometry.so')
lib.new_Visibilities.restype = ctypes.c_void_p

class Visibilities:

    def __init__(self, u, v, freq, real, imag, weights, baseline=None, \
            array_name="CARMA"):
        self.u = u
        self.v = v
        self.freq = freq
        self.real = real
        self.imag = imag
        self.weights = weights

        self.uvdist = numpy.sqrt(u**2 + v**2)
        self.amp = numpy.sqrt(real**2 + imag**2)
        self.phase = numpy.arctan2(imag, real)

        self.baseline = baseline
        self.array_name = array_name

        self.obj = lib.new_Visibilities( \
                u.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                freq.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                real.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                imag.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                self.uvdist.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                self.amp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                self.phase.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                ctypes.c_int(u.size), ctypes.c_int(freq.size))

    def __del__(self):
        lib.delete_Visibilities(ctypes.c_void_p(self.obj))

    def get_baselines(self, num):
        include = self.baseline == num

        return Visibilities(self.u[include], self.v[include], self.freq, \
                self.real[include,:], self.imag[include,:], \
                self.weights[include,:], baseline=self.baseline[include])

    def set_header(self, header):
        self.header = header

    def asFITS(self):
        hdulist = astropy.fits.HDUList([])

        nvis = self.u.size
        hdu = astropy.fits.PrimaryHDU(numpy.concatenate((self.u.reshape((1,nvis)), \
                self.v.reshape((1,nvis)),self.real.reshape((1,nvis)), \
                self.imag.reshape((1,nvis)), \
                self.weights.reshape((1,nvis))), axis=0))

        if self.header != None:
            hdu.header = self.header

        hdulist.append(hdu)

        return hdulist
