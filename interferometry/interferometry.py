import ctypes
import numpy
import h5py
import os
import astropy

lib = ctypes.cdll.LoadLibrary(os.path.dirname(__file__)+'/libinterferometry.so')
lib.new_Visibilities.restype = ctypes.c_void_p

class Visibilities:

    def __init__(self, u=None, v=None, freq=None, real=None, imag=None, \
            weights=None, baseline=None, array_name="CARMA"):
        if (u != None):
            self.u = u
            self.v = v
            self.uvdist = numpy.sqrt(u**2 + v**2)
        if (freq != None):
            self.freq = freq
        if (real != None):
            self.real = real
            self.imag = imag
            self.weights = weights
            self.amp = numpy.sqrt(real**2 + imag**2)
            self.phase = numpy.arctan2(imag, real)

        self.baseline = baseline
        self.array_name = array_name

        if ((u != None) & (freq != None) & (real != None)):
            self.obj = lib.new_Visibilities( \
                    u.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                    v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                    freq.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                    real.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                    imag.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                    weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                    self.uvdist.ctypes.data_as(ctypes.POINTER(\
                        ctypes.c_double)), \
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

    def read(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "r")
        else:
            f = usefile

        u = f['u'].value
        v = f['v'].value
        freq = f['freq'].value
        real = f['real'].value
        imag = f['imag'].value
        weights = f['weights'].value

        self.__init__(u, v, freq, real, imag, weights)

        if (usefile == None):
            f.close()

    def write(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "w")
        else:
            f = usefile

        u_dset = f.create_dataset("u", self.u.shape, dtype='f')
        u_dset[...] = self.u
        v_dset = f.create_dataset("v", self.v.shape, dtype='f')
        v_dset[...] = self.v
        freq_dset = f.create_dataset("freq", self.freq.shape, dtype='f')
        freq_dset[...] = self.freq
        real_dset = f.create_dataset("real", self.real.shape, dtype='f')
        real_dset[...] = self.real
        imag_dset = f.create_dataset("imag", self.imag.shape, dtype='f')
        imag_dset[...] = self.imag
        weights_dset = f.create_dataset("weights", self.weights.shape, \
                dtype='f')
        weights_dset[...] = self.weights

        if (usefile == None):
            f.close()
