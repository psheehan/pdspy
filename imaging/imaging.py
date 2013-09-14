import ctypes
import numpy
import os
import pyfits
from ..constants.physics import c

lib = ctypes.cdll.LoadLibrary(os.environ["HOME"]+ \
        '/Documents/Programs/Python_Programs/pdspy/imaging/libimaging.so')
lib.new_Image.restype = ctypes.c_void_p

class Image:

    def __init__(self,image,x=None,y=None,header=None,wave=None,freq=None, \
            unit=None,RA=None,Dec=None,unc=None,velocity=None):

        self.image = image
        self.x = x
        self.y = y
        self.RA = RA
        self.Dec = Dec
        self.header = header
        self.unit = unit
        self.unc = unc
        self.velocity = velocity

        if (wave == None) and (freq != None):
            self.freq = freq
            self.wave = c / freq
        elif (wave != None) and (freq == None):
            self.wave = wave
            self.freq = c / wave
        else:
            self.wave = wave
            self.freq = freq

        self.obj = lib.new_Image( \
                image.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                ctypes.c_int(image.shape[1]), ctypes.c_int(image.shape[0]), \
                ctypes.c_int(image.shape[2]))

        if (x != None) and (y != None):
            lib.set_xy(ctypes.c_void_p(self.obj), \
                    x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                    y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

        if (freq != None):
            lib.set_freq(ctypes.c_void_p(self.obj), \
                    freq.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                    wave.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

        if (unc != None):
            lib.set_unc(ctypes.c_void_p(self.obj), \
                    unc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    def __del__(self):
        lib.delete_Image(ctypes.c_void_p(self.obj))

    def asFITS(self):
        hdulist = pyfits.HDUList([])
        for i in range(self.image[0,0,:].size):
            hdu = pyfits.PrimaryHDU(self.image[:,:,i])

            if self.header != None:
                hdu.header = self.header[i]

            hdulist.append(hdu)

        return hdulist
