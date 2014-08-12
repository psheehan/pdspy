#import numpy
cimport numpy

cdef class VisibilitiesObject:
    cdef public numpy.ndarray u, v, freq, real, imag, weights, uvdist, amp, phase
    cdef public numpy.ndarray baseline
    cdef public str array_name

    def __init__(self, numpy.ndarray[double, ndim=1] u=None, \
            numpy.ndarray[double, ndim=1] v=None, \
            numpy.ndarray[double, ndim=1] freq=None, \
            numpy.ndarray[double, ndim=2] real=None, \
            numpy.ndarray[double, ndim=2] imag=None, \
            numpy.ndarray[double, ndim=2] weights=None, \
            baseline=None, array_name="CARMA"):

        if ((u != None) and (v != None) and (freq != None) and (real != None) \
                and (imag != None) and (weights != None)):
            self.u = u
            self.v = v
            self.uvdist = numpy.sqrt(u**2 + v**2)

            self.freq = freq

            self.real = real
            self.imag = imag
            self.weights = weights
            self.amp = numpy.sqrt(real**2 + imag**2)
            self.phase = numpy.arctan2(imag, real)

        self.baseline = baseline
        self.array_name = array_name
