import numpy
cimport numpy
import h5py
import astropy

cdef class VisibilitiesObject:
    cdef public numpy.ndarray u, v, freq, real, imag, weights, \
            uvdist, amp, phase
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


class Visibilities(VisibilitiesObject):

    def get_baselines(self, num):
        incl = self.baseline == num

        return Visibilities(self.u[incl], self.v[incl], self.freq, \
                self.real[incl,:], self.imag[incl,:], \
                self.weights[incl,:], baseline=self.baseline[incl])

    def set_header(self, header):
        self.header = header

    def asFITS(self):
        hdulist = astropy.fits.HDUList([])

        nvis = self.u.size
        hdu = astropy.fits.PrimaryHDU(numpy.concatenate((\
                self.u.reshape((1,nvis)), self.v.reshape((1,nvis)), \
                self.real.reshape((1,nvis)), self.imag.reshape((1,nvis)), \
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

        u = f['u'].value.astype(numpy.double)
        v = f['v'].value.astype(numpy.double)
        freq = f['freq'].value.astype(numpy.double)
        real = f['real'].value.astype(numpy.double)
        imag = f['imag'].value.astype(numpy.double)
        weights = f['weights'].value.astype(numpy.double)

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

def average(data, gridsize=256, binsize=None, radial=False):

    cdef numpy.ndarray[double, ndim=2] new_real, new_imag, new_weights
    cdef numpy.ndarray[unsigned int, ndim=1] i, j
    cdef unsigned int k, l
    
    cdef numpy.ndarray[double, ndim=1] u = data.u.copy()
    cdef numpy.ndarray[double, ndim=1] v = data.v.copy()
    cdef numpy.ndarray[double, ndim=1] uvdist = data.uvdist.copy()
    cdef numpy.ndarray[double, ndim=2] real = data.real.copy()
    cdef numpy.ndarray[double, ndim=2] imag = data.imag.copy()
    cdef numpy.ndarray[double, ndim=2] weights = data.weights.copy()
    
    # Set the weights equal to 0 when the point is flagged (i.e. weight < 0)
    weights = numpy.where(weights < 0,0.0,weights)
    # Set the weights equal to 0 when the real and imaginary parts are both 0
    weights[(real == 0) & (imag == 0)] = 0.0
    
    # Average over the U-V plane by creating bins to average over.
    
    if radial:
        new_u = numpy.linspace(binsize/2,(gridsize-0.5)*binsize,gridsize)
        new_u = new_u.reshape((1,gridsize))
        new_v = numpy.zeros((1,gridsize))
        new_real = numpy.zeros((1,gridsize))
        new_imag = numpy.zeros((1,gridsize))
        new_weights = numpy.zeros((1,gridsize))

        i = numpy.round(uvdist/binsize).astype(numpy.uint32)
        j = numpy.zeros(uvdist.size).astype(numpy.uint32)
    else:
        if gridsize%2 == 0:
            uu = numpy.linspace(-gridsize*binsize/2, (gridsize/2-1)*binsize, \
                    gridsize)
            vv = numpy.linspace(-gridsize*binsize/2, (gridsize/2-1)*binsize, \
                    gridsize)
        else:
            uu = numpy.linspace(-(gridsize-1)*binsize/2, \
                    (gridsize-1)*binsize/2, gridsize)
            vv = numpy.linspace(-(gridsize-1)*binsize/2, \
                    (gridsize-1)*binsize/2, gridsize)

        new_u, new_v = numpy.meshgrid(uu, vv)
        new_real = numpy.zeros((gridsize,gridsize))
        new_imag = numpy.zeros((gridsize,gridsize))
        new_weights = numpy.zeros((gridsize,gridsize))

        if gridsize%2 == 0:
            i = numpy.round(u/binsize+gridsize/2.).astype(numpy.uint32)
            j = numpy.round(v/binsize+gridsize/2.).astype(numpy.uint32)
        else:
            i = numpy.round(u/binsize+(gridsize-1)/2.).astype(numpy.uint32)
            j = numpy.round(v/binsize+(gridsize-1)/2.).astype(numpy.uint32)
    
    cdef int nuv = u.size
    cdef int nfreq = data.freq.size

    for k in range(nuv):
        for l in range(nfreq):
            new_real[j[k],i[k]] += (real[k,l]*weights[k,l])
            new_imag[j[k],i[k]] += (imag[k,l]*weights[k,l])
            new_weights[j[k],i[k]] += weights[k,l]
    
    good_data = new_weights != 0.0
    new_u = new_u[good_data]
    new_v = new_v[good_data]
    new_real = (new_real[good_data] / new_weights[good_data]).reshape( \
            (new_u.size,1))
    new_imag = (new_imag[good_data] / new_weights[good_data]).reshape( \
            (new_u.size,1))
    new_weights = new_weights[good_data].reshape((new_u.size,1))
    
    freq = numpy.array([data.freq.sum()/data.freq.size])
    
    return Visibilities(new_u,new_v,freq,new_real,new_imag,new_weights)
