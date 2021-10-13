import numpy
cimport numpy
import h5py
import astropy
cimport cython
import time
from libc.math cimport pi

@cython.auto_pickle(True)
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

        if (type(u) != type(None)) and (type(v) != type(None)):
            self.u = u
            self.v = v
            self.uvdist = numpy.sqrt(u**2 + v**2)

        if (type(freq) != type(None)):
            self.freq = freq

        if (type(real) != type(None)) and (type(imag) != type(None)):
            self.real = real
            self.imag = imag
            self.amp = numpy.sqrt(real**2 + imag**2)
            self.phase = numpy.arctan2(imag, real)

            if type(weights) != type(None):
                self.weights = weights
            else:
                self.weights = numpy.ones((self.real.shape[0],self.real.shape[1]))

        self.baseline = baseline
        self.array_name = array_name

    def __reduce__(self):
        return (rebuild, (self.u, self.v, self.freq, self.real, self.imag, \
                self.weights, self.baseline, self.array_name))

def rebuild(u, v, freq, real, imag, weights, baseline, array_name):
    return VisibilitiesObject(u, v, freq, real, imag, weights, baseline, \
            array_name)

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

        if type(self.header) != type(None):
            hdu.header = self.header

        hdulist.append(hdu)

        return hdulist

    """
    def __reduce__(self):
        return (self.rebuild, (self.u, self.v, self.freq, self.real, self.imag,\
                self.weights))

    def rebuild(u, v, freq, real, imag, weights):
        return Visibilities(u, v, freq, real, imag, weights)
    """

    def read(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "r")
        else:
            f = usefile

        u, v, freq, real, imag, weights = None, None, None, None, None, None

        if ('u' in f) and ('v' in f):
            u = f['u'][...].astype(numpy.double)
            v = f['v'][...].astype(numpy.double)
        if ('freq' in f):
            freq = f['freq'][...].astype(numpy.double)
        if ('real' in f) and ('imag' in f):
            real = f['real'][...].astype(numpy.double)
            imag = f['imag'][...].astype(numpy.double)

            if ('weights' in f):
                weights = f['weights'][...].astype(numpy.double)
            else:
                weights = numpy.ones(real.shape)

        self.__init__(u, v, freq, real, imag, weights)

        if (usefile == None):
            f.close()

    def write(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "w")
        else:
            f = usefile

        if (type(self.u) != type(None)) and (type(self.v) != type(None)):
            u_dset = f.create_dataset("u", self.u.shape, dtype='float64')
            u_dset[...] = self.u
            v_dset = f.create_dataset("v", self.v.shape, dtype='float64')
            v_dset[...] = self.v
        if (type(self.freq) != type(None)):
            freq_dset = f.create_dataset("freq", self.freq.shape, \
                    dtype='float64')
            freq_dset[...] = self.freq
        if (type(self.real) != type(None)) and (type(self.imag) != type(None)):
            real_dset = f.create_dataset("real", self.real.shape, \
                    dtype='float64')
            real_dset[...] = self.real
            imag_dset = f.create_dataset("imag", self.imag.shape, \
                    dtype='float64')
            imag_dset[...] = self.imag

            if type(self.weights) != type(None):
                if numpy.product(self.weights == numpy.ones(self.real.shape)) \
                        == 0:
                    weights_dset = f.create_dataset("weights", \
                            self.weights.shape, dtype='float64')
                    weights_dset[...] = self.weights

        if (usefile == None):
            f.close()

@cython.boundscheck(False)
def average(data, gridsize=256, binsize=None, radial=False, log=False, \
        logmin=None, logmax=None, mfs=False, mode="continuum"):

    cdef numpy.ndarray[double, ndim=1] u, v, freq, uvdist
    cdef numpy.ndarray[double, ndim=2] real, imag, weights
    cdef numpy.ndarray[double, ndim=3] new_real, new_imag, new_weights
    cdef numpy.ndarray[unsigned int, ndim=1] i, j
    cdef unsigned int k, n
    
    if mfs:
        vis = freqcorrect(data)
        u = vis.u
        v = vis.v
        uvdist = vis.uvdist
        freq = vis.freq
        real = vis.real
        imag = vis.imag
        weights = vis.weights
    else:
        u = data.u.copy()
        v = data.v.copy()
        uvdist = data.uvdist.copy()
        freq = data.freq.copy()
        real = data.real.copy()
        imag = data.imag.copy()
        weights = data.weights
    
    # Set the weights equal to 0 when the point is flagged (i.e. weight < 0)
    weights = numpy.where(weights < 0,0.0,weights)
    # Set the weights equal to 0 when the real and imaginary parts are both 0
    weights[(real == 0) & (imag == 0)] = 0.0
    
    good_data = uvdist != 0.0
    u = u[good_data]
    v = v[good_data]
    uvdist = uvdist[good_data]
    real = real[good_data,:]
    imag = imag[good_data,:]
    weights = weights[good_data,:]

    # Set some parameter numbers for future use.

    cdef int nuv = u.size
    cdef int nfreq = freq.size
    
    cdef int nchannels
    if mode == "continuum":
        nchannels = 1
    elif mode == "spectralline":
        nchannels = freq.size

    # Average over the U-V plane by creating bins to average over.
    
    if radial:
        if log:
            temp = numpy.linspace(numpy.log10(logmin), numpy.log10(logmax), \
                    gridsize+1)
            new_u = 10**((temp[1:] + temp[0:-1])/2)
        else:
            new_u = numpy.linspace(binsize/2,(gridsize-0.5)*binsize,gridsize)
        new_u = new_u.reshape((1,gridsize))
        new_v = numpy.zeros((1,gridsize))
        new_real = numpy.zeros((1,gridsize,nchannels))
        new_imag = numpy.zeros((1,gridsize,nchannels))
        new_weights = numpy.zeros((1,gridsize,nchannels))

        if log:
            dtemp = temp[1] - temp[0]
            i = numpy.round((numpy.log10(uvdist)- numpy.log10(logmin))/dtemp - \
                    0.5).astype(numpy.uint32)
            j = numpy.zeros(uvdist.size).astype(numpy.uint32)
        else:
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

        #new_u, new_v = numpy.meshgrid(uu, vv)
        new_u = numpy.zeros((gridsize,gridsize,nchannels))
        new_v = numpy.zeros((gridsize,gridsize,nchannels))
        new_real = numpy.zeros((gridsize,gridsize,nchannels))
        new_imag = numpy.zeros((gridsize,gridsize,nchannels))
        new_weights = numpy.zeros((gridsize,gridsize,nchannels))

        if gridsize%2 == 0:
            i = numpy.round(u/binsize+gridsize/2.).astype(numpy.uint32)
            j = numpy.round(v/binsize+gridsize/2.).astype(numpy.uint32)
        else:
            i = numpy.round(u/binsize+(gridsize-1)/2.).astype(numpy.uint32)
            j = numpy.round(v/binsize+(gridsize-1)/2.).astype(numpy.uint32)

    good_i = numpy.logical_and(i >= 0, i < gridsize)
    good_j = numpy.logical_and(j >= 0, j < gridsize)
    good = numpy.logical_and(good_i, good_j)
    if good.sum() < good.size:
        print("WARNING: uv.grid was supplied with a gridsize and binsize that do not cover the full range of the input data in the uv-plane and is cutting baselines that are outside of this grid. Make sure to check your results carefully.")

    u = u[good]
    v = v[good]
    real = real[good,:]
    imag = imag[good,:]
    weights = weights[good,:]

    i = i[good]
    j = j[good]

    nuv = u.size

    for k in range(nuv):
        for n in range(nfreq):
            if mode == "continuum":
                if not radial:
                    new_u[j[k],i[k],0] += u[k]*weights[k,n]
                    new_v[j[k],i[k],0] += v[k]*weights[k,n]
                new_real[j[k],i[k],0] += real[k,n]*weights[k,n]
                new_imag[j[k],i[k],0] += imag[k,n]*weights[k,n]
                new_weights[j[k],i[k],0] += weights[k,n]
            elif mode == "spectralline":
                if not radial:
                    new_u[j[k],i[k],n] += u[k]*weights[k,n]
                    new_v[j[k],i[k],n] += v[k]*weights[k,n]
                new_real[j[k],i[k],n] += real[k,n]*weights[k,n]
                new_imag[j[k],i[k],n] += imag[k,n]*weights[k,n]
                new_weights[j[k],i[k],n] += weights[k,n]

    good_data = new_weights != 0.0
    new_real[good_data] = new_real[good_data] / new_weights[good_data]
    new_imag[good_data] = new_imag[good_data] / new_weights[good_data]
    if not radial:
        new_u[good_data] = new_u[good_data] / new_weights[good_data]
        new_v[good_data] = new_v[good_data] / new_weights[good_data]

    good_data = numpy.any(good_data, axis=2)
    if not radial:
        new_u = (new_u*new_weights).sum(axis=2)[good_data] / \
                new_weights.sum(axis=2)[good_data]
        new_v = (new_v*new_weights).sum(axis=2)[good_data] / \
                new_weights.sum(axis=2)[good_data]
    else:
        new_u = new_u[good_data]
        new_v = new_v[good_data]

    good_data = numpy.dstack([good_data for m in range(nchannels)])

    if mode == "continuum":
        freq = numpy.array([data.freq.sum()/data.freq.size])

    return Visibilities(new_u, new_v, freq, \
            new_real[good_data].reshape((new_u.size,nchannels)),\
            new_imag[good_data].reshape((new_u.size,nchannels)), \
            new_weights[good_data].reshape((new_u.size,nchannels)))

@cython.boundscheck(False)
def grid(data, gridsize=256, binsize=2000.0, convolution="pillbox", \
        mfs=False, channel=None, imaging=False, weighting="natural", \
        robust=2, npixels=0, mode="continuum"):
    
    cdef numpy.ndarray[double, ndim=1] u, v, freq
    cdef numpy.ndarray[double, ndim=2] real, imag, weights, new_u, new_v
    cdef numpy.ndarray[double, ndim=3] new_real, new_imag, new_weights
    cdef numpy.ndarray[double, ndim=3] binned_weights
    cdef numpy.ndarray[unsigned int, ndim=2] i, j
    cdef unsigned int k, l, m, n, ninclude_min, ninclude_max, lmin, lmax, \
            mmin, mmax, ll, mm, ninclude, npix
    cdef double mean_freq, inv_freq

    if mfs:
        vis = freqcorrect(data)
        u = vis.u
        v = vis.v
        freq = vis.freq
        real = vis.real
        imag = vis.imag
        weights = vis.weights
    else:
        u = data.u
        v = data.v
        if channel != None:
            freq = numpy.array([data.freq[channel]])
            real = data.real[:,channel].reshape((data.real.shape[0],1))
            imag = data.imag[:,channel].reshape((data.real.shape[0],1))
            weights = data.weights[:,channel]. \
                    reshape((data.real.shape[0],1))
        else:
            freq = data.freq
            real = data.real
            imag = data.imag
            weights = data.weights
    
    # Set the weights equal to 0 when the point is flagged (i.e. weight < 0)
    weights = numpy.where(weights < 0, 0.0, weights)
    # Set the weights equal to 0 when the real and imaginary parts are both 0
    weights[(real==0) & (imag==0)] = 0.0

    # Set some parameter numbers for future use.
    
    cdef double convolve
    cdef double inv_binsize = 1. / binsize
    cdef int nuv = u.size
    cdef int nfreq = freq.size

    cdef int nchannels
    if mode == "continuum":
        nchannels = 1
    elif mode == "spectralline":
        nchannels = freq.size

    # Average over the U-V plane by creating bins to average over.
    
    if gridsize%2 == 0:
        uu = numpy.linspace(-gridsize*binsize/2, (gridsize/2-1)*binsize, \
                gridsize)
        vv = numpy.linspace(-gridsize*binsize/2, (gridsize/2-1)*binsize, \
                gridsize)
    else:
        uu = numpy.linspace(-(gridsize-1)*binsize/2, (gridsize-1)*binsize/2, \
                gridsize)
        vv = numpy.linspace(-(gridsize-1)*binsize/2, (gridsize-1)*binsize/2, \
                gridsize)

    new_u, new_v = numpy.meshgrid(uu, vv)
    new_real = numpy.zeros((gridsize,gridsize,nchannels))
    new_imag = numpy.zeros((gridsize,gridsize,nchannels))
    new_weights = numpy.zeros((gridsize,gridsize,nchannels))

    # Get the indices for binning.

    i = numpy.zeros((nuv, nfreq), dtype=numpy.uint32)
    j = numpy.zeros((nuv, nfreq), dtype=numpy.uint32)

    mean_freq = numpy.mean(freq)
    inv_freq = 1./mean_freq

    if gridsize%2 == 0:
        i = numpy.array(u.reshape((u.size,1))*freq*inv_freq/binsize+\
                gridsize/2., dtype=numpy.uint32)
        j = numpy.array(v.reshape((v.size,1))*freq*inv_freq/binsize+\
                gridsize/2., dtype=numpy.uint32)
    else:
        i = numpy.array(u.reshape((u.size,1))*freq*inv_freq/binsize+ \
                    (gridsize-1)/2., dtype=numpy.uint32)
        j = numpy.array(v.reshape((v.size,1))*freq*inv_freq/binsize+ \
                    (gridsize-1)/2., dtype=numpy.uint32)
    
    if convolution == "pillbox":
        convolve_func = ones
        ninclude = 3
    elif convolution == "expsinc":
        convolve_func = exp_sinc
        ninclude = 6

    if ninclude%2 == 0:
        ninclude_min = numpy.uint32(ninclude*0.5-1)
        ninclude_max = numpy.uint32(ninclude*0.5)
    else:
        ninclude_min = numpy.uint32((ninclude-1)*0.5)
        ninclude_max = numpy.uint32((ninclude-1)*0.5)

    # Check whether any data falls outside of the grid, and exclude.

    good_i = numpy.logical_and(i >= 0, i < gridsize)
    good_j = numpy.logical_and(j >= 0, j < gridsize)
    good = numpy.logical_and(good_i, good_j)
    if good.sum() < good.size:
        print("WARNING: uv.grid was supplied with a gridsize and binsize that do not cover the full range of the input data in the uv-plane and is cutting baselines that are outside of this grid. Make sure to check your results carefully.")

    # If we are using a non-uniform weighting scheme, adjust the data weights.

    if weighting in ["uniform","superuniform","robust"]:
        binned_weights = numpy.ones((gridsize,gridsize,nchannels))

        npix = npixels

        if weighting == "superuniform":
            npix = 3

        for k in range(nuv):
            for n in range(nfreq):
                if not good[k,n]:
                    continue

                if npix > j[k,n]:
                    lmin = 0
                else:
                    lmin = j[k,n] - npix

                lmax = int_min(j[k,n]+npix+1, gridsize)

                if npix > i[k,n]:
                    mmin = 0
                else:
                    mmin = i[k,n] - npix

                mmax = int_min(i[k,n]+npix+1, gridsize)

                for l in range(lmin, lmax):
                    for m in range(mmin, mmax):
                        if mode == "continuum":
                            binned_weights[l,m,0] += weights[k,n]
                        elif mode == "spectralline":
                            binned_weights[l,m,n] += weights[k,n]

        if weighting in ["uniform","superuniform"]:
            for k in range(nuv):
                for n in range(nfreq):
                    if not good[k,n]:
                        continue

                    l = j[k,n]
                    m = i[k,n]

                    weights[k,n] /= binned_weights[l,m,n]
        elif weighting == "robust":
            f2 = (5*10**(-robust))**2 / \
                    ((binned_weights**2).sum(axis=(0,1)) / weights.sum(axis=0))

            for k in range(nuv):
                for n in range(nfreq):
                    if not good[k,n]:
                        continue

                    l = j[k,n]
                    m = i[k,n]

                    weights[k,n] /= (1 + f2[n] * binned_weights[l,m,n])

    # Now actually go through and calculate the new visibilities.

    for k in range(nuv):
        for n in range(nfreq):
            if not good[k,n]:
                continue

            if ninclude_min > j[k,n]:
                lmin = 0
            else:
                lmin = j[k,n] - ninclude_min

            lmax = int_min(j[k,n]+ninclude_max+1, gridsize)

            if ninclude_min > i[k,n]:
                mmin = 0
            else:
                mmin = i[k,n] - ninclude_min

            mmax = int_min(i[k,n]+ninclude_max+1, gridsize)

            for l in range(lmin, lmax):
                for m in range(mmin, mmax):
                    convolve = convolve_func( (u[k]*freq[n]*inv_freq-\
                            new_u[l,m])*inv_binsize, \
                            (v[k]*freq[n]*inv_freq - new_v[l,m]) * inv_binsize)

                    if mode == "continuum":
                        new_real[l,m,0] += real[k,n]*weights[k,n]*convolve
                        new_imag[l,m,0] += imag[k,n]*weights[k,n]*convolve
                        new_weights[l,m,0] += weights[k,n]*convolve
                    elif mode == "spectralline":
                        new_real[l,m,n] += real[k,n]*weights[k,n]*convolve
                        new_imag[l,m,n] += imag[k,n]*weights[k,n]*convolve
                        new_weights[l,m,n] += weights[k,n]*convolve

    # If we are making an image, normalize the weights.

    if imaging:
        for n in range(nchannels):
            new_real[:,:,n] /= new_weights[:,:,n].sum()
            new_imag[:,:,n] /= new_weights[:,:,n].sum()
            new_weights[:,:,n] /= new_weights[:,:,n].sum()
    else:
        good = new_weights > 0
        new_real[good] = new_real[good] / new_weights[good]
        new_imag[good] = new_imag[good] / new_weights[good]

    if mode == "continuum":
        freq = numpy.array([data.freq.sum()/data.freq.size])
    
    return Visibilities(new_u.reshape(gridsize**2), new_v.reshape(gridsize**2),\
            freq, new_real.reshape((gridsize**2,nchannels)), \
            new_imag.reshape((gridsize**2,nchannels)), \
            new_weights.reshape((gridsize**2,nchannels)))

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline double int_abs(double a): return -a if a < 0 else a

cdef double sinc(double x):

    cdef double xp = x * pi

    return 1. - xp**2/6. + xp**4/120. - xp**6/5040. + xp**8/362880. - \
            xp**10/39916800. + xp**12/6227020800. - xp**14/1307674368000. + \
            xp**16/355687428096000.

cdef double exp(double x):

    return 1 + x + x**2/2. + x**3/6. + x**4/24. + x**5/120.

cdef double exp_sinc(double u, double v):
    
    cdef double inv_alpha1 = 1. / 1.55
    cdef double inv_alpha2 = 1. / 2.52
    cdef double norm = 2.350016262343186
    cdef int m = 6
    
    if (int_abs(u) >= m * 0.5) or (int_abs(v) >= m * 0.5):
        return 0.

    cdef double arr = sinc(u * inv_alpha1) * \
            sinc(v * inv_alpha1) * \
            exp(-1 * (u * inv_alpha2)**2) * \
            exp(-1 * (v * inv_alpha2)**2) / norm

    return arr

cdef double ones(double u, double v):
    
    cdef int m = 1

    if (int_abs(u) >= m * 0.5) or (int_abs(v) >= m * 0.5):
        return 0.

    cdef double arr = 1.0

    return arr

@cython.boundscheck(False)
def freqcorrect(data, freq=None):
    cdef numpy.ndarray[double, ndim=1] new_u, new_v, new_freq
    cdef numpy.ndarray[double, ndim=2] new_real, new_imag, new_weights

    if freq != None:
        new_freq = numpy.array([freq])
    else:
        new_freq = numpy.array([data.freq.mean()])

    cdef double inv_freq = 1./new_freq[0]
    cdef numpy.ndarray[double, ndim=1] scale = data.freq * inv_freq

    new_u = (data.u.reshape((data.u.size,1))*scale).reshape((data.real.size,))
    new_v = (data.v.reshape((data.v.size,1))*scale).reshape((data.real.size,))

    new_real = data.real.reshape((data.real.size,1))
    new_imag = data.imag.reshape((data.imag.size,1))
    new_weights = data.weights.reshape((data.weights.size,1))

    return Visibilities(new_u, new_v, new_freq, new_real, new_imag, \
            new_weights)

def chisq(data, model):
    chi_squared = chisq_calc(data.real, data.imag, data.weights, model.real, \
            model.imag, data.real.size)

    return chi_squared

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef float chisq_calc(numpy.ndarray[double, ndim=2] data_real, \
        numpy.ndarray[double, ndim=2] data_imag, \
        numpy.ndarray[double, ndim=2] data_weights, \
        numpy.ndarray[double, ndim=2] model_real, \
        numpy.ndarray[double, ndim=2] model_imag, int nuv):
    cdef double chisq = 0
    cdef double diff1, diff2
    cdef unsigned int i

    for i in range(nuv):
        diff1 = data_real[<unsigned int>i,0] - model_real[<unsigned int>i,0]
        diff2 = data_imag[<unsigned int>i,0] - model_imag[<unsigned int>i,0]
        chisq += (diff1*diff1 + diff2*diff2) * data_weights[<unsigned int>i,0]

    return chisq
