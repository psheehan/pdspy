import numpy
import scipy
import h5py
from .Dust import Dust

class DustGenerator:
    def __init__(self, dust, p=3.5):
        if type(dust) == str:
            self.read(dust)
        else:
            self.amax = numpy.logspace(-4.,0.,50)
            self.lam = dust.lam

            self.kabs = []
            self.ksca = []
            self.kext = []
            self.albedo = []

            for a in self.amax:
                dust.calculate_size_distribution_opacity(0.005e-4, a, p, \
                        coat_volume_fraction=0.0)

                self.kabs.append(dust.kabs)
                self.ksca.append(dust.ksca)
                self.kext.append(dust.kext)
                self.albedo.append(dust.albedo)

            self.kabs = numpy.array(self.kabs)
            self.ksca = numpy.array(self.ksca)
            self.kext = numpy.array(self.kext)
            self.albedo = numpy.array(self.albedo)

    def __call__(self, amax):
        f_kabs = scipy.interpolate.interp2d(self.lam, self.amax, \
                numpy.log10(self.kabs))
        f_ksca = scipy.interpolate.interp2d(self.lam, self.amax, \
                numpy.log10(self.ksca))

        kabs = 10.**f_kabs(self.lam, amax)
        ksca = 10.**f_ksca(self.lam, amax)

        d = Dust()
        d.set_properties(self.lam, kabs, ksca)

        return d

    def read(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "r")
        else:
            f = usefile

        self.lam = f['lam'].value
        self.amax = f['amax'].value

        if ('kabs' in f):
            self.kabs = f['kabs'].value
        if ('ksca' in f):
            self.ksca = f['ksca'].value
        if (hasattr(self, 'kabs') and hasattr(self, 'ksca')):
            self.kext = self.kabs + self.ksca
            self.albedo = self.ksca / self.kext

        if (usefile == None):
            f.close()

    def write(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "w")
        else:
            f = usefile

        if hasattr(self, 'amax'):
            amax_dset = f.create_dataset("amax", (self.amax.size,), dtype='f')
            amax_dset[...] = self.amax
        if hasattr(self, 'lam'):
            lam_dset = f.create_dataset("lam", (self.lam.size,), dtype='f')
            lam_dset[...] = self.lam
        
        if hasattr(self, 'kabs'):
            kabs_dset = f.create_dataset("kabs", self.kabs.shape, dtype='f')
            kabs_dset[...] = self.kabs
        if hasattr(self, 'ksca'):
            ksca_dset = f.create_dataset("ksca", self.ksca.shape, dtype='f')
            ksca_dset[...] = self.ksca
        if hasattr(self, 'g'):
            g_dset = f.create_dataset("g", self.g.shape, dtype='f')
            g_dset[...] = self.g

        if (usefile == None):
            f.close()
