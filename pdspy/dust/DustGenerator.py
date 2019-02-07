import numpy
import scipy
import h5py
from .Dust import Dust

class DustGenerator:
    def __init__(self, dust, with_dhs=False, fmax=0.8, nf=50, singlesize=False):
        if type(dust) == str:
            self.read(dust)
        else:
            self.rho = dust.rho

            self.amax = numpy.logspace(-4.,1.,60)
            self.p = numpy.linspace(2.5,4.5,11)
            self.lam = dust.lam

            self.kabs = []
            self.ksca = []
            self.kext = []
            self.albedo = []

            a = numpy.logspace(numpy.log10(0.05e-4),1.,500)
            if singlesize:
                self.amax = a
            kabsgrid = numpy.zeros((self.lam.size, a.size))
            kscagrid = numpy.zeros((self.lam.size, a.size))

            for i in range(a.size):
                if with_dhs:
                    dust.calculate_dhs_opacity(a[i], fmax=fmax, nf=nf, nang=1)
                else:
                    dust.calculate_opacity(a[i], coat_volume_fraction=0.0, \
                            nang=1)

                kabsgrid[:,i] = dust.kabs
                kscagrid[:,i] = dust.ksca

            for p in self.p:
                kabs_temp = []
                ksca_temp = []
                kext_temp = []
                albedo_temp = []

                for amax in self.amax:
                    if singlesize:
                        normfunc = numpy.zeros(a.size)
                        normfunc[a == amax] = 1.
                    else:
                        normfunc = a**(3-p)
                        normfunc[a > amax] = 0.

                    norm = scipy.integrate.trapz(normfunc, x=a)

                    kabs_temp.append(scipy.integrate.trapz(kabsgrid*normfunc,\
                            x=a, axis=1)/norm)
                    ksca_temp.append(scipy.integrate.trapz(kscagrid*normfunc,\
                            x=a, axis=1)/norm)
                    kext_temp.append(kabs_temp[-1] + ksca_temp[-1])
                    albedo_temp.append( ksca_temp[-1] / kext_temp[-1])

                self.kabs.append(kabs_temp)
                self.ksca.append(ksca_temp)
                self.kext.append(kext_temp)
                self.albedo.append(albedo_temp)

            self.kabs = numpy.array(self.kabs)
            self.ksca = numpy.array(self.ksca)
            self.kext = numpy.array(self.kext)
            self.albedo = numpy.array(self.albedo)

    def __call__(self, amax, p=None):
        if self.old:
            f_kabs = scipy.interpolate.interp2d(self.lam, self.amax, \
                    numpy.log10(self.kabs))
            f_ksca = scipy.interpolate.interp2d(self.lam, self.amax, \
                    numpy.log10(self.ksca))

            kabs = 10.**f_kabs(self.lam, amax)
            ksca = 10.**f_ksca(self.lam, amax)

            d = Dust()
            d.set_properties(self.lam, kabs, ksca)
        else:
            f_kabs = scipy.interpolate.RegularGridInterpolator(\
                    (self.p, self.amax, self.lam), numpy.log10(self.kabs))
            f_ksca = scipy.interpolate.RegularGridInterpolator(\
                    (self.p, self.amax, self.lam), numpy.log10(self.ksca))

            pts = numpy.array([[p, amax, lam] for lam in self.lam])

            kabs = 10.**f_kabs(pts)
            ksca = 10.**f_ksca(pts)

            d = Dust()
            d.set_properties(self.lam, kabs, ksca)

        return d

    def read(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "r")
        else:
            f = usefile

        self.lam = f['lam'][...]
        self.amax = f['amax'][...]

        if ('p' in f):
            self.p = f['p'][...]
            self.old = False
        else:
            self.old = True

        if ('kabs' in f):
            self.kabs = f['kabs'][...]
        if ('ksca' in f):
            self.ksca = f['ksca'][...]
        if (hasattr(self, 'kabs') and hasattr(self, 'ksca')):
            self.kext = self.kabs + self.ksca
            self.albedo = self.ksca / self.kext

        if ('rho' in f):
            self.rho = f['rho'][...][0]

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
        if hasattr(self, 'p'):
            amax_dset = f.create_dataset("p", (self.p.size,), dtype='f')
            amax_dset[...] = self.p
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

        if hasattr(self, 'rho'):
            rho_dset = f.create_dataset("rho", (1,), dtype='f')
            rho_dset[...] = [self.rho]

        if (usefile == None):
            f.close()
