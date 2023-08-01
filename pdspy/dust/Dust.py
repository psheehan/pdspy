import numpy
import scipy
import h5py
from ..constants.physics import c
from ..constants.math import pi
#from .bhmie import bhmie
#from .bhcoat import bhcoat
#from .dmilay import dmilay

class Dust:

    def add_coat(self, coat):
        self.coat = coat

    def calculate_optical_constants_on_wavelength_grid(self, lam):
        f = scipy.interpolate.interp1d(self.lam, self.n)
        n = f(lam)
        f = scipy.interpolate.interp1d(self.lam, self.k)
        k = f(lam)

        self.lam = lam
        self.nu = c / self.lam

        self.n = n
        self.k = k
        self.m = self.n + 1j*self.k

    def calculate_opacity_at_wavelength(self, lam):
        f = scipy.interpolate.interp1d(self.lam, self.kabs)
        kabs_interp = f(lam)

        return kabs_interp

    def calculate_size_distribution_opacity(self, amin, amax, p, \
            coat_volume_fraction=0.0, nang=1000, with_dhs=False, fmax=0.8, \
            nf=50):
        na = int(round(numpy.log10(amax) - numpy.log10(amin))*100+1)
        a = numpy.logspace(numpy.log10(amin),numpy.log10(amax),na)
        kabsgrid = numpy.zeros((self.lam.size,na))
        kscagrid = numpy.zeros((self.lam.size,na))
        
        normfunc = a**(3-p)

        for i in range(na):
            if with_dhs:
                self.calculate_dhs_opacity(a[i], fmax=fmax, nf=nf, nang=nang)
            else:
                self.calculate_opacity(a[i], \
                        coat_volume_fraction=coat_volume_fraction, nang=nang)
            
            kabsgrid[:,i] = self.kabs*normfunc[i]
            kscagrid[:,i] = self.ksca*normfunc[i]
        
        norm = scipy.integrate.trapz(normfunc,x=a)
        
        self.kabs = scipy.integrate.trapz(kabsgrid,x=a)/norm
        self.ksca = scipy.integrate.trapz(kscagrid,x=a)/norm
        self.kext = self.kabs + self.ksca
        self.albedo = self.ksca / self.kext

    def calculate_opacity(self, a, coat_volume_fraction=0.0, nang=1000):
        self.kabs = numpy.zeros(self.lam.size)
        self.ksca = numpy.zeros(self.lam.size)
        
        if not hasattr(self, 'coat'):
            mdust = 4*pi*a**3/3*self.rho
            
            for i in range(self.lam.size):
                x = 2*pi*a/self.lam[i]
                
                S1,S2,Qext,Qsca,Qback,gsca=bhmie(x,self.m[i],nang)
                
                Qabs = Qext - Qsca
                
                self.kabs[i] = pi*a**2*Qabs/mdust
                self.ksca[i] = pi*a**2*Qsca/mdust
        
        else:
            a_coat = a*(1+coat_volume_fraction)**(1./3)

            mdust = 4*pi*a**3/3*self.rho+ \
                    4*pi/3*(a_coat**3-a**3)*self.coat.rho
            
            for i in range(self.lam.size):
                x = 2*pi*a/self.lam[i]
                y = 2*pi*a_coat/self.lam[i]
                
                Qext,Qsca,Qback=bhcoat(x,y,self.m[i],self.coat.m[i])
                
                Qabs = Qext - Qsca
                
                self.kabs[i] = pi*a_coat**2*Qabs/mdust
                self.ksca[i] = pi*a_coat**2*Qsca/mdust

        self.kext = self.kabs + self.ksca
        self.albedo = self.ksca / self.kext

    def calculate_dhs_opacity(self, a, fmax=0.8, nf=50, nang=1000):

        self.kabs = numpy.zeros(self.lam.size)
        self.ksca = numpy.zeros(self.lam.size)
        
        for i in range(self.lam.size):
            for j, f in enumerate(numpy.linspace(0., fmax, nf)):
                x = 2*pi*a*f**(1./3)/self.lam[i]
                y = 2*pi*a/self.lam[i]

                if f == 0:
                    S1,S2,Qext,Qsca,Qback,gsca=bhmie(y,self.m[i],nang)
                else:
                    Qext, Qsca, Qback, g, M1, M2, S21, D21 = dmilay(\
                            a*f**(1./3), a, 2*pi/self.lam[i], self.m[i].real - \
                            1j*self.m[i].imag, 1.0+1j*0.0, [0.], 1., 1)
                
                Qabs = Qext - Qsca
                
                mdust = 4*pi*a**3*(1.-f)/3*self.rho

                self.kabs[i] += pi*a**2*Qabs/mdust * 1./fmax * fmax/nf
                self.ksca[i] += pi*a**2*Qsca/mdust * 1./fmax * fmax/nf
    
        self.kext = self.kabs + self.ksca
        self.albedo = self.ksca / self.kext

    def set_density(self, rho):
        self.rho = rho

    def set_optical_constants(self, lam, n, k):
        self.lam = lam
        self.nu = c / self.lam

        self.n = n
        self.k = k
        self.m = n+1j*k

    def set_optical_constants_from_draine(self, filename):
        opt_data = numpy.loadtxt(filename)

        self.lam = numpy.flipud(opt_data[:,0])*1.0e-4
        self.nu = c / self.lam

        self.n = numpy.flipud(opt_data[:,3])+1.0
        self.k = numpy.flipud(opt_data[:,4])
        self.m = self.n+1j*self.k

    def set_optical_constants_from_henn(self, filename):
        opt_data = numpy.loadtxt(filename)

        self.lam = opt_data[:,0]*1.0e-4
        self.nu = c / self.lam

        self.n = opt_data[:,1]
        self.k = opt_data[:,2]
        self.m = self.n+1j*self.k

    def set_optical_constants_from_jena(self, filename, type="standard"):
        opt_data = numpy.loadtxt(filename)

        if type == "standard":
            self.lam = numpy.flipud(1./opt_data[:,0])
            self.n = numpy.flipud(opt_data[:,1])
            self.k = numpy.flipud(opt_data[:,2])
        elif type == "umwave":
            self.lam = numpy.flipud(opt_data[:,0])*1.0e-4
            self.n = numpy.flipud(opt_data[:,1])
            self.k = numpy.flipud(opt_data[:,2])

        self.nu = c / self.lam
        self.m = self.n+1j*self.k

    def set_optical_constants_from_oss(self, filename):
        opt_data = numpy.loadtxt(filename)
        
        self.lam = opt_data[:,0] # in cm
        self.nu = c / self.lam

        self.n = opt_data[:,1]
        self.k = opt_data[:,2]
        self.m = self.n+1j*self.k

    def set_properties(self, lam, kabs, ksca):
        self.lam = lam
        self.nu = c / self.lam

        self.kabs = kabs
        self.ksca = ksca
        self.kext = self.kabs + self.ksca
        self.albedo = self.ksca / self.kext

    def set_properties_from_file(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "r")
        else:
            f = usefile

        self.lam = f['lam'][...]
        self.nu = c / self.lam

        if ('n' in f):
            self.n = f['n'][...]
        if ('k' in f):
            self.k = f['k'][...]
        if (hasattr(self, 'n') and hasattr(self, 'k')):
            self.m = self.n + 1j*self.k

        if ('rho' in f):
            self.rho = f['rho'][...][0]

        if ('kabs' in f):
            self.kabs = f['kabs'][...]
        if ('ksca' in f):
            self.ksca = f['ksca'][...]
        if (hasattr(self, 'kabs') and hasattr(self, 'ksca')):
            self.kext = self.kabs + self.ksca
            self.albedo = self.ksca / self.kext

        if ('coat' in f):
            self.coat = Dust()
            self.coat.set_properties_from_file(usefile=f['coat'])

        if (usefile == None):
            f.close()

    def write(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "w")
        else:
            f = usefile

        if hasattr(self, 'lam'):
            lam_dset = f.create_dataset("lam", (self.lam.size,), dtype='f')
            lam_dset[...] = self.lam
        if hasattr(self, 'n'):
            n_dset = f.create_dataset("n", (self.n.size,), dtype='f')
            n_dset[...] = self.n
        if hasattr(self, 'k'):
            k_dset = f.create_dataset("k", (self.k.size,), dtype='f')
            k_dset[...] = self.k
        
        if hasattr(self, 'rho'):
            rho_dset = f.create_dataset("rho", (1,), dtype='f')
            rho_dset[...] = [self.rho]

        if hasattr(self, 'kabs'):
            kabs_dset = f.create_dataset("kabs", (self.kabs.size,), dtype='f')
            kabs_dset[...] = self.kabs
        if hasattr(self, 'ksca'):
            ksca_dset = f.create_dataset("ksca", (self.ksca.size,), dtype='f')
            ksca_dset[...] = self.ksca
        if hasattr(self, 'g'):
            g_dset = f.create_dataset("g", (self.g.size,), dtype='f')
            g_dset[...] = self.g

        if hasattr(self, 'coat'):
            coat_group = f.create_group("coat")

            self.coat.write(usefile=coat_group)

        if (usefile == None):
            f.close()
