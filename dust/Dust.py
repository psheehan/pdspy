import numpy
from ..constants.physics import c
from ..constants.math import pi
from .bhmie import bhmie
from .bhcoat import bhcoat

class Dust:

    def add_coat(self, coat):
        self.coat = coat

    def calculate_size_distribution_opacity(self, amin, amax, p, \
            coat_volume_fraction=None):
        na = 1000
        a = numpy.logscale(numpy.log10(amin),numpy.log10(amax),na)
        kabsgrid=zeros((self.lam.size,na))
        kscagrid=zeros((self.lam.size,na))
        
        normfunc = a**(3-p)
        
        for i in range(na):
            self.calculate_opacity(a[i], \
                    coat_volume_fraction=coat_volume_fraction)
            
            kabsgrid[:,i] = self.kabs*normfunc[i]
            kscagrid[:,i] = self.ksca*normfunc[i]
        
        norm = trapz(normfunc,x=a)
        
        self.kabs = trapz(kabsgrid,x=a)/norm
        self.ksca = trapz(kscagrid,x=a)/norm
        self.kext = self.kabs + self.ksca
        self.albedo = self.ksca / self.kext

    def calculate_opacity(self, a, coat_volume_fraction=None):
        self.kabs = numpy.zeros(grain.lam.size)
        self.ksca = numpy.zeros(grain.lam.size)
        
        if self.coat == None:
            mdust = 4*pi*a**3/3*self.rho
            
            for i in range(self.lam.size):
                x = 2*pi*a/self.lam[i]
                
                S1,S2,Qext,Qsca,Qback,gsca=bhmie(x,self.m[i],1000)
                
                Qabs = Qext - Qsca
                
                self.kabs[i] = pi*grain.a**2*Qabs/mdust
                self.ksca[i] = pi*grain.a**2*Qsca/mdust
        
        else:
            a_coat = a*(1+coat_volume_fraction)**(1./3)

            mdust = 4*pi*a**3/3*self.rho+ \
                    4*pi/3*(a_coat**3-a**3)*self.coat.rho
            
            for i in arange(grain.lam.size):
                x = 2*pi*a/self.lam[i]
                y = 2*pi*a_coat/self.lam[i]
                
                Qext,Qsca,Qback=bhcoat(x,y,self.m[i],self.coat.m[i])
                
                Qabs = Qext - Qsca
                
                self.kabs[i] = pi*a_coat**2*Qabs/mdust
                self.ksca[i] = pi*a_coat**2*Qsca/mdust

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
        self.m = n+1j*k

    def set_optical_constants_from_henn(self, filename):
        opt_data = numpy.loadtxt(filename)

        self.lam = numpy.flipud(opt_data[:,0])*1.0e-4
        self.nu = c / self.lam

        self.n = numpy.flipud(optdata[:,3])
        self.k = numpy.flipud(optdata[:,4])
        self.m = n+1j*k

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
        self.m = n+1j*k

    def set_optical_constants_from_oss(self, filename):
        opt_data = numpy.loadtxt(filename)
        
        self.lam = opt_data[:,0] # in cm
        self.nu = c / self.lam

        self.n = opt_data[:,1]
        self.k = opt_data[:,2]
        self.m = n+1j*k

    def set_properties(self, lam, kabs, ksca):
        self.lam = lam
        self.nu = c / self.lam

        self.kabs = kabs
        self.ksca = ksca
        self.kext = self.kabs + self.ksca
        self.albedo = self.ksca / self.kext

    def set_properties_from_file(self, filename):
        print("Function not yet implemented.")

    def write(self, filename):
        print("Function not yet implemented.")
