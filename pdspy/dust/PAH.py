import numpy
import scipy
import h5py
from ..constants.physics import c
from ..constants.math import pi

class PAH:

    def calculate_size_distribution_opacity(self, amin, amax, p):
        na = int(round(numpy.log10(amax) - numpy.log10(amin))*100+1)
        a = numpy.logspace(numpy.log10(amin),numpy.log10(amax),na)
        kabsgrid = numpy.zeros((self.lam.size,na))
        kscagrid = numpy.zeros((self.lam.size,na))
        
        normfunc = a**(3-p)

        for i in range(na):
            self.calculate_opacity(a[i])
            
            kabsgrid[:,i] = self.kabs*normfunc[i]
            kscagrid[:,i] = self.ksca*normfunc[i]
        
        norm = scipy.integrate.trapz(normfunc,x=a)
        
        self.kabs = scipy.integrate.trapz(kabsgrid,x=a)/norm
        self.ksca = scipy.integrate.trapz(kscagrid,x=a)/norm
        self.kext = self.kabs + self.ksca
        self.albedo = self.ksca / self.kext

    def calculate_opacity(self, a):
        mdust = 4*pi*a**3/3*self.rho

        fabs = scipy.interpolate.interp2d(self.lam, self.a, \
                numpy.log10(self.Qabs), kind='linear')
        fsca = scipy.interpolate.interp2d(self.lam, self.a, \
                numpy.log10(self.Qsca), kind='linear')
        fext = scipy.interpolate.interp2d(self.lam, self.a, \
                numpy.log10(self.Qext), kind='linear')

        Qabs = 10.**fabs(self.lam, a)
        Qsca = 10.**fsca(self.lam, a)
        Qext = 10.**fext(self.lam, a)
        
        self.kabs = pi*a**2*Qabs/mdust
        self.ksca = pi*a**2*Qsca/mdust
        self.kext = pi*a**2*Qext/mdust
    
        self.albedo = self.ksca / self.kext

    def set_properties_from_draine(self, neutral, ion, qion):
        # Read in the neutral PAH data.

        f = open(neutral, "r")
        lines = f.readlines()
        f.close()

        self.nr = int(lines[7].split()[0])
        self.nlam = int(lines[8].split()[0])

        a_neutral = []
        optical_data_neutral = []
        for i in range(self.nr):
            a_neutral += [float(lines[10 + i*(self.nlam+3)].split()[0])]
            optical_data_neutral += [numpy.array([line.split() for line in \
                    lines[12+i*(self.nlam+3):12+i*(self.nlam+3)+self.nlam]], \
                    dtype=float)]

        a_neutral = numpy.array(a_neutral)
        optical_data_neutral = numpy.array(optical_data_neutral)
        lam_neutral = optical_data_neutral[:,:,0].mean(axis=0)

        # Read in the ionized PAH data.

        f = open(ion, "r")
        lines = f.readlines()
        f.close()

        nr = int(lines[7].split()[0])
        nlam = int(lines[8].split()[0])
        if nr != self.nr or (nlam != self.nlam):
            raise Warning("Ionized and neutral PAH files don't have the same "
                    "shape")

        a_ion = []
        optical_data_ion = []
        for i in range(self.nr):
            a_ion += [float(lines[10 + i*(self.nlam+3)].split()[0])]
            optical_data_ion += [numpy.array([line.split() for line in \
                    lines[12+i*(self.nlam+3):12+i*(self.nlam+3)+self.nlam]], \
                    dtype=float)]

        a_ion = numpy.array(a_ion)
        optical_data_ion = numpy.array(optical_data_ion)
        lam_ion = optical_data_ion[:,:,0].mean(axis=0)

        # Check that a and lam are the same between the two files.

        if not numpy.all(numpy.abs(a_ion - a_neutral)/a_neutral < 1.0e-6):
            raise Warning("Grain sizes are different between ionized and "
                    "neutral PAH data files.")
        else:
            self.a = 0.5*(a_ion + a_neutral) / 1.0e4

        if not numpy.all(numpy.abs(lam_ion - lam_neutral)/lam_neutral < 1.0e-6):
            raise Warning("Wavelengths are different between ionized and "
                    "neutral PAH data files.")
        else:
            self.lam = 0.5*(lam_ion + lam_neutral)[::-1] / 1.0e4

        # Load the ionization fraction data.

        data = numpy.loadtxt(qion, delimiter=",")
        qion_a = data[:,0]/1.0e8
        qion_q = data[:,1]

        # Merge the ionized and neutral data together based on qion.

        fqion = scipy.interpolate.interp1d(numpy.log10(qion_a), qion_q, \
                kind="linear", bounds_error=False, fill_value="extrapolate")

        q = fqion(numpy.log10(self.a))[:,numpy.newaxis,numpy.newaxis]

        optical_data = optical_data_ion*q + optical_data_neutral*(1-q)

        # Store the results in this object.

        self.Qext = optical_data[:,::-1,1]
        self.Qabs = optical_data[:,::-1,2]
        self.Qsca = optical_data[:,::-1,3]
        self.g = optical_data[:,::-1,4]

    def set_density(self, rho):
        self.rho = rho

    def set_properties_from_file(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "r")
        else:
            f = usefile

        self.a = f['a'][...]
        self.lam = f['lam'][...]
        self.nu = c / self.lam

        if ('Qabs' in f):
            self.Qabs = f['Qabs'][...]
        if ('Qsca' in f):
            self.Qsca = f['Qsca'][...]
        if ('Qext' in f):
            self.Qext = f['Qext'][...]
        if ('g' in f):
            self.g = f['g'][...]

        if ('rho' in f):
            self.rho = f['rho'][...][0]

        if ('kabs' in f):
            self.kabs = f['kabs'][...]
        if ('ksca' in f):
            self.ksca = f['ksca'][...]
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

        if hasattr(self, 'lam'):
            lam_dset = f.create_dataset("lam", (self.lam.size,), dtype='f')
            lam_dset[...] = self.lam
        if hasattr(self, 'a'):
            a_dset = f.create_dataset("a", (self.a.size,), dtype='f')
            a_dset[...] = self.a

        if hasattr(self, 'Qabs'):
            Qabs_dset = f.create_dataset("Qabs", self.Qabs.shape, dtype='f')
            Qabs_dset[...] = self.Qabs
        if hasattr(self, 'Qsca'):
            Qsca_dset = f.create_dataset("Qsca", self.Qsca.shape, dtype='f')
            Qsca_dset[...] = self.Qsca
        if hasattr(self, 'Qext'):
            Qext_dset = f.create_dataset("Qext", self.Qext.shape, dtype='f')
            Qext_dset[...] = self.Qext
        if hasattr(self, 'g'):
            g_dset = f.create_dataset("g", self.g.shape, dtype='f')
            g_dset[...] = self.g
        
        if hasattr(self, 'rho'):
            rho_dset = f.create_dataset("rho", (1,), dtype='f')
            rho_dset[...] = [self.rho]

        if hasattr(self, 'kabs'):
            kabs_dset = f.create_dataset("kabs", (self.kabs.size,), dtype='f')
            kabs_dset[...] = self.kabs
        if hasattr(self, 'ksca'):
            ksca_dset = f.create_dataset("ksca", (self.ksca.size,), dtype='f')
            ksca_dset[...] = self.ksca

        if (usefile == None):
            f.close()
