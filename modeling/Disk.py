import numpy
import h5py
from ..constants.astronomy import AU, M_sun
from ..constants.math import pi
from ..dust import Dust

class Disk:
    def __init__(self, mass=1.0e-3, rmin=0.1, rmax=300, plrho=2.37, h0=0.1, \
            plh=58./45., dust=None):
        self.mass = mass
        self.rmin = rmin
        self.rmax = rmax
        self.plrho = plrho
        self.h0 = h0
        self.plh = plh
        if (dust != None):
            self.dust = dust

    def density(self, r, theta, phi):
        ##### Disk Parameters
        
        rin = self.rmin * AU
        rout = self.rmax * AU
        mass = self.mass * M_sun
        h = self.h0 * AU
        plrho = self.plrho
        plh = self.plh

        ##### Set up the coordinates

        rt, tt, pp = numpy.meshgrid(0.5*(r[0:r.size-1]+r[1:r.size])*AU, \
                0.5*(theta[0:theta.size-1]+theta[1:theta.size]), \
                0.5*(phi[0:phi.size-1]+phi[1:phi.size]),indexing='ij')

        rr = rt*numpy.sin(tt)
        zz = rt*numpy.cos(tt)

        ##### Make the dust density model for a protoplanetary disk.
        
        rho0 = mass/((2*pi)**1.5*h*(1*AU)**(plrho-plh))*\
                (-plrho+plh+2)/(rout**(-plrho+plh+2)-rin**(-plrho+plh+2))
        hr = h * (rr / (1*AU))**(plh)
        rho0 = rho0 * (rr / (1*AU))**(-plrho)
        rho1 = numpy.exp(-0.5*(zz / hr)**2)
        rho = rho0 * rho1
        rho[(rr >= rout) ^ (rr <= rin)] = 0e0
        
        return rho

    def read(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "r")
        else:
            f = usefile

        self.mass = f['mass'].value
        self.rmin = f['rmin'].value
        self.rmax = f['rmax'].value
        self.plrho = f['plrho'].value
        self.h0 = f['h0'].value
        self.plh = f['plh'].value

        if ('Dust' in f):
            self.dust = Dust()
            self.dust.set_properties_from_file(usefile=f['Dust'])

        if (usefile == None):
            f.close()

    def write(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "w")
        else:
            f = usefile

        f['mass'] = self.mass
        f['rmin'] = self.rmin
        f['rmax'] = self.rmax
        f['plrho'] = self.plrho
        f['h0'] = self.h0
        f['plh'] = self.plh

        if hasattr(self, 'dust'):
            dust = f.create_group("Dust")
            self.dust.write(usefile=dust)

        if (usefile == None):
            f.close()
