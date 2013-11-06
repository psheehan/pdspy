import numpy
import h5py
from ..constants.physics import G, m_p
from ..constants.astronomy import AU, M_sun
from ..constants.math import pi
from ..dust import Dust
from ..gas import Gas

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
        self.gas = []
        self.abundance = []

    def add_gas(self, gas, abundance):
        self.gas.append(gas)
        self.abundance.append(abundance)

    def density(self, r, theta, phi):
        ##### Disk Parameters
        
        rin = self.rmin * AU
        rout = self.rmax * AU
        mass = self.mass * M_sun
        h = self.h0 * AU
        plrho = self.plrho
        plh = self.plh

        ##### Set up the coordinates

        rt, tt, pp = numpy.meshgrid(r*AU, theta, phi,indexing='ij')

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

    def number_density(self, r, theta, phi, gas=0):
        rho = self.density(r, theta, phi)

        n_H2 = rho * 100. / (2*m_p)

        n = n_H2 * self.abundance[gas]

        return n

    def surface_density(self, r):
        rin = self.rmin * AU
        rout = self.rmax * AU
        mass = self.mass * M_sun
        h = self.h0 * AU
        plrho = self.plrho
        plh = self.plh

        Sigma0 = (2-plrho+plh)*mass/(2*pi*(1*AU)**(plrho-plh)) / \
                (rout**(-plrho+plh+2) - rin**(-plrho+plh+2))

        Sigma = Sigma0 * r**(-plrho+plh)

        Sigma[(r >= rout/AU) ^ (r <= rin/AU)] = 0e0

        return Sigma

    def temperature(self, r, T_0=100., p=1):
        rin = self.rmin * AU
        rout = self.rmax * AU
        mass = self.mass * M_sun
        h = self.h0 * AU
        plrho = self.plrho
        plh = self.plh

        T = T_0 * r**(-p)

        T[(r >= rout/AU) ^ (r <= rin/AU)] = 0.0

        return T

    def velocity(self, r, theta, phi, mstar=0.5):
        mstar *= M_sun

        rt, tt, pp = numpy.meshgrid(r*AU, theta, phi,indexing='ij')
        rr = rt*numpy.sin(tt)

        v_r = numpy.zeros(rr.shape)
        v_theta = numpy.zeros(rr.shape)
        v_phi = numpy.sqrt(G*mstar/rr)

        return numpy.array((v_r, v_theta, v_phi))

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

        for name in f['Gas']:
            self.gas.append(Gas())
            self.abundance.append(f['Gas'][name]['Abundance'].value)
            self.gas[-1].set_properties_from_file(usefile=f['Gas'][name])

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

        gases = []
        if hasattr(self, 'gas'):
            gas = f.create_group("Gas")
            for i in range(len(self.gas)):
                gases.append(gas.create_group("Gas{0:d}".format(i)))
                gases[i]["Abundance"] = self.abundance[i]
                self.gas[i].write(usefile=gases[i])

        if (usefile == None):
            f.close()
