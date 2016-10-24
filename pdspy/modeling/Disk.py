import numpy
import h5py
from ..constants.physics import G, m_p
from ..constants.astronomy import AU, M_sun
from ..constants.math import pi
from ..dust import Dust
from ..gas import Gas

class Disk:
    def __init__(self, mass=1.0e-3, rmin=0.1, rmax=300, plrho=2.37, h0=0.1, \
            plh=58./45., t0=None, plt=None, dust=None, gap_rin=[], gap_rout=[],\
            gap_delta=[], tmid0=None, tatm0=None, zq0=None, pltgas=None, \
            delta=None):
        self.mass = mass
        self.rmin = rmin
        self.rmax = rmax
        self.plrho = plrho
        self.h0 = h0
        self.plh = plh
        self.t0 = t0
        self.plt = plt
        self.gap_rin = gap_rin
        self.gap_rout = gap_rout
        self.gap_delta = gap_delta
        if (dust != None):
            self.dust = dust

        # The gas parameter lists.

        self.gas = []
        self.abundance = []
        self.tmid0 = tmid0
        self.tatm0 = tatm0
        self.zq0 = zq0
        self.pltgas = pltgas
        self.delta = delta

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

        ##### Add any gaps to the disk.

        for i in range(len(self.gap_rin)):
            rho[(rr >= self.gap_rin[i]*AU) & (rr <= self.gap_rout[i]*AU)] *= \
                    self.gap_delta[i]
        
        return rho

    def temperature(self, r, theta, phi):
        ##### Disk Parameters
        
        rin = self.rmin * AU
        rout = self.rmax * AU
        t0 = self.t0
        plt = self.plt

        ##### Set up the coordinates

        rt, tt, pp = numpy.meshgrid(r*AU, theta, phi,indexing='ij')

        rr = rt*numpy.sin(tt)
        zz = rt*numpy.cos(tt)

        ##### Make the dust density model for a protoplanetary disk.
        
        t = t0 * (rt / rin)**(-plt)

        t[(rr >= rout) ^ (rr <= rin)] = 0e0
        
        return t

    def number_density(self, r, theta, phi, gas=0):
        rho = self.density(r, theta, phi)

        n_H2 = rho * 100. / (2*m_p)

        n = n_H2 * self.abundance[gas]

        return n

    def gas_temperature(self, r, theta, phi):
        ##### Disk Parameters
        
        rin = self.rmin * AU
        rout = self.rmax * AU
        pltgas = self.pltgas
        tmid0 = self.tmid0
        tatm0 = self.tatm0
        zq0 = self.zq0
        delta = self.delta

        ##### Set up the coordinates

        rt, tt, pp = numpy.meshgrid(r*AU, theta, phi,indexing='ij')

        rr = rt*numpy.sin(tt)
        zz = rt*numpy.cos(tt)

        ##### Make the dust density model for a protoplanetary disk.
        
        zq = zq0 * (rt / rin)**1.3

        tmid = tmid0 * (rt / rin)**(-pltgas)
        tatm = tatm0 * (rt / rin)**(-pltgas)

        t = numpy.zeros(tatm.shape)
        t[zz >= zq] = tatm[zz >= zq]
        t[zz < zq] = tatm[zz < zq] + (tmid[zz < zq] - tatm[zz < zq]) * \
                (numpy.cos(numpy.pi * zz[zz < zq] / (2*zq[zz < zq])))**2*delta
        
        return t

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

        dr = r[r > 0].min()
        Sigma[r == 0] = Sigma0 * (0.7*dr)**(-plrho+plh)

        for i in range(len(self.gap_rin)):
            Sigma[(r >= self.gap_rin[i]) & (r <= self.gap_rout[i])] *= \
                    self.gap_delta[i]
        
        return Sigma

    def temperature_1d(self, r):
        rin = self.rmin * AU
        rout = self.rmax * AU
        t0 = self.t0
        plt = self.plt

        T = t0 * r**(-plt)

        T[(r >= rout/AU) ^ (r <= rin/AU)] = 0.0

        dr = r[r > 0].min()
        T[r == 0] = t0 * (0.7*dr)**(-plt)

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

        if 't0' in f:
            self.t0 = f['t0'].value
            self.plt = f['plt'].value

        if 'tmid0' in f:
            self.tmid0 = f['tmid0'].value
            self.tatm0 = f['tatm0'].value
            self.zq0 = f['zq0'].value
            self.pltgas = f['pltgas'].value
            self.delta = f['delta'].value

        if ('Dust' in f):
            self.dust = Dust()
            self.dust.set_properties_from_file(usefile=f['Dust'])

        if ('Gas' in f):
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

        if self.t0 != None:
            f['t0'] = self.t0
            f['plt'] = self.plt

        if self.tmid0 != None:
            f['tmid0'] = self.tmid0
            f['tatm0'] = self.tatm0
            f['zq0'] = self.zq0
            f['pltgas'] = self.pltgas
            f['delta'] = self.delta

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
