import numpy
import h5py
from scipy.integrate import trapz
from ..constants.physics import G, m_p
from ..constants.astronomy import AU, M_sun
from ..constants.math import pi
from ..dust import Dust
from ..gas import Gas

class SettledDisk:
    def __init__(self, mass=1.0e-3, rmin=0.1, rmax=300, plrho=2.37, h0=0.1, \
            plh=58./45., t0=None, plt=None, dust=None, gap_rin=[], gap_rout=[],\
            gap_delta=[], tmid0=None, tatm0=None, zq0=None, pltgas=None, \
            delta=None, aturb=None, gaussian_gaps=False, amin=0.05, amax=1000.,\
            pla=3.5, alpha_settle=1.0e-3, gamma_taper=None):
        self.mass = mass
        self.rmin = rmin
        self.rmax = rmax
        self.plrho = plrho
        self.h0 = h0
        self.plh = plh
        self.gamma_taper = gamma_taper
        self.t0 = t0
        self.plt = plt
        self.gap_rin = gap_rin
        self.gap_rout = gap_rout
        self.gap_delta = gap_delta
        self.gaussian_gaps = gaussian_gaps

        # Dust parameters.

        self.amin = amin
        self.amax = amax
        self.pla = pla
        self.alpha_settle = alpha_settle

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
        self.aturb = aturb

    def add_gas(self, gas, abundance):
        self.gas.append(gas)
        self.abundance.append(abundance)

    def density(self, r, theta, phi, na=100, normalize=True):
        ##### Set up the coordinates

        rt, tt, pp = numpy.meshgrid(r*AU, theta, phi,indexing='ij')

        rr = rt*numpy.sin(tt)
        zz = rt*numpy.cos(tt)

        ##### Calculate the fraction of the mass in each grain size bin.

        aa = numpy.logspace(numpy.log10(self.amin), numpy.log10(self.amax),na+1)

        f = aa**-self.pla

        da = aa[1:] - aa[0:-1]

        fa3 = f * aa**3

        mass_frac = (fa3[1:] + fa3[0:-1]) / 2. * da / numpy.trapz(fa3, x=aa)
        
        ##### Get a list of the appropriate dust grain sizes.

        a = (aa[1:] + aa[0:-1]) / 2.

        ##### Make the gas density model for a protoplanetary disk.

        Sigma = self.surface_density(rr/AU, normalize=normalize)
        h_g = self.scale_height(rr/AU)

        rho = numpy.zeros(Sigma.shape + (100,))

        for i in range(na):
            gamma0 = 2.
            rho_mid = 100 * Sigma / (numpy.sqrt(2*numpy.pi)*h_g)

            b = (1 + gamma0)**-0.5 * self.alpha_settle * rho_mid * h_g / \
                    (self.dust.rho * a[i] * 1.0e-4)
            y = numpy.sqrt(b / (1. + b))
            h = y * h_g

            rho[:,:,:,i] = mass_frac[i] * Sigma / (numpy.sqrt(2*numpy.pi)*h) * \
                    numpy.exp(-0.5*(zz / h)**2)

        rho[numpy.isnan(rho)] = 0.

        return a, rho

    def number_density(self, r, theta, phi, gas=0):
        ##### Set up the coordinates

        rt, tt, pp = numpy.meshgrid(r*AU, theta, phi,indexing='ij')

        rr = rt*numpy.sin(tt)
        zz = rt*numpy.cos(tt)

        # Get the surface density and scale height.

        Sigma = self.surface_density(rr/AU)
        h_g = self.scale_height(rr/AU)

        # Now calculate the density.

        rho = Sigma / (numpy.sqrt(2*numpy.pi)*h_g) * numpy.exp(-0.5*(zz/h_g)**2)

        rho_gas = rho * 100

        rho_gas_critical = (100. / 0.8) * 2.37*m_p
        rho_gas[rho_gas < rho_gas_critical] = 1.0e-50

        n_H2 = rho_gas * 0.8 / (2.37*m_p)

        n = n_H2 * self.abundance[gas]

        return n

    def surface_density(self, r, normalize=True):
        # Get the disk parameters.

        rin = self.rmin * AU
        rout = self.rmax * AU
        mass = self.mass * M_sun
        gamma = self.plrho - self.plh

        # Set up the surface density.

        Sigma0 = (2-gamma)*mass/(2*pi*(1*AU)**(gamma)) / \
                (rout**(-gamma+2) - rin**(-gamma+2))

        Sigma = Sigma0 * r**(-gamma)

        Sigma[(r >= rout/AU) ^ (r <= rin/AU)] = 0e0

        # In case of r == 0 (a singularity), get the value from slightly off 0.

        dr = r[r > 0].min()
        Sigma[r == 0] = Sigma0 * (0.7*dr)**(-gamma)

        # Add gaps to the disk.

        for i in range(len(self.gap_rin)):
            if self.gaussian_gaps:
                gap_r = (self.gap_rin[i] + self.gap_rout[i])/2
                gap_w = self.gap_rout[i] - self.gap_rin[i]

                Sigma /= 1 + 1./self.gap_delta[i] * numpy.exp(-4*numpy.log(2.)*\
                        (r - gap_r)**2 / gap_w**2)
            else:
                Sigma[(r >= self.gap_rin[i]) & \
                        (r <= self.gap_rout[i])] *= self.gap_delta[i]
        
        ##### Normalize the surface density correctly.
        
        if normalize:
            r_high = numpy.logspace(numpy.log10(self.rmin), \
                    numpy.log10(self.rmax), 1000)
            Sigma_high = self.surface_density(r_high, normalize=False)

            scale = mass / (2*numpy.pi*trapz(r_high*AU*Sigma_high, r_high*AU))

            Sigma *= scale

        return Sigma

    def scale_height(self, r):
        return self.h0 * AU * r**self.plh

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
        
        t = t0 * (rr / (1*AU))**(-plt)

        t[(rr >= rout) ^ (rr <= rin)] = 0e0

        t[t > 10000.] = 10000.
        
        return t

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

        tmid = tmid0 * (rr / rin)**(-pltgas)
        tatm = tatm0 * (rr / rin)**(-pltgas)

        t = numpy.zeros(tatm.shape)
        t[zz >= zq] = tatm[zz >= zq]
        t[zz < zq] = tatm[zz < zq] + (tmid[zz < zq] - tatm[zz < zq]) * \
                (numpy.cos(numpy.pi * zz[zz < zq] / (2*zq[zz < zq])))**2*delta
        
        return t

    def microturbulence(self, r, theta, phi):
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
        
        aturb = numpy.ones(rr.shape)*self.aturb*1.0e5
        
        return aturb

    def velocity(self, r, theta, phi, mstar=0.5):
        mstar *= M_sun

        rt, tt, pp = numpy.meshgrid(r*AU, theta, phi,indexing='ij')

        rr = rt*numpy.sin(tt)
        zz = rt*numpy.cos(tt)

        v_r = numpy.zeros(rr.shape)
        v_theta = numpy.zeros(rr.shape)
        v_phi = numpy.sqrt(G*mstar*rr**2/rt**3)

        return numpy.array((v_r, v_theta, v_phi))

    def read(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "r")
        else:
            f = usefile

        self.mass = f['mass'][()]
        self.rmin = f['rmin'][()]
        self.rmax = f['rmax'][()]
        self.plrho = f['plrho'][()]
        self.h0 = f['h0'][()]
        self.plh = f['plh'][()]
        if 'gamma_taper' in f:
            self.gamma_taper = gamma_taper

        self.amin = f['amin'][()]
        self.amax = f['amax'][()]
        self.pla = f['pla'][()]
        self.alpha_settle = f['alpha_settle'][()]

        if 't0' in f:
            self.t0 = f['t0'][()]
            self.plt = f['plt'][()]

        if 'tmid0' in f:
            self.tmid0 = f['tmid0'][()]
            self.tatm0 = f['tatm0'][()]
            self.zq0 = f['zq0'][()]
            self.pltgas = f['pltgas'][()]
            self.delta = f['delta'][()]

        if 'aturb' in f:
            self.aturb = f['aturb'][()]

        if ('Dust' in f):
            self.dust = Dust()
            self.dust.set_properties_from_file(usefile=f['Dust'])

        if ('Gas' in f):
            for name in f['Gas']:
                self.gas.append(Gas())
                self.abundance.append(f['Gas'][name]['Abundance'][()])
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

        if self.gamma_taper != None:
            f['gamma_taper'] = self.gamma_taper

        if self.t0 != None:
            f['t0'] = self.t0
            f['plt'] = self.plt

        if self.tmid0 != None:
            f['tmid0'] = self.tmid0
            f['tatm0'] = self.tatm0
            f['zq0'] = self.zq0
            f['pltgas'] = self.pltgas
            f['delta'] = self.delta

        if self.aturb != None:
            f['aturb'] = self.aturb

        f['amin'] = self.amin
        f['amax'] = self.amax
        f['pla'] = self.pla
        f['alpha_settle'] = self.alpha_settle

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
