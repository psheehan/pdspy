import numpy
from scipy.optimize import brenth
from scipy.integrate import trapz
from ..constants.astronomy import AU, M_sun
from ..constants.math import pi
from ..constants.physics import G, m_p
from ..dust import Dust
from ..gas import Gas

class Envelope:

    def __init__(self, mass=1.0e-3, rmin=0.1, rmax=1000, pl=1.5, cavpl=1.0, \
            cavrfact=0.2, t0=None, tpl=None, dust=None):
        self.mass = mass
        self.rmin = rmin
        self.rmax= rmax
        self.pl = pl
        self.cavpl = cavpl
        self.cavrfact = cavrfact
        self.t0 = t0
        self.tpl = tpl
        if (dust != None):
            self.dust = dust
        self.gas = []
        self.abundance = []

    def add_gas(self, gas, abundance):
        self.gas.append(gas)
        self.abundance.append(abundance)

    def density(self, r, theta, phi, code="hyperion"):
        #numpy.seterr(all='ignore')
        ##### Star parameters

        mstar = M_sun
        
        ##### Envelope parameters

        rin = self.rmin * AU
        rout = self.rmax * AU
        mass = self.mass * M_sun
        pl = self.pl
        cavz0 = 1*AU
        cavpl = self.cavpl
        cavrfact = self.cavrfact

        # Set up the coordinates.
        
        rr, tt, pp = numpy.meshgrid(r*AU, theta, phi,indexing='ij')

        mu = numpy.cos(tt)

        ##### Make the dust density model for an Ulrich envelope.

        rho0 = 1.0
        
        rho = rho0 * (rr / rin)**(-pl)

        rho[(rr >= rout) ^ (rr <= rin)] = 0e0

        ##### Normalize the mass correctly.
        
        if theta.max() > pi/2:
            mdot = mass/(2*pi*trapz(trapz(rho*rr**2*numpy.sin(tt),tt,axis=1), \
                    rr[:,0,:],axis=0))[0]
        else:
            mdot = mass/(4*pi*trapz(trapz(rho*rr**2*numpy.sin(tt),tt,axis=1), \
                    rr[:,0,:],axis=0))[0]
        rho *= mdot

        ##### Add an outflow cavity.

        zz = rr*numpy.cos(tt)
        rho[numpy.abs(zz)-cavz0-(rr*numpy.sin(tt))**cavpl > 0.0] *= cavrfact
        
        #numpy.seterr(all='warn')

        return rho

    def temperature(self, r, theta, phi):
        ##### Disk Parameters
        
        rin = self.rmin * AU
        rout = self.rmax * AU
        t0 = self.t0
        tpl = self.tpl

        ##### Set up the coordinates

        rt, tt, pp = numpy.meshgrid(r*AU, theta, phi,indexing='ij')

        rr = rt*numpy.sin(tt)
        zz = rt*numpy.cos(tt)

        ##### Make the dust density model for a protoplanetary disk.
        
        t = t0 * (rt / rin)**(-tpl)

        t = numpy.sqrt(t**2 + 25.**2)

        t[(rt >= rout) ^ (rt <= rin)] = 0e0
        
        return t

    def number_density(self, r, theta, phi, gas=0):
        rho = self.density(r, theta, phi)

        n_H2 = rho * 100. / (2*m_p)

        n = n_H2 * self.abundance[gas]

        return n

    def velocity(self, r, theta, phi, mstar=0.5):
        mstar *= M_sun
        rcent = self.rcent * AU

        # Set up the coordinates.
        
        rr, tt, pp = numpy.meshgrid(r*AU, theta, phi,indexing='ij')

        mu = numpy.cos(tt)

        # Calculate mu0 at each r, theta combination.

        def func(mu0,r,mu,R_c):
            return mu0**3-mu0*(1-r/R_c)-mu*(r/R_c)

        mu0 = mu*0.
        for ir in range(rr.shape[0]):
            for it in range(rr.shape[1]):
                mu0[ir,it,0] = brenth(func,-1.0,1.0,args=(rr[ir,it,0], \
                        mu[ir,it,0],rcent))

        v_r = -numpy.sqrt(G*mstar/rr)*numpy.sqrt(1 + mu/mu0)
        v_theta = numpy.sqrt(G*mstar/rr) * (mu0 - mu) * \
                numpy.sqrt((mu0 + mu) / (mu0 * numpy.sin(tt)))
        v_phi = numpy.sqrt(G*mstar/rr) * numpy.sqrt((1 - mu0**2)/(1 - mu**2)) *\
                numpy.sqrt(1 - mu/mu0)

        return numpy.array((v_r, v_theta, v_phi))

    def read(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "r")
        else:
            f = usefile

        self.mass = f['mass'][()]
        self.rmin = f['rmin'][()]
        self.rmax = f['rmax'][()]
        self.pl = f['pl'][()]
        self.cavpl = f['cavpl'][()]
        self.cavrfact = f['cavrfact'][()]

        if ('Dust' in f):
            self.dust = Dust()
            self.dust.set_properties_from_file(usefile=f['Dust'])

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
        f['pl'] = self.pl
        f['cavpl'] = self.cavpl
        f['cavrfact'] = self.cavrfact

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
