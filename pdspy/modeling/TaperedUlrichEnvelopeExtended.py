import numpy
from scipy.optimize import brenth
from scipy.integrate import trapz
from astropy.constants import au, M_sun, G, m_p
from numpy import pi
from astropy.units import year
from ..dust import Dust
from ..gas import Gas

class TaperedUlrichEnvelopeExtended:

    def __init__(self, mass=1.0e-3, rmin=0.1, rmax=1000, rcent=30, cavpl=1.0, \
            cavrfact=0.2, gamma=1., theta_open=1., zoffset=1., t0=None, \
            tpl=None, dust=None, aturb=None):
        self.mass = mass
        self.rmin = rmin
        self.rmax= rmax
        self.rcent = rcent
        self.cavpl = cavpl
        self.cavrfact = cavrfact
        self.gamma = gamma
        self.t0 = t0
        self.tpl = tpl
        self.theta_open = theta_open
        self.zoffset = zoffset
        if (dust != None):
            self.dust = dust

        self.gas = []
        self.abundance = []
        self.aturb = aturb

    def add_gas(self, gas, abundance):
        self.gas.append(gas)
        self.abundance.append(abundance)

    def density(self, r, theta, phi, code="hyperion"):
        #numpy.seterr(all='ignore')
        ##### Star parameters

        mstar = M_sun.cgs.value
        
        ##### Envelope parameters

        rin = self.rmin * au.cgs.value
        rout = self.rmax * au.cgs.value
        mass = self.mass * M_sun.cgs.value
        rcent = self.rcent * au.cgs.value
        cavz0 = self.zoffset * au.cgs.value
        cavpl = self.cavpl
        cavrfact = self.cavrfact
        gamma = self.gamma
        theta_open = self.theta_open * numpy.pi / 180.

        cava = (rout*numpy.sin(theta_open/2)/au.cgs.value / numpy.tan(theta_open/2) - \
                cavz0/au.cgs.value) * (rout*numpy.sin(theta_open/2)/au)**-cavpl

        # Set up the coordinates.
        
        rr, tt, pp = numpy.meshgrid(r*au.cgs.value, theta, phi,indexing='ij')

        mu = numpy.cos(tt)

        RR = rr * numpy.sin(tt)
        zz = rr * numpy.cos(tt)

        # Calculate mu0 at each r, theta combination.

        def func(mu0,r,mu,R_c):
            return mu0**3-mu0*(1-r/R_c)-mu*(r/R_c)

        mu0 = mu*0.
        for ir in range(rr.shape[0]):
            for it in range(rr.shape[1]):
                mu0[ir,it,0] = brenth(func,-1.0,1.0,args=(rr[ir,it,0], \
                        mu[ir,it,0],rcent))

        ##### Make the dust density model for an Ulrich envelope.

        rho0 = 1.0
        
        rho = rho0 * (rr / rcent)**(-1.5) * (1 + mu/mu0)**(-0.5)* \
                (mu/mu0 + 2*mu0**2 * rcent/rr)**(-1)

        mid1 = (numpy.abs(mu) < 1.0e-10) & (rr < rcent)
        rho[mid1] = rho0 * (rr[mid1] / rcent)**(-0.5) * \
                (1. - rr[mid1] / rcent)**(-1) / 2.

        mid2 = (numpy.abs(mu) < 1.0e-10) & (rr > rcent)
        rho[mid2] = rho0 * (2.*rr[mid2]/rcent - 1)**(-0.5) * \
                (rr[mid2]/rcent - 1.)**(-1)

        rho *= numpy.exp(-(rr/rout)**(2-gamma))
        rho[rr <= rin] = 0e0

        ##### Add an outflow cavity.

        rho[numpy.abs(zz)/au.cgs.value-cavz0/au-cava*(RR/au)**cavpl > 0.0] *= cavrfact
        
        ##### Normalize the mass correctly.
        
        if theta.max() > pi/2:
            mdot = mass/(2*pi*trapz(trapz(rho*rr**2*numpy.sin(tt),tt,axis=1), \
                    rr[:,0,:],axis=0))[0]
        else:
            mdot = mass/(4*pi*trapz(trapz(rho*rr**2*numpy.sin(tt),tt,axis=1), \
                    rr[:,0,:],axis=0))[0]
        rho *= mdot

        #numpy.seterr(all='warn')

        return rho

    def temperature(self, r, theta, phi):
        ##### Disk Parameters
        
        rin = self.rmin * au.cgs.value
        rout = self.rmax * au.cgs.value
        t0 = self.t0
        tpl = self.tpl

        ##### Set up the coordinates

        rt, tt, pp = numpy.meshgrid(r*au.cgs.value, theta, phi,indexing='ij')

        rr = rt*numpy.sin(tt)
        zz = rt*numpy.cos(tt)

        ##### Make the dust density model for a protoplanetary disk.
        
        t = t0 * (rt / (1*au.cgs.value))**(-tpl)

        t[rt <= rin] = 0e0

        t[t > 10000.] = 10000.
        
        return t

    def number_density(self, r, theta, phi, gas=0):
        rho = self.density(r, theta, phi)

        rho_gas = rho * 100

        n_H2 = rho_gas * 0.8 / (2.37*m_p.cgs.value)

        n = n_H2 * self.abundance[gas]

        return n

    def microturbulence(self, r, theta, phi):
        ##### Set up the coordinates

        rt, tt, pp = numpy.meshgrid(r*au.cgs.value, theta, phi,indexing='ij')

        rr = rt*numpy.sin(tt)
        zz = rt*numpy.cos(tt)

        ##### Make the dust density model for a protoplanetary disk.
        
        aturb = numpy.ones(rr.shape)*self.aturb*1.0e5
        
        return aturb

    def velocity(self, r, theta, phi, mstar=0.5):
        mstar *= M_sun.cgs.value
        rcent = self.rcent * au.cgs.value

        # Set up the coordinates.
        
        rr, tt, pp = numpy.meshgrid(r*au.cgs.value, theta, phi,indexing='ij')

        mu = numpy.cos(tt)

        # Calculate mu0 at each r, theta combination.

        def func(mu0,r,mu,R_c):
            return mu0**3-mu0*(1-r/R_c)-mu*(r/R_c)

        mu0 = mu*0.
        for ir in range(rr.shape[0]):
            for it in range(rr.shape[1]):
                mu0[ir,it,0] = brenth(func,-1.0,1.0,args=(rr[ir,it,0], \
                        mu[ir,it,0],rcent))

        v_r = -numpy.sqrt(G.cgs.value*mstar/rr)*numpy.sqrt(1 + mu/mu0)
        v_theta = numpy.sqrt(G.cgs.value*mstar/rr) * (mu0 - mu) * \
                numpy.sqrt((mu0 + mu) / (mu0 * numpy.sin(tt)**2))
        v_phi = numpy.sqrt(G.cgs.value*mstar/rr) * numpy.sqrt((1 - mu0**2)/(1 - mu**2)) *\
                numpy.sqrt(1 - mu/mu0)

        return numpy.array((v_r, v_theta, v_phi))

    def calculate_mdot(self, r, theta, phi, mstar=0.5, code="hyperion"):
        #numpy.seterr(all='ignore')
        ##### Star parameters

        mstar *= M_sun.cgs.value
        
        ##### Envelope parameters

        rin = self.rmin * au.cgs.value
        rout = self.rmax * au.cgs.value
        mass = 100 * self.mass * M_sun.cgs.value
        rcent = self.rcent * au.cgs.value
        cavz0 = 1*au.cgs.value
        cavpl = self.cavpl
        cavrfact = self.cavrfact

        # Set up the coordinates.
        
        rr, tt, pp = numpy.meshgrid(r*au.cgs.value, theta, phi,indexing='ij')

        mu = numpy.cos(tt)

        RR = rr * numpy.sin(tt)
        zz = rr * numpy.cos(tt)

        # Calculate mu0 at each r, theta combination.

        def func(mu0,r,mu,R_c):
            return mu0**3-mu0*(1-r/R_c)-mu*(r/R_c)

        mu0 = mu*0.
        for ir in range(rr.shape[0]):
            for it in range(rr.shape[1]):
                mu0[ir,it,0] = brenth(func,-1.0,1.0,args=(rr[ir,it,0], \
                        mu[ir,it,0],rcent))

        ##### Make the dust density model for an Ulrich envelope.

        rho0 = 1.0
        
        rho = rho0 * (rr / rcent)**(-1.5) * (1 + mu/mu0)**(-0.5)* \
                (mu/mu0 + 2*mu0**2 * rcent/rr)**(-1)

        mid1 = (numpy.abs(mu) < 1.0e-10) & (rr < rcent)
        rho[mid1] = rho0 * (rr[mid1] / rcent)**(-0.5) * \
                (1. - rr[mid1] / rcent)**(-1) / 2.

        mid2 = (numpy.abs(mu) < 1.0e-10) & (rr > rcent)
        rho[mid2] = rho0 * (2.*rr[mid2]/rcent - 1)**(-0.5) * \
                (rr[mid2]/rcent - 1.)**(-1)

        rho[(rr >= rout) ^ (rr <= rin)] = 0e0

        ##### Normalize the mass correctly.
        
        if theta.max() > pi/2:
            prefix = mass/(2*pi*trapz(trapz(rho*rr**2*numpy.sin(tt),tt,axis=1),\
                    rr[:,0,:],axis=0))[0]
        else:
            prefix = mass/(4*pi*trapz(trapz(rho*rr**2*numpy.sin(tt),tt,axis=1),\
                    rr[:,0,:],axis=0))[0]

        mdot = prefix * (4*numpy.pi * numpy.sqrt(G.cgs.value * mstar) * rcent**1.5) / \
                (M_sun.cgs.value / year.cgs.value)

        return mdot


    def read(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "r")
        else:
            f = usefile

        self.mass = f['mass'][()]
        self.rmin = f['rmin'][()]
        self.rmax = f['rmax'][()]
        self.rcent = f['rcent'][()]
        self.cavpl = f['cavpl'][()]
        self.cavrfact = f['cavrfact'][()]
        self.gamma = f['gamma'][()]
        self.theta_open = f['theta_open'][()]
        self.zoffset = f['zoffset'][()]

        if 't0' in f:
            self.t0 = f['t0'][()]
            self.tpl = f['tpl'][()]

        if 'aturb' in f:
            self.aturb = f['aturb'][()]

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
        f['rcent'] = self.rcent
        f['cavpl'] = self.cavpl
        f['cavrfact'] = self.cavrfact
        f['gamma'] = self.gamma
        f['theta_open'] = self.theta_open
        f['zoffset'] = self.zoffset

        if self.t0 != None:
            f['t0'] = self.t0
            f['plt'] = self.plt

        if self.aturb != None:
            f['aturb'] = self.aturb

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
