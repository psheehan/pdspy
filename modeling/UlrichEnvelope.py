import numpy
from scipy.optimize import brenth
from scipy.integrate import trapz
from ..constants.astronomy import AU, M_sun
from ..constants.math import pi
from ..constants.physics import G
from ..dust import Dust

class UlrichEnvelope:

    def __init__(self, mass=1.0e-3, rmin=0.1, rmax=1000, rcent=30, cavpl=1.0, \
            cavrfact=0.2, dust=None):
        self.mass = mass
        self.rmin = rmin
        self.rmax= rmax
        self.rcent = rcent
        self.cavpl = cavpl
        self.cavrfact = cavrfact
        if (dust != None):
            self.dust = dust

    def density(self, r, theta, phi):
        #numpy.seterr(all='ignore')
        ##### Star parameters

        mstar = M_sun
        
        ##### Envelope parameters

        rin = self.rmin * AU
        rout = self.rmax * AU
        mass = self.mass * M_sun
        rcent = self.rcent * AU
        cavz0 = 1*AU
        cavpl = self.cavpl
        cavrfact = self.cavrfact

        # Set up the coordinates.
        
        rr, tt, pp = numpy.meshgrid(0.5*(r[0:r.size-1]+r[1:r.size])*AU, \
                0.5*(theta[0:theta.size-1]+theta[1:theta.size]), \
                0.5*(phi[0:phi.size-1]+phi[1:phi.size]),indexing='ij')

        mu = numpy.cos(tt)

        # Calculate mu0 at each r, theta combination.

        def func(mu0,r,mu,R_c):
            return mu0**3-mu0*(1-r/R_c)-mu*(r/R_c)

        mu0 = mu*0.
        for ir in range(rr.shape[0]):
            for it in range(rr.shape[1]):
                mu0[ir,it,0] = brenth(func,-1.0,1.0,args=(rr[ir,it,0], \
                        mu[ir,it,0],rcent))

        ##### Make the dust density model for an Ulrich envelope.
        
        rho = 1.0/(4*pi)*(G*mstar*rr**3)**(-0.5)*(1+mu/mu0)**(-0.5)* \
                (mu/mu0+2*mu0**2*rcent/rr)**(-1)

        mid1 = (numpy.abs(mu) < 1.0e-10) & (rr < rcent)

        mid2 = (numpy.abs(mu) < 1.0e-10) & (rr > rcent)

        rho[(rr >= rout) ^ (rr <= rin)] = 0e0

        ##### Normalize the mass correctly.
        
        mdot = mass/(4*pi*trapz(trapz(rho*rr**2*numpy.sin(tt),tt,axis=1), \
                rr[:,0,:],axis=0))[0]
        rho *= mdot

        ##### Add an outflow cavity.

        zz = rr*numpy.cos(tt)
        rho[zz-cavz0-(rr*numpy.sin(tt))**cavpl > 0.0] *= cavrfact
        
        #numpy.seterr(all='warn')

        return rho

    def read(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "r")
        else:
            f = usefile

        self.mass = f['mass'].value
        self.rmin = f['rmin'].value
        self.rmax = f['rmax'].value
        self.rcent = f['rcent'].value
        self.cavpl = f['cavpl'].value
        self.cavrfact = f['cavrfact'].value

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
        f['rcent'] = self.rcent
        f['cavpl'] = self.cavpl
        f['cavrfact'] = self.cavrfact

        if hasattr(self, 'dust'):
            dust = f.create_group("Dust")
            self.dust.write(usefile=dust)

        if (usefile == None):
            f.close()
