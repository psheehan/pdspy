import numpy
from scipy.optimize import brenth
from scipy.integrate import trapz
from ..constants.astronomy import AU, M_sun
from ..constants.math import pi
from ..constants.physics import G

def protoplanetary_disk(r, theta, rin=0.1, rout=200, mass=1.0e-3, plrho=2.37, \
        h=0.1, plh=58.0/45.0):
    
    ##### Disk Parameters
    
    rin *= AU
    rout *= AU
    mass *= M_sun
    h *= AU

    ##### Set up the coordinates

    rt, tt = numpy.meshgrid(0.5*(r[0:r.size-1]+r[1:r.size])*AU, \
            0.5*(theta[0:theta.size-1]+theta[1:theta.size]))

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

def ulrich_envelope(r, theta, mstar=0.5, rin=0.1, rout=500, mass=1.0e-3, \
        rcent=100, cavpl=1.0, cavrfact=0.2):
    
    ##### Star parameters

    mstar *= M_sun
    
    ##### Envelope parameters

    rin *= AU
    rout *= AU
    mass *= M_sun
    rcent *= AU
    cavz0 = 1*AU

    # Set up the coordinates.
    
    rr, tt = numpy.meshgrid(0.5*(r[0:r.size-1]+r[1:r.size])*AU, \
            0.5*(theta[0:theta.size-1]+theta[1:theta.size]))

    mu = numpy.cos(tt)

    # Calculate mu0 at each r, theta combination.

    def func(mu0,r,mu,R_c):
        return mu0**3-mu0*(1-r/R_c)-mu*(r/R_c)

    mu0 = mu*0.
    for ir in range(rr.shape[0]):
        for it in range(rr.shape[1]):
            mu0[ir,it] = brenth(func,-1.0,1.0,args=(rr[ir,it],mu[ir,it],rcent))

    ##### Make the dust density model for an Ulrich envelope.
    
    numpy.seterr(divide='ignore')
    rho = 1.0/(4*pi)*(G*mstar*rr**3)**(-0.5)*(1+mu/mu0)**(-0.5)* \
            (mu/mu0+2*mu0**2*rcent/rr)**(-1)

    mid1 = (numpy.abs(mu) < 1.0e-10) & (rr < rcent)

    mid2 = (numpy.abs(mu) < 1.0e-10) & (rr > rcent)

    numpy.seterr(divide='warn')
    rho[(rr >= rout) ^ (rr <= rin)] = 0e0

    ##### Normalize the mass correctly.
    
    mdot = mass/(4*pi*trapz(trapz(rho*rr**2*numpy.sin(tt),tt,axis=0),rr[0,:]))
    rho *= mdot

    ##### Add an outflow cavity.

    zz = rr*numpy.cos(tt)
    rho[zz-cavz0-(rr*numpy.sin(tt))**cavpl > 0.0] *= cavrfact
    
    return rho
