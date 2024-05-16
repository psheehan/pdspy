import numpy
import h5py
from astropy.constants import G, m_p, au, M_sun
from numpy import pi
from ..dust import Dust
from ..gas import Gas
from .TwoLayerDisk import TwoLayerDisk

class TwoLayerPringleDisk(TwoLayerDisk):
    def surface_density(self, r, normalize=True):
        # Get the disk parameters.

        rr = r * au.cgs.value
        rin = self.rmin * au.cgs.value
        rout = self.rmax * au.cgs.value
        mass = self.mass * M_sun.cgs.value
        gamma = self.plrho - self.plh
        if self.gamma_taper != None:
            gamma_taper = self.gamma_taper
        else:
            gamma_taper = gamma

        # Set up the surface density.

        Sigma0 = (2-gamma)*mass/(2*pi*rout**2)*numpy.exp((rin/rout)**(2-gamma))

        Sigma = Sigma0 * (rr/rout)**(-gamma) * \
                numpy.exp(-(rr/rout)**(2-gamma_taper))

        Sigma[r <= rin/au.cgs.value] = 0e0

        # In case of r == 0 (a singularity), get the value from slightly off 0.

        dr = rr[r > 0].min()
        Sigma[r == 0] = Sigma0 * (0.7*dr/rout)**(-gamma) * \
                numpy.exp(-(0.7*dr/rout)**(2-gamma))

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

            scale = mass / (2*numpy.pi*trapz(r_high*au.cgs.value*Sigma_high, r_high*au))

            Sigma *= scale

        return Sigma

    def scale_height(self, r):
        return self.h0 * au.cgs.value * (r / self.rmax)**self.plh

    def temperature(self, r, theta, phi):
        ##### Disk Parameters
        
        rin = self.rmin * au.cgs.value
        rout = self.rmax * au.cgs.value
        t0 = self.t0
        plt = self.plt

        ##### Set up the coordinates

        rt, tt, pp = numpy.meshgrid(r*au.cgs.value, theta, phi,indexing='ij')

        rr = rt*numpy.sin(tt)
        zz = rt*numpy.cos(tt)

        ##### Make the dust density model for a protoplanetary disk.
        
        t = t0 * (rr / (1*au.cgs.value))**(-plt)

        t[rr <= rin] = 0e0

        t[t > 10000] = 10000
        
        return t

    def temperature_1d(self, r):
        rin = self.rmin * au.cgs.value
        rout = self.rmax * au.cgs.value
        t0 = self.t0
        plt = self.plt

        T = t0 * r**(-plt)

        T[r <= rin/au.cgs.value] = 0.0

        dr = r[r > 0].min()
        T[r == 0] = t0 * (0.7*dr)**(-plt)

        return T

    def gas_temperature(self, r, theta, phi):
        ##### Disk Parameters
        
        rin = self.rmin * au.cgs.value
        rout = self.rmax * au.cgs.value
        pltgas = self.pltgas
        tmid0 = self.tmid0
        tatm0 = self.tatm0
        zq0 = self.zq0
        delta = self.delta

        ##### Set up the coordinates

        rt, tt, pp = numpy.meshgrid(r*au.cgs.value, theta, phi,indexing='ij')

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
