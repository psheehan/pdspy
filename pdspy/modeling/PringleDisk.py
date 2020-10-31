import numpy
import h5py
from scipy.integrate import trapz
from ..constants.physics import G, m_p
from ..constants.astronomy import AU, M_sun
from ..constants.math import pi
from ..dust import Dust
from ..gas import Gas
from .Disk import Disk

class PringleDisk(Disk):
    def surface_density(self, r, normalize=True):
        # Get the disk parameters.

        rr = r * AU
        rin = self.rmin * AU
        rout = self.rmax * AU
        mass = self.mass * M_sun
        gamma = self.plrho - self.plh
        if self.gamma_taper != None:
            gamma_taper = self.gamma_taper
        else:
            gamma_taper = gamma

        # Set up the surface density.

        Sigma0 = (2-gamma)*mass/(2*pi*rout**2)*numpy.exp((rin/rout)**(2-gamma))

        Sigma = Sigma0 * (rr/rout)**(-gamma) * \
                numpy.exp(-(rr/rout)**(2-gamma_taper))

        Sigma[r <= rin/AU] = 0e0

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
                    numpy.log10(10*self.rmax), 1000)
            Sigma_high = self.surface_density(r_high, normalize=False)

            scale = mass / (2*numpy.pi*trapz(r_high*AU*Sigma_high, r_high*AU))

            Sigma *= scale

        return Sigma

    def scale_height(self, r):
        return self.h0 * AU * (r / self.rmax)**self.plh

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

        t[rr <= rin] = 0e0

        t[t > 10000] = 10000
        
        return t

    def temperature_1d(self, r):
        rin = self.rmin * AU
        rout = self.rmax * AU
        t0 = self.t0
        plt = self.plt

        T = t0 * r**(-plt)

        T[r <= rin/AU] = 0.0

        dr = r[r > 0].min()
        T[r == 0] = t0 * (0.7*dr)**(-plt)

        return T
