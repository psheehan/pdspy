import scipy.integrate
import numpy
from ..constants.physics import G, m_p, k, m_p
from ..constants.astronomy import AU, M_sun
from .Disk import Disk

class DartoisDisk(Disk):
    def log_gas_density_high(self, mstar=0.5):
        # Set up the high resolution grid.

        r = numpy.logspace(numpy.log10(self.rmin), numpy.log10(5*self.rmax), \
                1000)
        z = numpy.hstack(([0], numpy.logspace(-3., numpy.log10(5*self.rmax), \
                1000)))
        phi = numpy.array([0.,2*numpy.pi])

        # Calculate the temperature structure.

        T = self.temperature(r, phi, z, coordsys="cylindrical")[:,0,:]

        # Now make the high resolution coordinates into 3D arrays.

        r, z = numpy.meshgrid(r, z, indexing='ij')

        # Now calculate the derivative of T.

        dlogT = numpy.gradient(numpy.log(T), z[0,:]*AU, axis=1)

        # Calculate the balance of gravity and pressure

        mu = 2.37
        cs = (k*T/(mu*m_p))**0.5
        M = mstar*M_sun

        gravity = 1./cs**2 * G*M*z*AU / ((r*AU)**2 + (z*AU)**2)**1.5

        # Now vertically integrate to get -logDens.

        logDens = -scipy.integrate.cumtrapz(dlogT+gravity, x=z*AU, initial=0., \
                axis=1)

        # Get the proper normalization.

        norm = 0.5*100*self.surface_density(r[:,0]) / scipy.integrate.trapz(\
                numpy.exp(logDens), x=z*AU, axis=1)

        norm_arr = numpy.zeros(r.shape)
        for i in range(r.shape[1]):
            norm_arr[:,i] = norm

        # Finally, calculate the scaled density.

        with numpy.errstate(divide="ignore"):
            logDens += numpy.where(norm_arr > 0, numpy.log(norm_arr), -300.)

        return r, z, logDens

    def log_number_density_high(self, mstar=0.5, gas=0):
        # Get the high resolution gas density in cylindrical coordinates.

        r, z, logrho_gas = self.log_gas_density_high(mstar=mstar)

        logn_H2 = logrho_gas + numpy.log(0.8 / (2.37*m_p))

        if gas < 0:
            return r, z, logn_H2
        else:
            # Take account of the abundance.

            logn = logn_H2 + numpy.log(self.abundance[gas])

            # Account for freezeout as well.

            T = self.temperature(r[:,0], numpy.array([0.,2*numpy.pi]), z[0,:], \
                    coordsys="cylindrical")[:,0,:]

            frozen = T < self.freezeout[gas]

            # Account for photodissociation.

            n_H2 = numpy.exp(logn_H2)

            cumn = -scipy.integrate.cumtrapz(n_H2[:,::-1], x=z[:,::-1]*AU, \
                    axis=1, initial=0.)[:,::-1]

            dissociated = cumn < 0.79 * 1.59e21 / 0.706

            # Now actually reduce the density.

            logn[numpy.logical_or(frozen, dissociated)] += numpy.log(1.0e-8)

            return r, z, logn

    def log_gas_pressure_high(self, mstar=0.5):
        # Get the number density and temperature.

        r, z, logn = self.log_number_density_high(mstar=mstar, gas=-1)

        T = self.temperature(r[:,0], numpy.array([0.,2*numpy.pi]), z[0,:], \
                coordsys="cylindrical")[:,0,:]

        logT = numpy.log(T)

        # And calculate the pressure.

        logP = logn + logT + numpy.log(k)

        return r, z, logP


    def gas_density(self, x1, x2, x3, coordsys="spherical", mstar=0.5):
        ##### Set up the coordinates

        if coordsys == "spherical":
            rt, tt, pp = numpy.meshgrid(x1*AU, x2, x3, indexing='ij')

            rr = rt*numpy.sin(tt)
            zz = rt*numpy.cos(tt)
        elif coordsys == "cartesian":
            xx, yy, zz = numpy.meshgrid(x1*AU, x2*AU, x3*AU, indexing='ij')

            rr = (xx**2 + yy**2)**0.5
        elif coordsys == "cylindrical":
            rr, pp, zz = numpy.meshgrid(x1*AU, x2, x3*AU, indexing='ij')

        # Get the high resolution log of the gas density.

        r, z, logDens = self.log_gas_density_high(mstar=mstar)

        # Now, interpolate that density onto the actual grid of interest.

        f = scipy.interpolate.RegularGridInterpolator((r[:,0],z[0,:]), logDens,\
                bounds_error=False, fill_value=-300.)

        points = numpy.empty((rr.size, 2))
        points[:,0] = rr.reshape((-1,))/AU
        points[:,1] = zz.reshape((-1,))/AU

        logDens_interp = f(points).reshape(rr.shape)

        Dens = numpy.exp(logDens_interp)

        return Dens

    def number_density(self, w1, w2, w3, gas=0, coordsys="spherical", \
            mstar=0.5):
        # Get the high resolution log of the gas density.

        r, z, logn = self.log_number_density_high(mstar=mstar, gas=gas)

        rho = numpy.sqrt(r**2 + z**2)
        theta = numpy.pi/2 - numpy.arctan(z / r)

        # Now, interpolate that density onto the actual grid of interest by
        # averaging all of the high resolution points that fall within the 
        # lower resolution cell.

        n = numpy.exp(logn)

        n_interp = scipy.stats.binned_statistic_2d(rho.flatten(), \
                theta.flatten(), n.flatten(), statistic="mean", \
                bins=(w1, w2)).statistic[:,:,numpy.newaxis]

        n_interp[numpy.isnan(n_interp)] = 0.

        return n_interp

    def temperature(self, x1, x2, x3, coordsys="spherical"):
        ##### Disk Parameters
        
        rin = self.rmin * AU
        rout = self.rmax * AU
        pltgas = self.pltgas
        tmid0 = self.tmid0
        tatm0 = self.tatm0
        zq0 = self.zq0 * AU
        delta = self.delta

        ##### Set up the coordinates

        if coordsys == "spherical":
            rt, tt, pp = numpy.meshgrid(x1*AU, x2, x3, indexing='ij')

            rr = rt*numpy.sin(tt)
            zz = rt*numpy.cos(tt)
        elif coordsys == "cartesian":
            xx, yy, zz = numpy.meshgrid(x1*AU, x2*AU, x3*AU, indexing='ij')

            rr = (xx**2 + yy**2)**0.5
        elif coordsys == "cylindrical":
            rr, pp, zz = numpy.meshgrid(x1*AU, x2, x3*AU, indexing='ij')

        ##### Make the dust density model for a protoplanetary disk.
        
        zq = zq0 * (rr / (1*AU))**1.3

        with numpy.errstate(divide="ignore"):
            tmid = numpy.where(rr > 0, tmid0 * (rr / (1*AU))**(-pltgas), 0.1)
            tatm = numpy.where(rr > 0, tatm0 * (rr / (1*AU))**(-pltgas), 0.1)

        t = numpy.zeros(tatm.shape)
        t[zz >= zq] = tatm[zz >= zq]
        t[zz < zq] = tatm[zz < zq] + (tmid[zz < zq] - tatm[zz < zq]) * \
                (numpy.cos(numpy.pi * zz[zz < zq] / (2*zq[zz < zq])))**(2*delta)

        ##### Catch an temperatures greater than 10000 K and set to that value,
        ##### as that is likely unphysical.

        t[t > 10000] = 10000
        
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

        # Calculate the sound speed.

        T = self.temperature(r, theta, phi, coordsys="spherical")

        # Parameterize in terms of the sound speed.
        
        mu = 2.37
        cs = (k*T/(mu*m_p))**0.5

        aturb = cs*self.aturb
        
        return aturb

    def velocity(self, r, theta, phi, mstar=0.5):
        rt, tt, pp = numpy.meshgrid(r*AU, theta, phi,indexing='ij')

        rr = rt*numpy.sin(tt)
        zz = rt*numpy.cos(tt)

        v_r = numpy.zeros(rr.shape)
        v_theta = numpy.zeros(rr.shape)

        # Calculate the Keplerian velocity.

        v_phi_kepler = numpy.sqrt(G*mstar*M_sun*rr**2/rt**3)

        # Calculate the pressure contribution to the velocity.

        R, z, logP = self.log_gas_pressure_high(mstar=mstar)
        R, z, logDens = self.log_gas_density_high(mstar=mstar)

        P = numpy.exp(logP)
        Dens = numpy.exp(logDens)
        Dens[logDens < -200.] = 0.

        dP = numpy.gradient(P, R[:,0]*AU, axis=0)

        with numpy.errstate(divide="ignore", invalid="ignore"):
            v_phi_pressure2 = numpy.where(Dens > 0, dP / Dens * R*AU, 0.)

        # Now, interpolate onto the proper grid.

        f = scipy.interpolate.RegularGridInterpolator((R[:,0],z[0,:]), \
                v_phi_pressure2, bounds_error=False, fill_value=0.)

        points = numpy.empty((rr.size, 2))
        points[:,0] = rr.reshape((-1,))/AU
        points[:,1] = zz.reshape((-1,))/AU

        v_phi_pressure2_interp = f(points).reshape(rr.shape)

        # Finally, add the two together.

        with numpy.errstate(divide="ignore", invalid="ignore"):
            v_phi = numpy.where(v_phi_kepler**2 + v_phi_pressure2_interp > 0, \
                    numpy.sqrt(v_phi_kepler**2 + v_phi_pressure2_interp), 1.0)

        return numpy.array((v_r, v_theta, v_phi))

