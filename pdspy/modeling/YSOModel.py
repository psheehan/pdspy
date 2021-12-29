import numpy
import h5py
from .Model import Model
from .Disk import Disk
from .DartoisDisk import DartoisDisk
from .TwoLayerDisk import TwoLayerDisk
from .SettledDisk import SettledDisk
from .PringleDisk import PringleDisk
from .DartoisPringleDisk import DartoisPringleDisk
from .SettledPringleDisk import SettledPringleDisk
from .TwoLayerPringleDisk import TwoLayerPringleDisk
from .Envelope import Envelope
from .UlrichEnvelope import UlrichEnvelope
from .UlrichEnvelopeExtended import UlrichEnvelopeExtended
from .TaperedUlrichEnvelope import TaperedUlrichEnvelope
from .TaperedUlrichEnvelopeExtended import TaperedUlrichEnvelopeExtended
from .Star import Star
from ..constants.physics import h, c, G, m_p, k
from ..constants.astronomy import AU, M_sun, kms, R_sun, Jy, pc
from ..constants.math import pi
from ..misc import B_nu
from ..imaging import Image, imtovis

class YSOModel(Model):
    r"""
    A Model that specifically represents a young star, including a star, disk, and envelope.
    """

    def add_star(self, mass=0.5, luminosity=1, temperature=4000.):
        self.grid.add_star(Star(mass=mass, luminosity=luminosity, \
                temperature=temperature))

    def set_cartesian_grid(self, xmin, xmax, nx):
        x = numpy.linspace(xmin, xmax, nx)
        y = numpy.linspace(xmin, xmax, nx)
        z = numpy.linspace(xmin, xmax, nx)

        self.grid.set_cartesian_grid(x, y, z)

    def set_cylindrical_grid(self, rmin, rmax, nr, nz, nphi):
        r = numpy.linspace(rmin, rmax, nr)
        phi = numpy.linspace(0.0, 2*numpy.pi, nphi)
        z = numpy.linspace(0.,rmax, nz)

        self.grid.set_cylindrical_grid(r, phi, z)

    def set_spherical_grid(self, rmin, rmax, nr, ntheta, nphi, log=True, \
            code="radmc3d"):
        if log:
            r = numpy.logspace(numpy.log10(rmin), numpy.log10(rmax), nr)
        else:
            r = numpy.linspace(rmin, rmax, nr)
        if (code == "hyperion"):
            r = numpy.hstack([0.0,r])

        if (code == "radmc3d"):
            theta = numpy.linspace(0.0, numpy.pi/2, ntheta)
        elif (code == "hyperion"):
            theta = numpy.linspace(0.0, numpy.pi, ntheta)

        phi = numpy.linspace(0.0, 2*numpy.pi, nphi)

        self.grid.set_spherical_grid(r, theta, phi)

    def add_ambient_medium(self, dens=1.0e-24):
        for i in range(len(self.grid.density)):
            self.grid.density[i] += dens

    def add_disk(self, mass=1.0e-3, rmin=0.1, rmax=300, plrho=2.37, h0=0.1, \
            plh=58./45., dust=None,  t0=None, plt=None, gas=None, \
            abundance=None, tmid0=None, tatm0=None, zq0=None, pltgas=None, \
            delta=None, gap_rin=[], gap_rout=[], gap_delta=[], \
            gaussian_gaps=False, aturb=None):
        self.disk = Disk(mass=mass, rmin=rmin, rmax=rmax, plrho=plrho, h0=h0, \
                plh=plh, dust=dust, t0=t0, plt=plt, tmid0=tmid0, tatm0=tatm0, \
                zq0=zq0, pltgas=pltgas, delta=delta, gap_rin=gap_rin, \
                gap_rout=gap_rout, gap_delta=gap_delta, aturb=aturb, \
                gaussian_gaps=gaussian_gaps)

        if (dust != None):
            self.grid.add_density(self.disk.density(self.grid.r, \
                    self.grid.theta, self.grid.phi),dust)

        if (gas != None):
            if (type(gas) == list):
                for i in range(len(gas)):
                    self.disk.add_gas(gas[i], abundance[i])
                    self.grid.add_number_density(self.disk.number_density(\
                            self.grid.r, self.grid.theta, self.grid.phi, \
                            gas=i), gas[i])
                    self.grid.add_velocity(self.disk.velocity(self.grid.r, \
                            self.grid.theta, self.grid.phi, \
                            mstar=self.grid.stars[0].mass))
                    if aturb != None:
                        self.grid.add_microturbulence(self.disk.\
                                microturbulence(self.grid.r, self.grid.theta, \
                                self.grid.phi))
            else:
                self.disk.add_gas(gas, abundance)
                self.grid.add_number_density(self.disk.number_density( \
                        self.grid.r, self.grid.theta, self.grid.phi, \
                        gas=0), gas)
                self.grid.add_velocity(self.disk.velocity(self.grid.r, \
                        self.grid.theta, self.grid.phi, \
                        mstar=self.grid.stars[0].mass))
                if aturb != None:
                    self.grid.add_microturbulence(self.disk.microturbulence(\
                            self.grid.r, self.grid.theta, self.grid.phi))

        if t0 != None:
            self.grid.add_temperature(self.disk.temperature(self.grid.r, \
                    self.grid.theta, self.grid.phi))
        if tmid0 != None:
            self.grid.add_gas_temperature(self.disk.gas_temperature( \
                    self.grid.r, self.grid.theta, self.grid.phi))

    def add_dartois_disk(self, mass=1.0e-3, rmin=0.1, rmax=300, plrho=2.37, \
            h0=0.1, plh=58./45., dust=None,  t0=None, plt=None, gas=None, \
            abundance=None, freezout=0., tmid0=None, tatm0=None, zq0=None, \
            pltgas=None, delta=None, gap_rin=[], gap_rout=[], gap_delta=[], \
            gaussian_gaps=False, aturb=None):
        self.disk = DartoisDisk(mass=mass, rmin=rmin, rmax=rmax, plrho=plrho, \
                h0=h0, plh=plh, dust=dust, t0=t0, plt=plt, tmid0=tmid0, \
                tatm0=tatm0, zq0=zq0, pltgas=pltgas, delta=delta, \
                gap_rin=gap_rin, gap_rout=gap_rout, gap_delta=gap_delta, \
                aturb=aturb, gaussian_gaps=gaussian_gaps)

        if (dust != None):
            self.grid.add_density(self.disk.density(self.grid.r, \
                    self.grid.theta, self.grid.phi),dust)

        if (gas != None):
            if (type(gas) == list):
                for i in range(len(gas)):
                    self.disk.add_gas(gas[i], abundance[i], freezeout[i])
                    self.grid.add_number_density(self.disk.number_density(\
                            self.grid.w1, self.grid.w2, self.grid.w3, \
                            gas=i, mstar=self.grid.stars[0].mass), gas[i])
                    self.grid.add_velocity(self.disk.velocity(self.grid.r, \
                            self.grid.theta, self.grid.phi, \
                            mstar=self.grid.stars[0].mass))
                    if aturb != None:
                        self.grid.add_microturbulence(self.disk.\
                                microturbulence(self.grid.r, self.grid.theta, \
                                self.grid.phi))
            else:
                self.disk.add_gas(gas, abundance, freezeout)
                self.grid.add_number_density(self.disk.number_density( \
                        self.grid.w1, self.grid.w2, self.grid.w3, \
                        gas=0, mstar=self.grid.stars[0].mass), gas)
                self.grid.add_velocity(self.disk.velocity(self.grid.r, \
                        self.grid.theta, self.grid.phi, \
                        mstar=self.grid.stars[0].mass))
                if aturb != None:
                    self.grid.add_microturbulence(self.disk.microturbulence(\
                            self.grid.r, self.grid.theta, self.grid.phi))

        if t0 != None:
            self.grid.add_temperature(self.disk.temperature(self.grid.r, \
                    self.grid.theta, self.grid.phi))
        if tmid0 != None:
            self.grid.add_temperature(self.disk.temperature(self.grid.r, \
                    self.grid.theta, self.grid.phi))

    def add_twolayer_disk(self, mass=1.0e-3, rmin=0.1, rmax=300, plrho=2.37, \
            h0=0.1, plh=58./45., dust=None,  t0=None, plt=None, gas=None, \
            abundance=None, tmid0=None, tatm0=None, zq0=None, pltgas=None, \
            delta=None, gap_rin=[], gap_rout=[], gap_delta=[], \
            gaussian_gaps=False, aturb=None, amin=1., amax=1000., fmax=0.8, \
            alpha_settle=1.0e-3):
        self.disk = TwoLayerDisk(mass=mass, rmin=rmin, rmax=rmax, plrho=plrho, \
                h0=h0, plh=plh, dust=dust, t0=t0, plt=plt, tmid0=tmid0, \
                tatm0=tatm0, zq0=zq0, pltgas=pltgas, delta=delta, \
                gap_rin=gap_rin, gap_rout=gap_rout, gap_delta=gap_delta, \
                aturb=aturb, gaussian_gaps=gaussian_gaps, amin=amin, amax=amax,\
                fmax=fmax, alpha_settle=alpha_settle)

        if (dust != None):
            a, rho = self.disk.density(self.grid.r, self.grid.theta, \
                    self.grid.phi)

            for i in range(len(a)):
                self.grid.add_density(rho[:,:,:,i], self.disk.dust(a[i]/1e4, \
                        pla))

        if (gas != None):
            if (type(gas) == list):
                for i in range(len(gas)):
                    self.disk.add_gas(gas[i], abundance[i])
                    self.grid.add_number_density(self.disk.number_density(\
                            self.grid.r, self.grid.theta, self.grid.phi, \
                            gas=i), gas[i])
                    self.grid.add_velocity(self.disk.velocity(self.grid.r, \
                            self.grid.theta, self.grid.phi, \
                            mstar=self.grid.stars[0].mass))
                    if aturb != None:
                        self.grid.add_microturbulence(self.disk.\
                                microturbulence(self.grid.r, self.grid.theta, \
                                self.grid.phi))
            else:
                self.disk.add_gas(gas, abundance)
                self.grid.add_number_density(self.disk.number_density( \
                        self.grid.r, self.grid.theta, self.grid.phi, \
                        gas=0), gas)
                self.grid.add_velocity(self.disk.velocity(self.grid.r, \
                        self.grid.theta, self.grid.phi, \
                        mstar=self.grid.stars[0].mass))
                if aturb != None:
                    self.grid.add_microturbulence(self.disk.microturbulence(\
                            self.grid.r, self.grid.theta, self.grid.phi))

        if t0 != None:
            self.grid.add_temperature(self.disk.temperature(self.grid.r, \
                    self.grid.theta, self.grid.phi))
        if tmid0 != None:
            self.grid.add_gas_temperature(self.disk.gas_temperature( \
                    self.grid.r, self.grid.theta, self.grid.phi))

    def add_settled_disk(self, mass=1.0e-3, rmin=0.1, rmax=300, plrho=2.37, \
            h0=0.1, plh=58./45., dust=None,  t0=None, plt=None, gas=None, \
            abundance=None, tmid0=None, tatm0=None, zq0=None, pltgas=None, \
            delta=None, gap_rin=[], gap_rout=[], gap_delta=[], \
            gaussian_gaps=False, aturb=None, amin=0.05, amax=1000., pla=3.5, \
            alpha_settle=1.0e-3, na=100):
        self.disk = SettledDisk(mass=mass, rmin=rmin, rmax=rmax, plrho=plrho, \
                h0=h0, plh=plh, dust=dust, t0=t0, plt=plt, tmid0=tmid0, \
                tatm0=tatm0, zq0=zq0, pltgas=pltgas, delta=delta, \
                gap_rin=gap_rin, gap_rout=gap_rout, gap_delta=gap_delta, \
                aturb=aturb, gaussian_gaps=gaussian_gaps, amin=amin, amax=amax,\
                pla=pla, alpha_settle=alpha_settle)

        if (dust != None):
            a, rho = self.disk.density(self.grid.r, self.grid.theta, \
                    self.grid.phi, na=na)

            for i in range(len(a)):
                self.grid.add_density(rho[:,:,:,i], self.disk.dust(a[i]/1e4, \
                        pla))

        if (gas != None):
            if (type(gas) == list):
                for i in range(len(gas)):
                    self.disk.add_gas(gas[i], abundance[i])
                    self.grid.add_number_density(self.disk.number_density(\
                            self.grid.r, self.grid.theta, self.grid.phi, \
                            gas=i), gas[i])
                    self.grid.add_velocity(self.disk.velocity(self.grid.r, \
                            self.grid.theta, self.grid.phi, \
                            mstar=self.grid.stars[0].mass))
                    if aturb != None:
                        self.grid.add_microturbulence(self.disk.\
                                microturbulence(self.grid.r, self.grid.theta, \
                                self.grid.phi))
            else:
                self.disk.add_gas(gas, abundance)
                self.grid.add_number_density(self.disk.number_density( \
                        self.grid.r, self.grid.theta, self.grid.phi, \
                        gas=0), gas)
                self.grid.add_velocity(self.disk.velocity(self.grid.r, \
                        self.grid.theta, self.grid.phi, \
                        mstar=self.grid.stars[0].mass))
                if aturb != None:
                    self.grid.add_microturbulence(self.disk.microturbulence(\
                            self.grid.r, self.grid.theta, self.grid.phi))

        if t0 != None:
            self.grid.add_temperature(self.disk.temperature(self.grid.r, \
                    self.grid.theta, self.grid.phi))
        if tmid0 != None:
            self.grid.add_gas_temperature(self.disk.gas_temperature( \
                    self.grid.r, self.grid.theta, self.grid.phi))

    def add_pringle_disk(self, mass=1.0e-3, rmin=0.1, rmax=300, plrho=2.37, \
            h0=0.1, plh=58./45., dust=None,  t0=None, plt=None, gas=None, \
            abundance=None, tmid0=None, tatm0=None, zq0=None, pltgas=None, \
            delta=None, gap_rin=[], gap_rout=[], gap_delta=[], \
            gaussian_gaps=False, aturb=None, gamma_taper=None):
        self.disk = PringleDisk(mass=mass, rmin=rmin, rmax=rmax, plrho=plrho, \
                h0=h0, plh=plh, dust=dust, t0=t0, plt=plt, tmid0=tmid0, \
                tatm0=tatm0, zq0=zq0, pltgas=pltgas, delta=delta, \
                gap_rin=gap_rin, gap_rout=gap_rout, gap_delta=gap_delta, \
                aturb=aturb, gaussian_gaps=gaussian_gaps, \
                gamma_taper=gamma_taper)

        if (dust != None):
            if self.grid.coordsystem == "spherical":
                self.grid.add_density(self.disk.density(self.grid.r, \
                        self.grid.theta, self.grid.phi),dust)
            elif self.grid.coordsystem == "cartesian":
                self.grid.add_density(self.disk.density(self.grid.x, \
                        self.grid.y, self.grid.z, \
                        coordsys=self.grid.coordsystem),dust)
            elif self.grid.coordsystem == "cylindrical":
                self.grid.add_density(self.disk.density(self.grid.rho, \
                        self.grid.phi, self.grid.z, \
                        coordsys=self.grid.coordsystem),dust)

        if (gas != None):
            if (type(gas) == list):
                for i in range(len(gas)):
                    self.disk.add_gas(gas[i], abundance[i])
                    self.grid.add_number_density(self.disk.number_density( \
                            self.grid.r, self.grid.theta, self.grid.phi, \
                            gas=i), gas[i])
                    self.grid.add_velocity(self.disk.velocity(self.grid.r, \
                            self.grid.theta, self.grid.phi, \
                            mstar=self.grid.stars[0].mass))
                    if aturb != None:
                        self.grid.add_microturbulence(self.disk.\
                                microturbulence(self.grid.r, self.grid.theta, \
                                self.grid.phi))
            else:
                self.disk.add_gas(gas, abundance)
                self.grid.add_number_density(self.disk.number_density( \
                        self.grid.r, self.grid.theta, self.grid.phi, \
                        gas=0), gas)
                self.grid.add_velocity(self.disk.velocity(self.grid.r, \
                        self.grid.theta, self.grid.phi, \
                        mstar=self.grid.stars[0].mass))
                if aturb != None:
                    self.grid.add_microturbulence(self.disk.microturbulence( \
                            self.grid.r, self.grid.theta, self.grid.phi))

        if t0 != None:
            self.grid.add_temperature(self.disk.temperature(self.grid.r, \
                    self.grid.theta, self.grid.phi))
        if tmid0 != None:
            self.grid.add_gas_temperature(self.disk.gas_temperature( \
                    self.grid.r, self.grid.theta, self.grid.phi))

    def add_dartois_pringle_disk(self, mass=1.0e-3, rmin=0.1, rmax=300, \
            plrho=2.37, h0=0.1, plh=58./45., dust=None,  t0=None, plt=None, \
            gas=None, abundance=None, freezeout=0., tmid0=None, tatm0=None, \
            zq0=None, pltgas=None, delta=None, gap_rin=[], gap_rout=[], \
            gap_delta=[], gaussian_gaps=False, aturb=None, gamma_taper=None):
        self.disk = DartoisPringleDisk(mass=mass, rmin=rmin, rmax=rmax, \
                plrho=plrho, h0=h0, plh=plh, dust=dust, t0=t0, plt=plt, \
                tmid0=tmid0, tatm0=tatm0, zq0=zq0, pltgas=pltgas, delta=delta, \
                gap_rin=gap_rin, gap_rout=gap_rout, gap_delta=gap_delta, \
                aturb=aturb, gaussian_gaps=gaussian_gaps, \
                gamma_taper=gamma_taper)

        if (dust != None):
            if self.grid.coordsystem == "spherical":
                self.grid.add_density(self.disk.density(self.grid.r, \
                        self.grid.theta, self.grid.phi),dust)
            elif self.grid.coordsystem == "cartesian":
                self.grid.add_density(self.disk.density(self.grid.x, \
                        self.grid.y, self.grid.z, \
                        coordsys=self.grid.coordsystem),dust)
            elif self.grid.coordsystem == "cylindrical":
                self.grid.add_density(self.disk.density(self.grid.rho, \
                        self.grid.phi, self.grid.z, \
                        coordsys=self.grid.coordsystem),dust)

        if (gas != None):
            if (type(gas) == list):
                for i in range(len(gas)):
                    self.disk.add_gas(gas[i], abundance[i], freezeout[i])
                    self.grid.add_number_density(self.disk.number_density( \
                            self.grid.w1, self.grid.w2, self.grid.w3, \
                            gas=i, mstar=self.grid.stars[0].mass), gas[i])
                    self.grid.add_velocity(self.disk.velocity(self.grid.r, \
                            self.grid.theta, self.grid.phi, \
                            mstar=self.grid.stars[0].mass))
                    if aturb != None:
                        self.grid.add_microturbulence(self.disk.\
                                microturbulence(self.grid.r, self.grid.theta, \
                                self.grid.phi))
            else:
                self.disk.add_gas(gas, abundance, freezeout)
                self.grid.add_number_density(self.disk.number_density( \
                        self.grid.w1, self.grid.w2, self.grid.w3, \
                        gas=0, mstar=self.grid.stars[0].mass), gas)
                self.grid.add_velocity(self.disk.velocity(self.grid.r, \
                        self.grid.theta, self.grid.phi, \
                        mstar=self.grid.stars[0].mass))
                if aturb != None:
                    self.grid.add_microturbulence(self.disk.microturbulence( \
                            self.grid.r, self.grid.theta, self.grid.phi))

        if t0 != None:
            self.grid.add_temperature(self.disk.temperature(self.grid.r, \
                    self.grid.theta, self.grid.phi))
        if tmid0 != None:
            self.grid.add_temperature(self.disk.temperature(self.grid.r, \
                    self.grid.theta, self.grid.phi))

    def add_twolayer_pringle_disk(self, mass=1.0e-3, rmin=0.1, rmax=300, \
            plrho=2.37, h0=0.1, plh=58./45., dust=None,  t0=None, plt=None, \
            gas=None, abundance=None, tmid0=None, tatm0=None, zq0=None, \
            pltgas=None, delta=None, gap_rin=[], gap_rout=[], gap_delta=[], \
            gaussian_gaps=False, aturb=None, amin=1., amax=1000., fmax=0.8, \
            alpha_settle=1.0e-3):
        self.disk = TwoLayerPringleDisk(mass=mass, rmin=rmin, rmax=rmax, \
                plrho=plrho, h0=h0, plh=plh, dust=dust, t0=t0, plt=plt, \
                tmid0=tmid0, tatm0=tatm0, zq0=zq0, pltgas=pltgas, delta=delta, \
                gap_rin=gap_rin, gap_rout=gap_rout, gap_delta=gap_delta, \
                aturb=aturb, gaussian_gaps=gaussian_gaps, amin=amin, amax=amax,\
                fmax=fmax, alpha_settle=alpha_settle)

        if (dust != None):
            a, rho = self.disk.density(self.grid.r, self.grid.theta, \
                    self.grid.phi)

            for i in range(len(a)):
                self.grid.add_density(rho[:,:,:,i], self.disk.dust(a[i]/1e4, \
                        pla))

        if (gas != None):
            if (type(gas) == list):
                for i in range(len(gas)):
                    self.disk.add_gas(gas[i], abundance[i])
                    self.grid.add_number_density(self.disk.number_density(\
                            self.grid.r, self.grid.theta, self.grid.phi, \
                            gas=i), gas[i])
                    self.grid.add_velocity(self.disk.velocity(self.grid.r, \
                            self.grid.theta, self.grid.phi, \
                            mstar=self.grid.stars[0].mass))
                    if aturb != None:
                        self.grid.add_microturbulence(self.disk.\
                                microturbulence(self.grid.r, self.grid.theta, \
                                self.grid.phi))
            else:
                self.disk.add_gas(gas, abundance)
                self.grid.add_number_density(self.disk.number_density( \
                        self.grid.r, self.grid.theta, self.grid.phi, \
                        gas=0), gas)
                self.grid.add_velocity(self.disk.velocity(self.grid.r, \
                        self.grid.theta, self.grid.phi, \
                        mstar=self.grid.stars[0].mass))
                if aturb != None:
                    self.grid.add_microturbulence(self.disk.microturbulence(\
                            self.grid.r, self.grid.theta, self.grid.phi))

        if t0 != None:
            self.grid.add_temperature(self.disk.temperature(self.grid.r, \
                    self.grid.theta, self.grid.phi))
        if tmid0 != None:
            self.grid.add_gas_temperature(self.disk.gas_temperature( \
                    self.grid.r, self.grid.theta, self.grid.phi))

    def add_settled_pringle_disk(self, mass=1.0e-3, rmin=0.1, rmax=300, \
            plrho=2.37, h0=0.1, plh=58./45., dust=None,  t0=None, plt=None, \
            gas=None, abundance=None, tmid0=None, tatm0=None, zq0=None, \
            pltgas=None, delta=None, gap_rin=[], gap_rout=[], gap_delta=[], \
            gaussian_gaps=False, aturb=None, amin=0.05, amax=1000., pla=3.5, \
            alpha_settle=1.0e-3, na=100, gamma_taper=None):
        self.disk = SettledPringleDisk(mass=mass, rmin=rmin, rmax=rmax, \
                plrho=plrho, h0=h0, plh=plh, dust=dust, t0=t0, plt=plt, \
                tmid0=tmid0, tatm0=tatm0, zq0=zq0, pltgas=pltgas, delta=delta, \
                gap_rin=gap_rin, gap_rout=gap_rout, gap_delta=gap_delta, \
                aturb=aturb, gaussian_gaps=gaussian_gaps, amin=amin, amax=amax,\
                pla=pla, alpha_settle=alpha_settle, gamma_taper=None)

        if (dust != None):
            a, rho = self.disk.density(self.grid.r, self.grid.theta, \
                    self.grid.phi, na=na)

            for i in range(len(a)):
                self.grid.add_density(rho[:,:,:,i], self.disk.dust(a[i]/1e4, \
                        pla))

        if (gas != None):
            if (type(gas) == list):
                for i in range(len(gas)):
                    self.disk.add_gas(gas[i], abundance[i])
                    self.grid.add_number_density(self.disk.number_density(\
                            self.grid.r, self.grid.theta, self.grid.phi, \
                            gas=i), gas[i])
                    self.grid.add_velocity(self.disk.velocity(self.grid.r, \
                            self.grid.theta, self.grid.phi, \
                            mstar=self.grid.stars[0].mass))
                    if aturb != None:
                        self.grid.add_microturbulence(self.disk.\
                                microturbulence(self.grid.r, self.grid.theta, \
                                self.grid.phi))
            else:
                self.disk.add_gas(gas, abundance)
                self.grid.add_number_density(self.disk.number_density( \
                        self.grid.r, self.grid.theta, self.grid.phi, \
                        gas=0), gas)
                self.grid.add_velocity(self.disk.velocity(self.grid.r, \
                        self.grid.theta, self.grid.phi, \
                        mstar=self.grid.stars[0].mass))
                if aturb != None:
                    self.grid.add_microturbulence(self.disk.microturbulence( \
                            self.grid.r, self.grid.theta, self.grid.phi))

        if t0 != None:
            self.grid.add_temperature(self.disk.temperature(self.grid.r, \
                    self.grid.theta, self.grid.phi))
        if tmid0 != None:
            self.grid.add_gas_temperature(self.disk.gas_temperature( \
                    self.grid.r, self.grid.theta, self.grid.phi))

    def add_envelope(self, mass=1.0e-3, rmin=0.1, rmax=1000, pl=1.5, \
            cavpl=1.0, cavrfact=0.2, t0=None, tpl=None, dust=None, gas=None, \
            abundance=None, tmid0=None, aturb=None):
        self.envelope = Envelope(mass=mass, rmin=rmin, rmax=rmax, \
                pl=pl, cavpl=cavpl, cavrfact=cavrfact, t0=t0, tpl=tpl, \
                dust=dust)

        if (dust != None):
            self.grid.add_density(self.envelope.density(self.grid.r, \
                    self.grid.theta, self.grid.phi),dust)

        if (gas != None):
            if (type(gas) == list):
                for i in range(len(gas)):
                    self.envelope.add_gas(gas[i], abundance[i])
                    self.grid.add_number_density(self.envelope.number_density( \
                            self.grid.r, self.grid.theta, self.grid.phi, \
                            gas=i), gas[i])
                    self.grid.add_velocity(self.envelope.velocity(self.grid.r, \
                            self.grid.theta, self.grid.phi, \
                            mstar=self.grid.stars[0].mass))
                    if aturb != None:
                        self.grid.add_microturbulence(self.disk.\
                                microturbulence(self.grid.r, self.grid.theta, \
                                self.grid.phi))
            else:
                self.envelope.add_gas(gas, abundance)
                self.grid.add_number_density(self.envelope.number_density( \
                        self.grid.r, self.grid.theta, self.grid.phi, gas=0), \
                        gas)
                self.grid.add_velocity(self.envelope.velocity(self.grid.r, \
                        self.grid.theta, self.grid.phi, \
                        mstar=self.grid.stars[0].mass))

                if aturb != None:
                    self.grid.add_microturbulence(self.disk.microturbulence( \
                            self.grid.r, self.grid.theta, self.grid.phi))
        if t0 != None:
            self.grid.add_temperature(self.envelope.temperature(self.grid.r, \
                    self.grid.theta, self.grid.phi))
        if tmid0 != None:
            self.grid.add_gas_temperature(self.envelope.gas_temperature( \
                    self.grid.r, self.grid.theta, self.grid.phi))

    def add_ulrich_envelope(self, mass=1.0e-3, rmin=0.1, rmax=1000, rcent=300, \
            cavpl=1.0, cavrfact=0.2, t0=None, tpl=None, dust=None, gas=None, \
            abundance=None, tmid0=None, rcent_ne_rdisk=False, aturb=None):
        if rcent_ne_rdisk:
            self.envelope = UlrichEnvelope(mass=mass, rmin=rmin, rmax=rmax, \
                    rcent=rcent, cavpl=cavpl, cavrfact=cavrfact, t0=t0, \
                    tpl=tpl, dust=dust, aturb=aturb)
        elif hasattr(self, 'disk'):
            self.envelope = UlrichEnvelope(mass=mass, rmin=rmin, rmax=rmax, \
                    rcent=self.disk.rmax, cavpl=cavpl, cavrfact=cavrfact, \
                    t0=t0, tpl=tpl, dust=dust, aturb=aturb)
        else:
            self.envelope = UlrichEnvelope(mass=mass, rmin=rmin, rmax=rmax, \
                    rcent=rcent, cavpl=cavpl, cavrfact=cavrfact, t0=t0, \
                    tpl=tpl, dust=dust, aturb=aturb)

        if (dust != None):
            self.grid.add_density(self.envelope.density(self.grid.r, \
                    self.grid.theta, self.grid.phi),dust)

        if (gas != None):
            if (type(gas) == list):
                for i in range(len(gas)):
                    self.envelope.add_gas(gas[i], abundance[i])
                    self.grid.add_number_density(self.envelope.number_density( \
                            self.grid.r, self.grid.theta, self.grid.phi, \
                            gas=i), gas[i])
                    self.grid.add_velocity(self.envelope.velocity(self.grid.r, \
                            self.grid.theta, self.grid.phi, \
                            mstar=self.grid.stars[0].mass))
                    if aturb != None:
                        self.grid.add_microturbulence(self.envelope.\
                                microturbulence(self.grid.r, self.grid.theta, \
                                self.grid.phi))
            else:
                self.envelope.add_gas(gas, abundance)
                self.grid.add_number_density(self.envelope.number_density( \
                        self.grid.r, self.grid.theta, self.grid.phi, gas=0), \
                        gas)
                self.grid.add_velocity(self.envelope.velocity(self.grid.r, \
                        self.grid.theta, self.grid.phi, \
                        mstar=self.grid.stars[0].mass))
                if aturb != None:
                    self.grid.add_microturbulence(self.envelope.\
                            microturbulence(self.grid.r, self.grid.theta, \
                            self.grid.phi))

        if t0 != None:
            self.grid.add_temperature(self.envelope.temperature(self.grid.r, \
                    self.grid.theta, self.grid.phi))
        if tmid0 != None:
            self.grid.add_gas_temperature(self.envelope.gas_temperature( \
                    self.grid.r, self.grid.theta, self.grid.phi))

    def add_ulrichextended_envelope(self, mass=1.0e-3, rmin=0.1, rmax=1000, \
            rcent=300, cavpl=1.0, cavrfact=0.2, theta_open=45., zoffset=1., \
            t0=None, tpl=None, dust=None, gas=None, abundance=None, tmid0=None,\
            rcent_ne_rdisk=False, aturb=None):
        if rcent_ne_rdisk:
            self.envelope = UlrichEnvelopeExtended(mass=mass, rmin=rmin, \
                    rmax=rmax, rcent=rcent, cavpl=cavpl, cavrfact=cavrfact, \
                    theta_open=theta_open, zoffset=zoffset, t0=t0, tpl=tpl, \
                    dust=dust, aturb=aturb)
        elif hasattr(self, 'disk'):
            self.envelope = UlrichEnvelopeExtended(mass=mass, rmin=rmin, \
                    rmax=rmax, rcent=self.disk.rmax, cavpl=cavpl, \
                    cavrfact=cavrfact, theta_open=theta_open, zoffset=zoffset, \
                    t0=t0, tpl=tpl, dust=dust, aturb=aturb)
        else:
            self.envelope = UlrichEnvelopeExtended(mass=mass, rmin=rmin, \
                    rmax=rmax, rcent=rcent, cavpl=cavpl, cavrfact=cavrfact, \
                    theta_open=theta_open, zoffset=zoffset, t0=t0, tpl=tpl, \
                    dust=dust, aturb=aturb)

        if (dust != None):
            self.grid.add_density(self.envelope.density(self.grid.r, \
                    self.grid.theta, self.grid.phi),dust)

        if (gas != None):
            if (type(gas) == list):
                for i in range(len(gas)):
                    self.envelope.add_gas(gas[i], abundance[i])
                    self.grid.add_number_density(self.envelope.number_density( \
                            self.grid.r, self.grid.theta, self.grid.phi, \
                            gas=i), gas[i])
                    self.grid.add_velocity(self.envelope.velocity(self.grid.r, \
                            self.grid.theta, self.grid.phi, \
                            mstar=self.grid.stars[0].mass))
                    if aturb != None:
                        self.grid.add_microturbulence(self.envelope.\
                                microturbulence(self.grid.r, self.grid.theta, \
                                self.grid.phi))
            else:
                self.envelope.add_gas(gas, abundance)
                self.grid.add_number_density(self.envelope.number_density( \
                        self.grid.r, self.grid.theta, self.grid.phi, gas=0), \
                        gas)
                self.grid.add_velocity(self.envelope.velocity(self.grid.r, \
                        self.grid.theta, self.grid.phi, \
                        mstar=self.grid.stars[0].mass))
                if aturb != None:
                    self.grid.add_microturbulence(self.envelope.\
                            microturbulence(self.grid.r, self.grid.theta, \
                            self.grid.phi))

        if t0 != None:
            self.grid.add_temperature(self.envelope.temperature(self.grid.r, \
                    self.grid.theta, self.grid.phi))
        if tmid0 != None:
            self.grid.add_gas_temperature(self.envelope.gas_temperature( \
                    self.grid.r, self.grid.theta, self.grid.phi))

    def add_tapered_ulrich_envelope(self, mass=1.0e-3, rmin=0.1, rmax=1000, \
            rcent=300, cavpl=1.0, cavrfact=0.2, gamma=1., t0=None, tpl=None, \
            dust=None, gas=None, abundance=None, tmid0=None, \
            rcent_ne_rdisk=False, aturb=None):
        if rcent_ne_rdisk:
            self.envelope = TaperedUlrichEnvelope(mass=mass, rmin=rmin, \
                    rmax=rmax, rcent=rcent, gamma=gamma, cavpl=cavpl, \
                    cavrfact=cavrfact, t0=t0, tpl=tpl, dust=dust, aturb=aturb)
        elif hasattr(self, 'disk'):
            self.envelope = TaperedUlrichEnvelope(mass=mass, rmin=rmin, \
                    rmax=rmax, rcent=self.disk.rmax, gamma=gamma, cavpl=cavpl, \
                    cavrfact=cavrfact, t0=t0, tpl=tpl, dust=dust, aturb=aturb)
        else:
            self.envelope = TaperedUlrichEnvelope(mass=mass, rmin=rmin, \
                    rmax=rmax, rcent=rcent, gamma=gamma, cavpl=cavpl, \
                    cavrfact=cavrfact, t0=t0, tpl=tpl, dust=dust, aturb=aturb)

        if (dust != None):
            self.grid.add_density(self.envelope.density(self.grid.r, \
                    self.grid.theta, self.grid.phi),dust)

        if (gas != None):
            if (type(gas) == list):
                for i in range(len(gas)):
                    self.envelope.add_gas(gas[i], abundance[i])
                    self.grid.add_number_density(self.envelope.number_density( \
                            self.grid.r, self.grid.theta, self.grid.phi, \
                            gas=i), gas[i])
                    self.grid.add_velocity(self.envelope.velocity(self.grid.r, \
                            self.grid.theta, self.grid.phi, \
                            mstar=self.grid.stars[0].mass))
                    if aturb != None:
                        self.grid.add_microturbulence(self.envelope.\
                                microturbulence(self.grid.r, self.grid.theta, \
                                self.grid.phi))
            else:
                self.envelope.add_gas(gas, abundance)
                self.grid.add_number_density(self.envelope.number_density( \
                        self.grid.r, self.grid.theta, self.grid.phi, gas=0), \
                        gas)
                self.grid.add_velocity(self.envelope.velocity(self.grid.r, \
                        self.grid.theta, self.grid.phi, \
                        mstar=self.grid.stars[0].mass))
                if aturb != None:
                    self.grid.add_microturbulence(self.envelope.\
                            microturbulence(self.grid.r, self.grid.theta, \
                            self.grid.phi))

        if t0 != None:
            self.grid.add_temperature(self.envelope.temperature(self.grid.r, \
                    self.grid.theta, self.grid.phi))
        if tmid0 != None:
            self.grid.add_gas_temperature(self.envelope.gas_temperature( \
                    self.grid.r, self.grid.theta, self.grid.phi))

    def add_tapered_ulrichextended_envelope(self, mass=1.0e-3, rmin=0.1, \
            rmax=1000, rcent=300, cavpl=1.0, cavrfact=0.2, gamma=1., \
            theta_open=45., zoffset=1., t0=None, tpl=None, dust=None, gas=None,\
            abundance=None, tmid0=None, rcent_ne_rdisk=False, aturb=None):
        if rcent_ne_rdisk:
            self.envelope = TaperedUlrichEnvelopeExtended(mass=mass, rmin=rmin,\
                    rmax=rmax, rcent=rcent, gamma=gamma, cavpl=cavpl, \
                    cavrfact=cavrfact, theta_open=theta_open, zoffset=zoffset, \
                    t0=t0, tpl=tpl, dust=dust, aturb=aturb)
        elif hasattr(self, 'disk'):
            self.envelope = TaperedUlrichEnvelopeExtended(mass=mass, rmin=rmin,\
                    rmax=rmax, rcent=self.disk.rmax, gamma=gamma, cavpl=cavpl, \
                    cavrfact=cavrfact, theta_open=theta_open, zoffset=zoffset, \
                    t0=t0, tpl=tpl, dust=dust, aturb=aturb)
        else:
            self.envelope = TaperedUlrichEnvelopeExtended(mass=mass, rmin=rmin,\
                    rmax=rmax, rcent=rcent, gamma=gamma, cavpl=cavpl, \
                    cavrfact=cavrfact, theta_open=theta_open, zoffset=zoffset, \
                    t0=t0, tpl=tpl, dust=dust, aturb=aturb)

        if (dust != None):
            self.grid.add_density(self.envelope.density(self.grid.r, \
                    self.grid.theta, self.grid.phi),dust)

        if (gas != None):
            if (type(gas) == list):
                for i in range(len(gas)):
                    self.envelope.add_gas(gas[i], abundance[i])
                    self.grid.add_number_density(self.envelope.number_density( \
                            self.grid.r, self.grid.theta, self.grid.phi, \
                            gas=i), gas[i])
                    self.grid.add_velocity(self.envelope.velocity(self.grid.r, \
                            self.grid.theta, self.grid.phi, \
                            mstar=self.grid.stars[0].mass))
                    if aturb != None:
                        self.grid.add_microturbulence(self.envelope.\
                                microturbulence(self.grid.r, self.grid.theta, \
                                self.grid.phi))
            else:
                self.envelope.add_gas(gas, abundance)
                self.grid.add_number_density(self.envelope.number_density( \
                        self.grid.r, self.grid.theta, self.grid.phi, gas=0), \
                        gas)
                self.grid.add_velocity(self.envelope.velocity(self.grid.r, \
                        self.grid.theta, self.grid.phi, \
                        mstar=self.grid.stars[0].mass))
                if aturb != None:
                    self.grid.add_microturbulence(self.envelope.\
                            microturbulence(self.grid.r, self.grid.theta, \
                            self.grid.phi))

        if t0 != None:
            self.grid.add_temperature(self.envelope.temperature(self.grid.r, \
                    self.grid.theta, self.grid.phi))
        if tmid0 != None:
            self.grid.add_gas_temperature(self.envelope.gas_temperature( \
                    self.grid.r, self.grid.theta, self.grid.phi))

    def run_simple_dust_image(self, name=None, i=0., pa=0., npix=256, dx=1., \
            nu=230., kappa0=0.1, beta0=1, delta_beta=1, r0beta=100., plbeta=1, \
            dpc=140):
        # Convert inclination and position angle to radians.

        i *= numpy.pi / 180.
        pa *= numpy.pi / 180.

        # Set up the image plane.

        if npix%2 == 0:
            xx = numpy.linspace(-npix/2*dx*dpc, (npix/2-1)*dx*dpc, npix)
            yy = numpy.linspace(-npix/2*dx*dpc, (npix/2-1)*dx*dpc, npix)
        else:
            xx = numpy.linspace(-(npix-1)/2*dx*dpc, (npix-1)/2*dx*dpc, npix)
            yy = numpy.linspace(-(npix-1)/2*dx*dpc, (npix-1)/2*dx*dpc, npix)

        x, y = numpy.meshgrid(xx, yy)

        # Calculate physical coordinates from image plane coordinates.

        xpp = x*numpy.cos(pa) - y*numpy.sin(pa)
        ypp = (x*numpy.sin(pa) + y*numpy.cos(pa))/numpy.cos(i)
        rpp = numpy.sqrt(xpp**2 + ypp**2)
        phipp = numpy.arctan2(ypp, xpp) + pi/2

        # Calculate physical quantities.

        Sigma = self.disk.surface_density(rpp)

        T = self.disk.temperature_1d(rpp)

        B = B_nu(nu*1e9, T)

        beta = beta0 + delta_beta * (rpp / r0beta)**plbeta

        kappa = kappa0 * (nu / 1000.)**beta

        # Now do the actual calculation.

        I = B * (1. - numpy.exp(-kappa * Sigma / numpy.cos(i)))

        # Adjust the scale of I to be in the appropriate units.

        I = I / Jy * ((xx[1] - xx[0]) * AU / (dpc * pc)) * \
                ((yy[1] - yy[0]) * AU / (dpc * pc))

        self.images[name] =  Image(I.reshape((npix,npix,1,1)), x=xx/dpc, \
                y=yy/dpc, freq=numpy.array([nu])*1.0e9)

    def run_simple_dust_visibilities(self, name=None, i=0., pa=0., npix=256, \
            dx=1., nu=230., kappa0=0.1, beta0=1, delta_beta=1, r0beta=100., \
            plbeta=1, dpc=140):

        self.run_simple_dust_image(name="temp", i=i, pa=pa, npix=npix, \
                dx=dx, nu=nu, kappa0=kappa0, beta0=beta0, r0beta=r0beta, \
                delta_beta=delta_beta, plbeta=plbeta, dpc=dpc)

        self.visibilities[name] = imtovis(self.images["temp"])

        self.images.pop("temp")
    
    def run_simple_gas_image(self, i=0., pa=0., npix=256, dx=1., species=0, \
            trans=0, vstart=-10, dv=0.5, nv=40, n=0.5, T0=10000., plT=1, \
            v_z=0.):
        # Get a few constants from the molecule.

        A = self.disk.gas[species].A_ul[trans]
        nu0 = self.disk.gas[species].nu[trans]
        m_mol = self.disk.gas[species].mass * m_p

        # Set up the image plane.

        xx = numpy.linspace(-(npix-1)/2*dx, (npix-1)/2*dx, 256)
        yy = numpy.linspace(-(npix-1)/2*dx, (npix-1)/2*dx, 256)
        v = numpy.linspace(vstart*kms, (vstart+(nv-1)*dv)*kms, nv)
        nn = nu0 * (1 - v/c)

        x, y, nu = numpy.meshgrid(xx, yy, nn)

        # Calculate physical coordinates from image plane coordinates.

        xpp = -x*numpy.cos(pa) - y*numpy.sin(pa)
        ypp = (-x*numpy.sin(pa) + y*numpy.cos(pa))/numpy.cos(i)
        rpp = numpy.sqrt(xpp**2 + ypp**2)
        phipp = numpy.arctan2(ypp, xpp) + pi/2

        # Calculate physical quantities.

        Sigma = self.disk.surface_density(rpp)

        T = self.disk.temperature_1d(rpp, T_0=T0, p=plT)

        a_tot = numpy.sqrt(2*k*T/m_mol)
        print(a_tot.min(), a_tot.max())

        phi_dot_n = numpy.sin(phipp)*numpy.sin(i)*numpy.cos(pa) - \
                numpy.cos(phipp)*numpy.sin(i)*numpy.sin(pa)

        v_dot_n = numpy.sqrt(G*self.grid.stars[0].mass*M_sun/(rpp*AU)) * \
                phi_dot_n + v_z
        v_dot_n[(rpp >= self.disk.rmax) ^ (rpp <= self.disk.rmin)] = 0.0

        phi = numpy.zeros(rpp.shape)
        phi[a_tot != 0] = c / (a_tot[a_tot != 0]*nu0*numpy.sqrt(pi)) * \
                numpy.exp(-c**2*(nu[a_tot != 0]*(1 + v_dot_n[a_tot != 0]/c) - \
                nu0)**2 / (a_tot[a_tot != 0]**2 * nu0**2))

        # Now do the actual calculation.

        I = h*nu/(4*pi)*A*n*Sigma*phi

        return I, v
    
    def make_hyperion_symmetric(self):
        for i in range(len(self.grid.temperature)):
            ntheta = len(self.grid.theta)
            upper = self.grid.temperature[i][:,0:int(ntheta/2),:]
            lower = self.grid.temperature[i][:,int(ntheta/2):,:][:,::-1,:]
            average = 0.5 * (upper + lower)

            self.grid.temperature[i][:,0:int(ntheta/2),:] = average
            self.grid.temperature[i][:,int(ntheta/2):,:] =  average[:,::-1,:]

    def convert_hyperion_to_radmc3d(self):
        self.grid.r = self.grid.r[1:]
        self.grid.w1 = self.grid.w1[1:]

        ntheta = len(self.grid.theta)
        self.grid.theta = self.grid.theta[0:int(ntheta/2)]
        self.grid.w2 = self.grid.w2[0:int(ntheta/2+1)]
        
        for i in range(len(self.grid.density)):
            self.grid.density[i] = self.grid.density[i][1:,0:int(ntheta/2),:]
        for i in range(len(self.grid.temperature)):
            self.grid.temperature[i] = self.grid.temperature[i][1:,0:int(ntheta/2),:]
        for i in range(len(self.grid.number_density)):
            self.grid.number_density[i] = self.grid.number_density[i][1:,0:int(ntheta/2),:]
        for i in range(len(self.grid.microturbulence)):
            self.grid.microturbulence[i] = self.grid.microturbulence[i][1:,0:int(ntheta/2),:]
        for i in range(len(self.grid.velocity)):
            self.grid.velocity[i] = self.grid.velocity[i][:,1:,0:int(ntheta/2),:]

    def read_yso(self, filename):
        f = h5py.File(filename, "r")

        if ('Disk' in f):
            self.disk = Disk()
            self.disk.read(usefile=f['Disk'])

        if ('Envelope' in f):
            self.envelope = UlrichEnvelope()
            self.envelope.read(usefile=f['Envelope'])

        self.read(usefile=f)

        f.close()

    def write_yso(self, filename):
        f = h5py.File(filename, "w")

        self.write(usefile=f)

        if hasattr(self, "disk"):
            disk = f.create_group("Disk")
            self.disk.write(usefile=disk)

        if hasattr(self, "envelope"):
            envelope = f.create_group("Envelope")
            self.envelope.write(usefile=envelope)

        f.close()
