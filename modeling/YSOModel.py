import numpy
import h5py
from .Model import Model
from .Disk import Disk
from .UlrichEnvelope import UlrichEnvelope
from .Star import Star
from ..constants.physics import h, c, G, m_p, k
from ..constants.astronomy import AU, M_sun
from ..constants.math import pi

class YSOModel(Model):

    def add_star(self, mass=0.5, luminosity=1, temperature=4000.):
        self.grid.add_star(Star(mass=mass, luminosity=luminosity, \
                temperature=temperature))

    def set_cartesian_grid(self, xmin, xmax, nx):
        x = numpy.linspace(xmin, xmax, nx)
        y = numpy.linspace(xmin, xmax, nx)
        z = numpy.linspace(xmin, xmax, nx)

        self.grid.set_cartesian_grid(x, y, z)

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

    def add_disk(self, mass=1.0e-3, rmin=0.1, rmax=300, plrho=2.37, h0=0.1, \
            plh=58./45., dust="dustkappa_yso.inp"):
        self.disk = Disk(mass=mass, rmin=rmin, rmax=rmax, plrho=plrho, h0=h0, \
                plh=plh,dust=dust)

        self.grid.add_density(self.disk.density(self.grid.r, self.grid.theta, \
                self.grid.phi),dust)

    def add_ulrich_envelope(self, mass=1.0e-3, rmin=0.1, rmax=1000, cavpl=1.0, \
            cavrfact=0.2, dust="dustkappa_yso.inp"):
        self.envelope = UlrichEnvelope(mass=mass, rmin=rmin, rmax=rmax, \
                rcent=self.disk.rmax, cavpl=cavpl, cavrfact=cavrfact, dust=dust)

        self.grid.add_density(self.envelope.density(self.grid.r, \
                self.grid.theta, self.grid.phi),dust)

    def run_simple_gas_image(self, i=0., pa=0., nu0=230.0e11, A=1.0e-3, n=0.5):
        # Define a few convenience functions.

        def Sigma(x, y, i, pa):
            xpp = -x*numpy.cos(pa) - y*numpy.sin(pa)
            ypp = (-x*numpy.sin(pa) + y*numpy.cos(pa))/numpy.cos(i)

            rpp = numpy.sqrt(xpp**2 + ypp**2)

            return self.disk.surface_density(rpp)

        def T(x, y, i, pa, T0=100., p=1):
            xpp = -x*numpy.cos(pa) - y*numpy.sin(pa)
            ypp = (-x*numpy.sin(pa) + y*numpy.cos(pa))/numpy.cos(i)

            rpp = numpy.sqrt(xpp**2 + ypp**2)

            return self.disk.temperature(rpp, T_0=T0, p=p)

        def a_tot(x, y, i, pa, m_mol=m_p, T0=1000., p=1):
            return numpy.sqrt(2*k*T(x, y, i, pa, T0=T0, p=p)/m_mol)

        def v_dot_n(x, y, i, pa, v_z=0.):
            xpp = -x*numpy.cos(pa) - y*numpy.sin(pa)
            ypp = (-x*numpy.sin(pa) + y*numpy.cos(pa))/numpy.cos(i)

            rpp = numpy.sqrt(xpp**2 + ypp**2)
            phipp = numpy.arctan2(ypp, xpp) + pi/2

            phi_dot_n = numpy.sin(phipp)*numpy.sin(i)*numpy.cos(pa) - \
                    numpy.cos(phipp)*numpy.sin(i)*numpy.sin(pa)

            v = numpy.sqrt(G*self.grid.stars[0].mass*M_sun/(rpp*AU))*phi_dot_n + v_z

            v[(rpp >= self.disk.rmax) ^ (rpp <= self.disk.rmin)] = 0.0

            return v

        def phi_tilde(x, y, i, pa, nu, nu_ij=230.e11):
            return c / (a_tot(x,y,i,pa)*nu_ij*numpy.sqrt(pi)) * \
                    numpy.exp(-c**2 * (nu - nu_ij)**2 / \
                    (a_tot(x,y,i,pa)**2 * nu_ij**2))

        def phi(x, y, i, pa, nu, nu_ij=230.e11):
            return phi_tilde(x, y, i, pa, nu*(1 + \
                    v_dot_n(x,y,i,pa,v_z=-2.5e5)/c))

        # Now do the actual calculation.

        x = numpy.linspace(-100, 100, 256)
        y = numpy.linspace(-100, 100, 256)

        v = numpy.linspace(-10e5, 9.5e5, 40)
        nu = nu0 * (1 - v/c)

        xx, yy, nn = numpy.meshgrid(x, y, nu)

        I = h*nu/(4*pi)*A*n*Sigma(xx,yy,i,pa)*phi(xx,yy,i,pa,nn)

        return I, v

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

        disk = f.create_group("Disk")
        if hasattr(self, "disk"):
            self.disk.write(usefile=disk)

        envelope = f.create_group("Envelope")
        if hasattr(self, "envelope"):
            self.envelope.write(usefile=envelope)

        f.close()
