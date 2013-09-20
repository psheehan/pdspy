import numpy
from .Model import Model
from .Disk import Disk
from .UlrichEnvelope import UlrichEnvelope
from .Star import Star

class YSOModel(Model):

    def add_star(self, mstar=0.5, lstar=1, tstar=4000.):
        self.grid.add_star(Star(mstar, lstar, tstar))

    def set_spherical_grid(self, rmin, rmax, nr, ntheta, nphi, log=True):
        if log:
            r = numpy.logspace(numpy.log10(rmin), numpy.log10(rmax), nr)
        else:
            r = numpy.linspace(rmin, rmax, nr)

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
