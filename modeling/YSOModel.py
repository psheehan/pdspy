import numpy
from .Model import Model
from .Disk import Disk
from .UlrichEnvelope import UlrichEnvelope
from .Star import Star

class YSOModel(Model):
    def set_grid(self, rmin, rmax, nr, ntheta):
        self.r = numpy.logspace(numpy.log10(rmin), numpy.log10(rmax), nr)
        self.theta = numpy.linspace(0.0, numpy.pi, ntheta)
        self.phi = numpy.array([0.0, 2*numpy.pi])

    def add_star(self, mstar=0.5, lstar=1, tstar=4000.):
        self.stars.append(Star(mstar, lstar, tstar))

    def add_disk(self, mass=1.0e-3, rmin=0.1, rmax=300, plrho=2.37, h0=0.1, \
            plh=58./45., dust="dustkappa_yso.inp"):
        self.disk = Disk(mass=mass, rmin=rmin, rmax=rmax, plrho=plrho, h0=h0, \
                plh=plh,dust=dust)

        self.density.append(self.disk.density(self.r, self.theta))
        self.dust.append(dust)

    def add_ulrich_envelope(self, mass=1.0e-3, rmin=0.1, rmax=1000, cavpl=1.0, \
            cavrfact=0.2, dust="dustkappa_yso.inp"):
        self.envelope = UlrichEnvelope(mass=mass, rmin=rmin, rmax=rmax, \
                rcent=self.disk.rmax, cavpl=cavpl, cavrfact=cavrfact, dust=dust)

        self.density.append(self.envelope.density(self.r, self.theta))
        self.dust.append(dust)
