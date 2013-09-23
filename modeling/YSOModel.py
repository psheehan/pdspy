import numpy
import h5py
from .Model import Model
from .Disk import Disk
from .UlrichEnvelope import UlrichEnvelope
from .Star import Star

class YSOModel(Model):

    def add_star(self, mass=0.5, luminosity=1, temperature=4000.):
        self.grid.add_star(Star(mass=mass, luminosity=luminosity, \
                temperature=temperature))

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
