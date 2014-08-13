import numpy
import h5py
from .Model import Model
from .Disk import Disk
from .UlrichEnvelope import UlrichEnvelope
from .Star import Star
from ..constants.physics import h, c, G, m_p, k
from ..constants.astronomy import AU, M_sun, kms, R_sun
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
            plh=58./45., dust=None, gas=None, abundance=None):
        self.disk = Disk(mass=mass, rmin=rmin, rmax=rmax, plrho=plrho, h0=h0, \
                plh=plh, dust=dust)

        if (dust != None):
            self.grid.add_density(self.disk.density(self.grid.r, \
                    self.grid.theta, self.grid.phi),dust)

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
            else:
                self.disk.add_gas(gas, abundance)
                self.grid.add_number_density(self.disk.number_density( \
                        self.grid.r, self.grid.theta, self.grid.phi, \
                        gas=0), gas)
                self.grid.add_velocity(self.disk.velocity(self.grid.r, \
                        self.grid.theta, self.grid.phi, \
                        mstar=self.grid.stars[0].mass))

    def add_ulrich_envelope(self, mass=1.0e-3, rmin=0.1, rmax=1000, rcent=300, \
            cavpl=1.0, cavrfact=0.2, dust=None, gas=None, abundance=None, \
            rcent_ne_rdisk=False):
        if rcent_ne_rdisk:
            self.envelope = UlrichEnvelope(mass=mass, rmin=rmin, rmax=rmax, \
                    rcent=rcent, cavpl=cavpl, cavrfact=cavrfact, dust=dust)
        elif hasattr(self, 'disk'):
            self.envelope = UlrichEnvelope(mass=mass, rmin=rmin, rmax=rmax, \
                    rcent=self.disk.rmax, cavpl=cavpl, cavrfact=cavrfact, \
                    dust=dust)
        else:
            self.envelope = UlrichEnvelope(mass=mass, rmin=rmin, rmax=rmax, \
                    rcent=rcent, cavpl=cavpl, cavrfact=cavrfact, dust=dust)

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
            else:
                self.envelope.add_gas(gas, abundance)
                self.grid.add_number_density(self.envelope.number_density( \
                        self.grid.r, self.grid.theta, self.grid.phi, gas=0), \
                        gas)
                self.grid.add_velocity(self.envelope.velocity(self.grid.r, \
                        self.grid.theta, self.grid.phi, \
                        mstar=self.grid.stars[0].mass))

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

        T = self.disk.temperature(rpp, T_0=T0, p=plT)

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
            upper = self.grid.temperature[i][:,0:ntheta/2,:]
            lower = self.grid.temperature[i][:,ntheta/2:,:][:,::-1,:]
            average = 0.5 * (upper + lower)

            self.grid.temperature[i][:,0:ntheta/2,:] = average
            self.grid.temperature[i][:,ntheta/2:,:] =  average[:,::-1,:]

    def convert_hyperion_to_radmc3d(self):
        self.grid.r = self.grid.r[1:]
        self.grid.w1 = self.grid.w1[1:]

        ntheta = len(self.grid.theta)
        self.grid.theta = self.grid.theta[0:ntheta/2]
        self.grid.w2 = self.grid.w2[0:ntheta/2+1]
        
        for i in range(len(self.grid.density)):
            self.grid.density[i] = self.grid.density[i][1:,0:ntheta/2,:]
        for i in range(len(self.grid.temperature)):
            self.grid.temperature[i] = self.grid.temperature[i][1:,0:ntheta/2,:]
        for i in range(len(self.grid.number_density)):
            self.grid.number_density[i] = self.grid.number_density[i][1:,0:ntheta/2,:]
        for i in range(len(self.grid.velocity)):
            self.grid.velocity[i] = self.grid.velocity[i][:,1:,0:ntheta/2,:]

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
