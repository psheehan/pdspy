import numpy
import h5py
import os
from hyperion.dust import IsotropicDust
from hyperion.model import Model as HypModel
from hyperion.model import ModelOutput
from .. import radmc3d
from .Grid import Grid
from ..constants.astronomy import AU, M_sun, R_sun, L_sun
from ..constants.physics import c

class Model:

    def __init__(self):
        self.grid = Grid()
        self.images = {}
        self.spectra = {}
        self.visibilities = {}

    def run_thermal(self, nphot=1e6, mrw=False, pda=False, code="radmc3d", \
            **keywords):
        if (code == "radmc3d"):
            self.run_thermal_radmc3d(nphot=nphot, mrw=mrw, **keywords)
        else:
            self.run_thermal_hyperion(nphot=nphot, mrw=mrw, **keywords)

    def run_thermal_hyperion(self, nphot=1e6, mrw=False, pda=False, **keywords):
        d = []
        for i in range(len(self.grid.dust)):
            d.append(IsotropicDust( \
                    self.grid.dust[i].nu[::-1].astype(numpy.float64), \
                    self.grid.dust[i].albedo[::-1].astype(numpy.float64), \
                    self.grid.dust[i].kext[::-1].astype(numpy.float64)))

        m = HypModel()
        if (self.grid.coordsystem == "cartesian"):
            m.set_cartesian_grid(self.grid.w1*AU, self.grid.w2*AU, \
                    self.grid.w3*AU)
        elif (self.grid.coordsystem == "cylindrical"):
            m.set_cylindrical_polar_grid(self.grid.w1*AU, self.grid.w3*AU, \
                    self.grid.w2)
        elif (self.grid.coordsystem == "spherical"):
            m.set_spherical_polar_grid(self.grid.w1*AU, self.grid.w2, \
                    self.grid.w3)

        for i in range(len(self.grid.density)):
            if (self.grid.coordsystem == "cartesian"):
                m.add_density_grid(numpy.transpose(self.grid.density[i], \
                        axes=(2,1,0)), d[i])
            if (self.grid.coordsystem == "cylindrical"):
                m.add_density_grid(numpy.transpose(self.grid.density[i], \
                        axes=(1,2,0)), d[i])
            if (self.grid.coordsystem == "spherical"):
                m.add_density_grid(numpy.transpose(self.grid.density[i], \
                        axes=(2,1,0)), d[i])

        sources = []
        for i in range(len(self.grid.stars)):
            sources.append(m.add_spherical_source())
            sources[i].luminosity = self.grid.stars[i].luminosity* L_sun
            sources[i].radius = self.grid.stars[i].radius * R_sun
            sources[i].temperature = self.grid.stars[i].temperature

        m.set_mrw(mrw)
        m.set_pda(pda)
        m.set_n_initial_iterations(20)
        m.set_n_photons(initial=nphot, imaging=0)
        m.set_convergence(True, percentile=99., absolute=2., relative=1.02)

        m.write("test.rtin")

        m.run("test.rtout", mpi=False)

        n = ModelOutput("test.rtout")

        grid = n.get_quantities()

        self.grid.temperature = []
        temperature = grid.quantities['temperature']
        for i in range(len(temperature)):
            if (self.grid.coordsystem == "cartesian"):
                self.grid.temperature.append(numpy.transpose(temperature[i], \
                        axes=(2,1,0)))
            if (self.grid.coordsystem == "cylindrical"):
                self.grid.temperature.append(numpy.transpose(temperature[i], \
                        axes=(2,0,1)))
            if (self.grid.coordsystem == "spherical"):
                self.grid.temperature.append(numpy.transpose(temperature[i], \
                        axes=(2,1,0)))

        os.system("rm test.rtin test.rtout")

    def run_thermal_radmc3d(self, nphot=1e6, mrw=False, **keywords):
        radmc3d.write.control(nphot_therm=nphot, **keywords)

        mstar = []
        rstar = []
        xstar = []
        ystar = []
        zstar = []
        tstar = []

        for i in range(len(self.grid.stars)):
            mstar.append(self.grid.stars[i].mass*M_sun)
            rstar.append(self.grid.stars[i].radius*R_sun)
            xstar.append(self.grid.stars[i].x*AU)
            ystar.append(self.grid.stars[i].y*AU)
            zstar.append(self.grid.stars[i].z*AU)
            tstar.append(self.grid.stars[i].temperature)

        radmc3d.write.stars(rstar, mstar, self.grid.lam, xstar, ystar, zstar, \
                tstar=tstar)

        radmc3d.write.wavelength_micron(self.grid.lam)

        if (self.grid.coordsystem == "cartesian"):
            radmc3d.write.amr_grid(self.grid.w1*AU, self.grid.w2*AU, \
                    self.grid.w3*AU, coordsystem=self.grid.coordsystem)
        elif(self.grid.coordsystem == "cylindrical"):
            radmc3d.write.amr_grid(self.grid.w1*AU, self.grid.w2, \
                    self.grid.w3*AU, coordsystem=self.grid.coordsystem)
        elif(self.grid.coordsystem == "spherical"):
            radmc3d.write.amr_grid(self.grid.w1*AU, self.grid.w2, \
                    self.grid.w3, coordsystem=self.grid.coordsystem)

        radmc3d.write.dust_density(self.grid.density)

        dustopac = []
        for i in range(len(self.grid.dust)):
            dustopac.append("dustkappa_{0:d}.inp".format(i))
            radmc3d.write.dustkappa("{0:d}".format(i), \
                    self.grid.dust[i].lam*1.0e4, self.grid.dust[i].kabs, \
                    ksca=self.grid.dust[i].ksca)

        radmc3d.write.dustopac(dustopac)

        radmc3d.run.thermal()

        self.grid.temperature = radmc3d.read.dust_temperature()
        for i in range(len(self.grid.temperature)):
            self.grid.temperature[i] = self.grid.temperature[i].reshape( \
                    self.grid.density[i].shape)

        os.system("rm *.out *.inp *.dat")

    def read(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "r")
        else:
            f = usefile

        if ('Grid' in f):
            self.grid = Grid()
            self.grid.read(usefile=f['Grid'])

        if (usefile == None):
            f.close()

    def write(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "w")
        else:
            f = usefile

        grid = f.create_group("Grid")
        if hasattr(self, 'grid'):
            self.grid.write(usefile=grid)

        spectra = f.create_group("Spectra")

        images = f.create_group("Images")

        visibilities = f.create_group("Visibilities")

        if (usefile == None):
            f.close()
