import h5py
from .. import radmc3d
from .Grid import Grid

class Model:

    def __init__(self):
        self.grid = Grid()
        self.images = {}
        self.spectra = {}
        self.visibilities = {}

    def run_thermal(self, nphot=1e6, mrw=False, code="radmc3d", **keywords):
        if (code == "radmc3d"):
            self.run_thermal_radmc3d(nphot=nphot, mrw=mrw, **keywords)
        else:
            print("Sorry, but {0:s} is not a supported code right "
                  "now.\n".format(code))

    def run_thermal_radmc3d(self, nphot=1e6, mrw=False, **keywords):
        radmc3d.write.control(nphot_therm=nphot, **keywords)

        mstar = []
        rstar = []
        xstar = []
        ystar = []
        zstar = []
        tstar = []

        for i in range(len(self.stars)):
            mstar.append(self.grid.stars[i].mass)
            rstar.append(self.grid.stars[i].radius)
            xstar.append(self.grid.stars[i].x)
            ystar.append(self.grid.stars[i].y)
            zstar.append(self.grid.stars[i].z)
            tstar.append(self.grid.stars[i].temperature)

        radmc3d.write.stars(rstar, mstar, self.lam, xstar, ystar, zstar, \
                tstar=tstar)

        radmc3d.write.wavelength_micron(lam)

        radmc3d.write.amr_grid(self.grid.w1, self.grid.w2, self.grid.w3, \
                coordsystem=self.grid.coordsystem)

        radmc3d.write.dust_density(self.grid.density)

        #radmc3d.write.dustopac(self.grid.dust)

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
