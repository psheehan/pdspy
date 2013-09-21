import numpy

class Grid:

    def __init__(self):
        self.density = []
        self.dust = []
        self.temperature = []
        self.stars = []

    def add_density(self, density, dust):
        self.density.append(density)
        self.dust.append(dust)

    def add_star(self, star):
        self.stars.append(star)

    def add_temperature(self, temperature):
        self.temperature.append(temperature)

    def set_cartesian_grid(self, x, y, z):
        self.coordsystem = "cartesian"

        self.x = x
        self.y = y
        self.z = z

        self.w1 = self.x
        self.w2 = self.y
        self.w3 = self.z

    def set_cylindrical_grid(self, r, phi, z):
        self.coordsystem = "cylindrical"

        self.x = r
        self.y = phi
        self.z = z

        self.w1 = self.r
        self.w2 = self.phi
        self.w3 = self.z

    def set_spherical_grid(self, r, theta, phi):
        self.coordsystem = "spherical"

        self.r = r
        self.theta = theta
        self.phi = phi

        self.w1 = self.r
        self.w2 = self.theta
        self.w3 = self.phi

    def set_wavelength_grid(self, lmin, lmax, nlam, log=False):
        if log:
            self.lam = numpy.logspace(numpy.log10(lmin), numpy.log10(lmax), \
                    nlam)
        else:
            self.lam = numpy.linspace(lmin, lmax, nlam)
