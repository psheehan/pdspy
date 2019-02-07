import numpy
import h5py
from ..constants.physics import sigma
from ..constants.astronomy import L_sun, R_sun

class Star:
    def __init__(self, mass=0.5, luminosity=1.0, temperature=4000., x=0.0, \
            y=0.0, z=0.0):
        self.mass = mass
        self.luminosity = luminosity
        self.temperature = temperature
        self.radius = (luminosity*L_sun/ \
                (4*numpy.pi*sigma*temperature**4))**(1./2)/R_sun
        self.x = x
        self.y = y
        self.z = z

    def read(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "r")
        else:
            f = usefile

        self.mass = f['mass'][()]
        self.luminosity = f['luminosity'][()]
        self.temperature = f['temperature'][()]
        self.radius = f['radius'][()]
        self.x = f['x'][()]
        self.y = f['y'][()]
        self.z = f['z'][()]

        if (usefile == None):
            f.close()

    def write(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "w")
        else:
            f = usefile

        f['mass'] = self.mass
        f['luminosity'] = self.luminosity
        f['temperature'] = self.temperature
        f['radius'] = self.radius

        f['x'] = self.x
        f['y'] = self.y
        f['z'] = self.z

        if (usefile == None):
            f.close()
