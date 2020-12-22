import numpy
import h5py
from ..dust import Dust
from ..gas import Gas
from .Star import Star

class Grid:

    def __init__(self):
        self.density = []
        self.dust = []
        self.temperature = []
        self.stars = []
        self.number_density = []
        self.gas_temperature = []
        self.gas = []
        self.velocity = []
        self.microturbulence = []
        self.scattering_phase_freq = []
        self.scattering_phase = []

    def add_density(self, density, dust):
        self.density.append(density)
        self.dust.append(dust)

    def add_star(self, star):
        self.stars.append(star)

    def add_temperature(self, temperature):
        self.temperature.append(temperature)

    def add_number_density(self, number_density, gas):
        self.number_density.append(number_density)
        self.gas.append(gas)

    def add_gas_temperature(self, gas_temperature):
        self.gas_temperature.append(gas_temperature)

    def add_velocity(self, velocity):
        self.velocity.append(velocity)

    def add_microturbulence(self, microturbulence):
        self.microturbulence.append(microturbulence)

    def add_scattering_phase(self, freq, scattering_phase):
        self.scattering_phase_freq = freq
        self.scattering_phase = scattering_phase

    def set_cartesian_grid(self, w1, w2, w3):
        self.coordsystem = "cartesian"

        self.x = 0.5*(w1[0:w1.size-1] + w1[1:w1.size])
        self.y = 0.5*(w2[0:w2.size-1] + w2[1:w2.size])
        self.z = 0.5*(w3[0:w3.size-1] + w3[1:w3.size])

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def set_cylindrical_grid(self, w1, w2, w3):
        self.coordsystem = "cylindrical"

        self.rho = 0.5*(w1[0:w1.size-1] + w1[1:w1.size])
        self.phi = 0.5*(w2[0:w2.size-1] + w2[1:w2.size])
        self.z = 0.5*(w3[0:w3.size-1] + w3[1:w3.size])

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def set_spherical_grid(self, w1, w2, w3):
        self.coordsystem = "spherical"

        self.r = 0.5*(w1[0:w1.size-1] + w1[1:w1.size])
        self.theta = 0.5*(w2[0:w2.size-1] + w2[1:w2.size])
        self.phi = 0.5*(w3[0:w3.size-1] + w3[1:w3.size])

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def set_wavelength_grid(self, lmin, lmax, nlam, log=False):
        if log:
            self.lam = numpy.logspace(numpy.log10(lmin), numpy.log10(lmax), \
                    nlam)
        else:
            self.lam = numpy.linspace(lmin, lmax, nlam)

    def read(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "r")
        else:
            f = usefile

        coordsystem = f['coordsystem'][()]
        w1 = f['w1'][...]
        w2 = f['w2'][...]
        w3 = f['w3'][...]

        if type(coordsystem) == bytes:
            coordsystem = coordsystem.decode('utf-8')

        if (coordsystem == 'cartesian'):
            self.set_cartesian_grid(w1, w2, w3)
        elif (coordsystem == 'cylindrical'):
            self.set_cylindrical_grid(w1, w2, w3)
        elif (coordsystem == 'spherical'):
            self.set_spherical_grid(w1, w2, w3)

        density = f['Density']
        for name in density:
            self.density.append(density[name][...])

        dust = f['Dust']
        for name in dust:
            d = Dust()
            d.set_properties_from_file(usefile=dust[name])
            self.dust.append(d)

        temperature = f['Temperature']
        for name in temperature:
            self.temperature.append(temperature[name][...])

        stars = f['Stars']
        for name in stars:
            star = Star()
            star.read(usefile=stars[name])
            self.stars.append(star)

        number_density = f['NumberDensity']
        for name in number_density:
            self.number_density.append(number_density[name][...])

        if 'GasTemperature' in f:
            gas_temperature = f['GasTemperature']
            for name in gas_temperature:
                self.gas_temperature.append(gas_temperature[name][...])

        if 'Microturbulence' in f:
            microturbulence = f['Microturbulence']
            for name in microturbulence:
                self.microturbulence.append(microturbulence[name][...])

        if "ScatteringPhase" in f:
            scattering_phase = f["ScatteringPhase"]
            self.scattering_phase = scattering_phase["ScatteringPhase"][...]
            self.scattering_phase_freq = \
                    scattering_phase["ScatteringPhaseFreq"][...]

        gas = f['Gas']
        for name in gas:
            g = Gas()
            g.set_properties_from_file(usefile=gas[name])
            self.gas.append(g)

        velocity = f['Velocity']
        for name in velocity:
            self.velocity.append(velocity[name][...])

        if ('lam' in f):
            self.lam = f['lam'][...]

        if (usefile == None):
            f.close()

    def write(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "w")
        else:
            f = usefile

        f['coordsystem'] = self.coordsystem
        w1_dset = f.create_dataset("w1", (self.w1.size,), dtype='f')
        w1_dset[...] = self.w1
        w2_dset = f.create_dataset("w2", (self.w2.size,), dtype='f')
        w2_dset[...] = self.w2
        w3_dset = f.create_dataset("w3", (self.w3.size,), dtype='f')
        w3_dset[...] = self.w3

        density = f.create_group("Density")
        density_dsets = []
        for i in range(len(self.density)):
            density_dsets.append(density.create_dataset( \
                    "Density{0:d}".format(i), self.density[i].shape, dtype='f'))
            density_dsets[i][...] = self.density[i]

        dust = f.create_group("Dust")
        dust_groups = []
        for i in range(len(self.dust)):
            dust_groups.append(dust.create_group("Dust{0:d}".format(i)))
            self.dust[i].write(usefile=dust_groups[i])

        temperature = f.create_group("Temperature")
        temperature_dsets = []
        for i in range(len(self.temperature)):
            temperature_dsets.append(temperature.create_dataset( \
                    "Temperature{0:d}".format(i), self.temperature[i].shape, \
                    dtype='f'))
            temperature_dsets[i][...] = self.temperature[i]

        stars = f.create_group("Stars")
        stars_groups = []
        for i in range(len(self.stars)):
            stars_groups.append(stars.create_group("Star{0:d}".format(i)))
            self.stars[i].write(usefile=stars_groups[i])

        number_density = f.create_group("NumberDensity")
        number_density_dsets = []
        for i in range(len(self.number_density)):
            number_density_dsets.append(number_density.create_dataset( \
                    "NumberDensity{0:d}".format(i), \
                    self.number_density[i].shape, dtype='f'))
            number_density_dsets[i][...] = self.number_density[i]

        gas_temperature = f.create_group("GasTemperature")
        gas_temperature_dsets = []
        for i in range(len(self.gas_temperature)):
            gas_temperature_dsets.append(gas_temperature.create_dataset( \
                    "GasTemperature{0:d}".format(i), \
                    self.gas_temperature[i].shape, dtype='f'))
            gas_temperature_dsets[i][...] = self.gas_temperature[i]

        microturbulence = f.create_group("Microturbulence")
        microturbulence_dsets = []
        for i in range(len(self.microturbulence)):
            microturbulence_dsets.append(microturbulence.create_dataset( \
                    "Microturbulence{0:d}".format(i), \
                    self.microturbulence[i].shape, dtype='f'))
            microturbulence_dsets[i][...] = self.microturbulence[i]

        if len(self.scattering_phase) > 0:
            scattering_phase = f.create_group("ScatteringPhase")
            scattering_phase_dsets = []
            scattering_phase_dsets.append(scattering_phase.create_dataset( \
                    "ScatteringPhase", self.scattering_phase.shape, dtype='f'))
            scattering_phase_dsets[0][...] = self.scattering_phase
            scattering_phase_dsets.append(scattering_phase.create_dataset( \
                    "ScatteringPhaseFreq", self.scattering_phase_freq.shape, \
                    dtype='f'))
            scattering_phase_dsets[1][...] = self.scattering_phase_freq

        gas = f.create_group("Gas")
        gas_groups = []
        for i in range(len(self.gas)):
            gas_groups.append(gas.create_group("Gas{0:d}".format(i)))
            self.gas[i].write(usefile=gas_groups[i])

        velocity = f.create_group("Velocity")
        velocity_dsets = []
        for i in range(len(self.velocity)):
            velocity_dsets.append(velocity.create_dataset("Velocity{0:d}". \
                    format(i), self.velocity[i].shape, dtype='f'))
            velocity_dsets[i][...] = self.velocity[i]

        if hasattr(self, 'lam'):
            lam_dset = f.create_dataset("lam", self.lam.shape, dtype='f')
            lam_dset[...] = self.lam

        if (usefile == None):
            f.close()
