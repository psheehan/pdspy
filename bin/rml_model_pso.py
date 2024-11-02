#!/usr/bin/env python3

from pdspy.modeling import RMLModel
from pdspy import dust
import pdspy.plotting as plotting
import pdspy.utils as utils
import argparse
import numpy
import sys

################################################################################
#
# Parse command line arguments.
#
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--object')
parser.add_argument('-n', '--ncpus', type=int, default=1)
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-f', '--ftcode', type=str, default="galario")
args = parser.parse_args()

# Set the number of cpus to use.

ncpus = args.ncpus

# Get the source name and check that it has been set correctly.

source = args.object

if source == None:
    print("--object must be specified")
    sys.exit()

################################################################################
#
# Read in the data.
#
################################################################################

# Import the configuration file information.

config = utils.load_config()

# Read in the data.

visibilities, images, spectra = utils.load_data(config)

################################################################################
#
# Fit the model to the data.
#
################################################################################

model = RMLModel()
model.add_star(luminosity=1.0)
model.set_spherical_grid(0.1, 500., 100, 101, 2, code="radmc3d")

dust_gen = dust.DustGenerator(dust.__path__[0]+"/data/diana_wice.hdf5")
dust = dust_gen(1000. / 1e4, 3.5)

model.add_dust_layer(dust)

model.grid.set_wavelength_grid(0.1,1.0e5,500,log=True)

def negative_log_likelihood(p, visibilities, images, spectra):
    params = p.copy()
    params[2:] = 10.**params[2:]
    model.set_parameter_vector(params, verbose=True)

    return model.log_likelihood(visibilities, images, spectra, verbose=True)

# Hyper-parameter of the algorithm
c1 = c2 = 0.1
w = 0.8

# Create particles
n_particles = 5
X = numpy.random.rand(model.get_parameter_vector().size, n_particles)
V = numpy.random.randn(model.get_parameter_vector().size, n_particles)

X[0,:] *= 90. # Inclination from 0 to 90.
X[1,:] *= 180. # Inclination from 0 to 90.
X[2,:] *= 2. # Log luminosity from 0. to 2.
X[3:,:] = numpy.broadcast_to(X[3,:], (X[3:,0].size, X[3,:].size))*6 - 3. + numpy.log10(model.grid.density[0].mean())

V[0:2,:] *= 5.
V[2:,:] = numpy.broadcast_to(V[2,:], (V[2:,0].size, V[2,:].size))* 0.1

# Initialize data
pbest = X
pbest_obj = numpy.array([negative_log_likelihood(x, visibilities, images, spectra) for x in X.T])
gbest = pbest[:, pbest_obj.argmin()]
gbest_obj = pbest_obj.min()

def update():
    "Function to do one iteration of particle swarm optimization"
    global V, X, pbest, pbest_obj, gbest, gbest_obj
    # Update params
    r1, r2 = numpy.random.rand(2)
    V = w * V + c1*r1*(pbest - X) + c2*r2*(gbest.reshape(-1,1)-X)
    X = X + V
    obj = numpy.array([negative_log_likelihood(x, visibilities, images, spectra) for x in X.T])
    pbest[:, (pbest_obj >= obj)] = X[:, (pbest_obj >= obj)]
    pbest_obj = numpy.array([pbest_obj, obj]).min(axis=0)
    gbest = pbest[:, pbest_obj.argmin()]
    gbest_obj = pbest_obj.min()

for i in range(3):
    update()
