#!/usr/bin/env python3

import emcee
import numpy

# Load in the current information.

chain = numpy.load("chain.npy")
prob = numpy.load("prob.npy")

nwalkers = chain.shape[0]
nsteps = chain.shape[1]
ndim = chain.shape[2]

# Create an artificial random state.

state = numpy.random.mtrand.RandomState().get_state()

# Adjust prob to be the same shape as the chain.

if prob.shape[1] < chain.shape[1]:
    new_prob = numpy.zeros((chain.shape[0],chain.shape[1]))
    new_prob[:,-prob.shape[1]:] = prob.copy()
    prob = new_prob.copy()

# Create an emcee backend.

backend = emcee.backends.HDFBackend("results.hdf5")
backend.reset(nwalkers, ndim)

backend.grow(nsteps, None)

# Now loop through the steps and save to the backend.

for i in range(nsteps):
    sample = emcee.State(chain[:,i,:], log_prob=prob[:,i], random_state=state)
    accepted = numpy.ones((nwalkers,))

    backend.save_step(sample, accepted)
