#!/usr/bin/env python

from scipy.integrate import trapz
from pdspy.constants.astronomy import AU, M_sun
from pdspy.constants.physics import c
import pdspy.interferometry as uv
import pdspy.modeling as modeling
import pdspy.utils as utils
import scipy.integrate
import emcee
import numpy
import glob
import sys
import os

# First check what version of the code was run.

if len(glob.glob("chain.npy")) > 0:
    version = "emcee"
elif len(glob.glob("results.hdf5")) > 0:
    version = "emcee3"
elif len(glob.glob("samples.npy")) > 0 or len(glob.glob("sampler.p")) > 0:
    version = "dynesty"
else:
    version = "none"
    print("Did not detect any pdspy results... just upgrading config.py and data.")

if version != "none":
    print("Detected results from the {0:s} version of pdspy.".format(version))

if version == "dynesty":
    print("** Please note that because of the way that dynesty is run, it is impossible to update sampler.p. This means you will not be able to resume an in-progress fit, nor will you be able to re-run disk_model_nested.py, flared_model_nested, etc. within this model directory and get correct results. The resulting samples.npy will be updated so that you can still work with the results, if they exist. Otherwise, you will need to re-run the fit. **")

# Now load in the configuration file.

config = utils.load_config()

# Determine what files are going to be changed.

print("Checking which files need to be updated...")

changed_files = ["config.py"]

for f in config.visibilities["file"]:
    if os.path.exists("/".join(f.split("/")[0:-1]+[".upgraded_to_pdspy2"])):
        print(f+" has already been upgraded... skipping.")
    else:
        changed_files.append(f)

if version == "emcee":
    if os.path.exists("chain.npy"):
        changed_files += ["chain.npy","pos.npy","prob.npy"]
elif version == "emcee3":
    if os.path.exists("results.hdf5"):
        changed_files += ["results.hdf5"]
elif version == "dynesty":
    if os.path.exists("samples.npy"):
        changed_files += ["samples.npy"]
    if os.path.exists("sampler.p"):
        changed_files += ["sampler.p"]

print("Permanent changes will be made to the following files:")
for f in changed_files:
    print("    {0:s}".format(f))
result = input("Proceed? (y/n)")

if result == "n":
    sys.exit(0)

# Check whether this directory was already converted.

if os.path.exists(".upgraded_to_pdspy2"):
    print("Based on the presence of a .upgraded_to_pdspy2 file in this directory, it seems that this directory has already been converted to v2. Re-running the code is dangerous and would likely lead to incorrect results. If you do really want to re-run, please restore the files in this directory to their original state and then delete the .upgraded_to_pdspy file.")
    sys.exit(0)
else:
    f = open(".upgraded_to_pdspy2","w")
    f.write("upgraded_to_pdspy2\n")
    f.close()

# Update the visibility data to take the complex conjugate.

for j, f in enumerate(config.visibilities["file"]):
    if not os.path.exists("/".join(f.split("/")[0:-1]+[".upgraded_to_pdspy2"])):
        print("Updating "+f)

        data = uv.Visibilities()
        data.read(f)

        data.imag *= -1.

        # Check whether there's a difference in config.visibilities["lam"][j]
        # and the frequency of the data.

        if data.freq.size == 1:
            data.freq[0] = c / (float(config.visibilities["lam"][j])*1.0e-4)
            
            model = "disk"
        else:
            model = "flared"

        # Write out the updated data file.

        os.system("mv "+f+" "+f+".backup")

        data.write(f)

        u = open("/".join(f.split("/")[0:-1]+[".upgraded_to_pdspy2"]),"w")
        u.write("upgraded_to_pdspy2\n")
        u.close()

# Finally, update config.py to have the appropriate numbers.

print("Updating config.py")

os.system("cp config.py config.py.backup")

f = open("config.py","r")
lines = f.readlines()
f.close()

new_lines = []
for i, line in enumerate(lines):
    if "x0" in line or "y0" in line:
        val = line[line.find("[")+1 : line.find("]")]
        vals = val.split(",")
        new_vals = [str(float(v)*-1) for v in vals]
        if "fixed" in line and len(vals) == 2:
            new_vals = new_vals[::-1]
        line = line.replace(val, ",".join(new_vals))
    elif '"pa"' in line:
        val = line[line.find("[")+1 : line.find("]")]
        vals = val.split(",")
        new_vals = [str(-float(v)) for v in vals][::-1]
        line = line.replace(val, ",".join(new_vals))
    new_lines.append(line)

f = open("config.py","w")
for line in new_lines:
    f.write(line)
f.close()

# First, if the results are emcee < 3, convert to emcee3.

if version == "emcee" and os.path.exists("chain.npy"):
    print("Converting to emcee v3")

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
        sample = emcee.State(chain[:,i,:], log_prob=prob[:,i], \
                random_state=state)
        accepted = numpy.ones((nwalkers,))

        backend.save_step(sample, accepted)

    # Finally, move the old results to .backup files.

    os.system("mv chain.npy chain.npy.backup")
    os.system("mv prob.npy prob.npy.backup")
    os.system("mv pos.npy pos.npy.backup")

    version = "emcee3"

# Now, change the values of tne appropriate parameters in the results.

if os.path.exists("samples.npy") or os.path.exists("results.hdf5"):
    if version == "dynesty":
        samples = numpy.load("samples.npy")
    elif version == "emcee3":
        results = emcee.backends.HDFBackend("results.hdf5")
        chain = results.get_chain()
        prob = results.get_log_prob()

    # Get the keys of parameters and the index of particular ones that need to 
    # be updated.

    keys = []
    for key in sorted(config.parameters.keys()):
        if not config.parameters[key]["fixed"]:
            keys.append(key)
    ndim = len(keys)

    for ikey in range(ndim):
        if keys[ikey] == "x0":
            ix0 = ikey
        elif keys[ikey] == "y0":
            iy0 = ikey
        elif keys[ikey] == "pa":
            ipa = ikey
        elif keys[ikey] == "logM_disk":
            iMdisk = ikey
        elif keys[ikey] == "logM_env":
            iMenv = ikey

    # Define a function that calculates the correct masses.

    def calculate_correct_masses(params, disk=True, envelope=True):
        # Generate the model.

        m = modeling.run_disk_model(config.visibilities, config.images, \
                config.spectra, params, config.parameters, plot=False, \
                ncpus=4, ncpus_highmass=4, source="V883Ori", \
                no_radiative_transfer=True)

        rho, theta, phi = numpy.meshgrid(m.grid.r*AU, m.grid.theta, \
                m.grid.phi, indexing='ij')

        # Update the disk density by calculating the unnormalized density, i.e. 
        # how we used to do it, and integrating to get the normalization.

        if disk:
            r_high = numpy.logspace(numpy.log10(m.disk.rmin), \
                    numpy.log10(10*m.disk.rmax), 1000)
            Sigma_high = m.disk.surface_density(r_high, normalize=False)

            disk_mass = 2*numpy.pi*trapz(r_high*AU*Sigma_high, r_high*AU)
        else:
            disk_mass = None

        # Update the envelope mass as well.

        if envelope:
            env_density_new = m.envelope.density(m.grid.r, m.grid.theta, \
                    m.grid.phi)

            m.envelope.cavrfact = 1.0
            env_density_old = m.envelope.density(m.grid.r, m.grid.theta, \
                    m.grid.phi)

            scale = (env_density_old / env_density_new).min()

            envelope_density = scale * env_density_new

            env_mass = (4*numpy.pi*scipy.integrate.trapz(scipy.integrate.trapz(\
                    envelope_density*rho**2*numpy.sin(theta), theta, axis=1), \
                    rho[:,0,:],axis=0))[0]
        else:
            env_mass = None

        return disk_mass, env_mass

    # Now do the conversions.

    if version == "dynesty":
        print("Updating samples.npy")

        for i in range(samples.shape[0]):
            if "x0" in keys:
                samples[i,ix0] *= -1
            if "y0" in keys:
                samples[i,iy0] *= -1
            if "pa" in keys:
                samples[i,ipa] *= -1
                if model == "disk":
                    samples[i,ipa] += 180

            params = dict(zip(keys,samples[i,:]))
            disk_mass, env_mass = calculate_correct_masses(params, \
                    disk="logM_disk" in keys, envelope="logM_env" in keys)

            if "logM_disk" in keys:
                samples[i,iMdisk] = numpy.log10(disk_mass/M_sun)
            if "logM_env" in keys:
                samples[i,iMenv] = numpy.log10(env_mass/M_sun)

        os.system("cp samples.npy samples.npy.backup")
        numpy.save("samples.npy", samples)

    elif version == "emcee3":
        print("Updating results.hdf5")

        for i in range(chain.shape[0]):
            for j in range(chain.shape[1]):
                # Fix x0, y0, and pa.

                if "x0" in keys:
                    chain[i,j,ix0] *= -1
                if "y0" in keys:
                    chain[i,j,iy0] *= -1
                if "pa" in keys:
                    chain[i,j,ipa] *= -1
                    if model == "disk":
                        chain[i,j,ipa] += 180

                params = dict(zip(keys,chain[i,j,:]))
                disk_mass, env_mass = calculate_correct_masses(params, \
                        disk="logM_disk" in keys, envelope="logM_env" in keys)

                if "logM_disk" in keys:
                    chain[i,j,iMdisk] = numpy.log10(disk_mass/M_sun)
                if "logM_env" in keys:
                    chain[i,j,iMenv] = numpy.log10(env_mass/M_sun)

        # Save the updated chain.

        os.system("cp results.hdf5 results.hdf5.backup")

        nsteps = chain.shape[0]
        nwalkers = chain.shape[1]
        ndim = chain.shape[2]

        state = numpy.random.mtrand.RandomState().get_state()

        backend.reset(nwalkers, ndim)
        backend.grow(nsteps, None)

        for i in range(nsteps):
            sample = emcee.State(chain[i], log_prob=prob[i], random_state=state)
            accepted = numpy.ones((nwalkers,))

            backend.save_step(sample, accepted)

# If sampler.p exists, move it to backup so you can't resume a fit.

if version == "dynesty" and os.path.exists("sampler.p"):
    print("Moving sampler.p to sampler.p.backup so the dynesty fit cannot be resumed.")
    os.system("mv sampler.p sampler.p.backup")
