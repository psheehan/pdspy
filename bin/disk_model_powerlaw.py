#!/usr/bin/env python3

from pdspy.constants.astronomy import arcsec
import pdspy.interferometry as uv
import pdspy.spectroscopy as sp
import pdspy.imaging as im
import pdspy.modeling as modeling
import pdspy.plotting as plotting
import matplotlib.pyplot as plt
import argparse
import numpy
import sys
import os
import emcee
import corner
from mpi4py import MPI

comm = MPI.COMM_WORLD

################################################################################
#
# Parse command line arguments.
#
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--object')
parser.add_argument('-r', '--resume', action='store_true')
parser.add_argument('-c', '--withhyperion', action='store_true')
parser.add_argument('-p', '--resetprob', action='store_true')
parser.add_argument('-a', '--action', type=str, default="run")
parser.add_argument('-n', '--ncpus', type=int, default=1)
parser.add_argument('-m', '--ncpus_highmass', type=int, default=8)
parser.add_argument('-e', '--withexptaper', action='store_true')
parser.add_argument('-t', '--timelimit', type=int, default=7200)
parser.add_argument('-b', '--trim', type=str, default="")
parser.add_argument('-s', '--SED', action='store_true')
parser.add_argument('-i', '--nice', action='store_true')
parser.add_argument('-l', '--nicelevel', type=int, default=19)
args = parser.parse_args()

# Check whether we are using MPI.

withmpi = comm.Get_size() > 1

# Set the number of cpus to use.

ncpus = args.ncpus
ncpus_highmass = args.ncpus_highmass

# Set the nice level of the subprocesses.

if args.nice:
    nice = args.nicelevel
else:
    nice = None

# Get the source name and check that it has been set correctly.

source = args.object

if source == None:
    print("--object must be specified")
    sys.exit()

# Determine what action should be taken and set some variables appropriately.

if args.action not in ['run','plot']:
    print("Please select a valid action")
    sys.exit(0)

if args.action == 'plot':
    args.resume = True

# Get the items we want to trim.

if args.trim == "":
    trim = []
else:
    trim = args.trim.split(",")

################################################################################
#
# Create a function which returns a model of the data.
#
################################################################################

# Define a likelihood function.

def lnlike(params, visibilities, images, spectra, parameters, plot):

    m = modeling.run_disk_model(visibilities, images, spectra, params, \
            parameters, plot, ncpus=ncpus, ncpus_highmass=ncpus_highmass, \
            with_hyperion=args.withhyperion, timelimit=args.timelimit, \
            source=source, nice=nice, run_thermal=False)

    # Catch whether the model timed out.

    if m == 0.:
        return -numpy.inf

    # A list to put all of the chisq into.

    chisq = []

    # Calculate the chisq for the visibilities.

    for j in range(len(visibilities["file"])):
        chisq.append(-0.5*(numpy.sum((visibilities["data"][j].real - \
                m.visibilities[visibilities["lam"][j]].real)**2 * \
                visibilities["data"][j].weights)) + \
                -0.5*(numpy.sum((visibilities["data"][j].imag - \
                m.visibilities[visibilities["lam"][j]].imag)**2 * \
                visibilities["data"][j].weights)))

    # Calculate the chisq for all of the images.

    for j in range(len(images["file"])):
        chisq.append(-0.5 * (numpy.sum((images["data"][j].image - \
                m.images[images["lam"][j]].image)**2 / \
                images["data"][j].unc**2)))

    # Calculate the chisq for the SED.

    if "total" in spectra:
        chisq.append(-0.5 * (numpy.sum((spectra["total"].flux - \
                m.spectra["SED"].flux)**2 / spectra["total"].unc**2)))

    # Return the sum of the chisq.

    return numpy.array(chisq).sum()

# Define a prior function.

def lnprior(params, parameters, visibilities):
    for key in parameters:
        if not parameters[key]["fixed"]:
            if parameters[key]["limits"][0] <= params[key] <= \
                    parameters[key]["limits"][1]:
                pass
            else:
                return -numpy.inf

    # Make sure that the radii are correct.

    if "logR_in" in params:
        R_in = 10.**params["logR_in"]
    else:
        R_in = 10.**parameters["logR_in"]["value"]

    if "logR_disk" in params:
        R_disk = 10.**params["logR_disk"]
    else:
        R_disk = 10.**parameters["logR_disk"]["value"]

    if "logR_env" in params:
        R_env = 10.**params["logR_env"]
    else:
        R_env = 10.**parameters["logR_env"]["value"]

    if R_in <= R_disk <= R_env:
        pass
    else:
        return -numpy.inf

    # Check that we aren't allowing any absurdly dense models.

    if "logR_env" in params and "logM_env" in params:
        if params["logR_env"] < 0.5 * params["logM_env"] + 4.:
            return -numpy.inf
        else:
            pass

    # Check that the cavity actually falls within the disk.

    if not parameters["logR_cav"]["fixed"]:
        if R_in <= 10.**params["logR_cav"] <= R_disk:
            pass
        else:
            return -numpy.inf

    # Check that the gap is reasonable.

    if not parameters["logR_gap1"]["fixed"]:
        if R_in <= 10.**params["logR_gap1"] - params["w_gap1"]/2:
            pass
        else:
            return -numpy.inf

    # Everything was correct, so continue on.

    lnp = 0.0

    # Add in the priors.

    for i in range(len(visibilities["file"])):
        if (not parameters["flux_unc{0:d}".format(i+1)]["fixed"]) and \
                (parameters["flux_unc{0:d}".format(i+1)]["prior"] == "gaussian"):
            lnp += -0.5 * (params["flux_unc{0:d}".format(i+1)] - \
                    parameters["flux_unc{0:d}".format(i+1)]["value"])**2 / \
                    parameters["flux_unc{0:d}".format(i+1)]["sigma"]**2

    # Return

    return lnp

# Define a probability function.

def lnprob(p, visibilities, images, spectra, parameters, plot):

    keys = []
    for key in sorted(parameters.keys()):
        if not parameters[key]["fixed"]:
            keys.append(key)

    params = dict(zip(keys, p))

    lp = lnprior(params, parameters, visibilities)

    if not numpy.isfinite(lp):
        return -numpy.inf

    return lp + lnlike(params, visibilities, images, spectra, parameters, \
            plot)

################################################################################
#
# In case we are restarting this from the same job submission, delete any
# temporary directories associated with this run.
#
################################################################################

os.system("rm -r /tmp/temp_{0:s}_{1:d}".format(source, comm.Get_rank()))

################################################################################
#
# Set up a pool for parallel runs.
#
################################################################################

if args.action == "run":
    if withmpi:
        pool = emcee.utils.MPIPool(loadbalance=True)

        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    else:
        pool = None

################################################################################
#
# Read in the data.
#
################################################################################

# Import the configuration file information.

sys.path.insert(0, '')

from config import *

# Get the correct binsize and number of bins for averaging the visibility data.

visibilities["gridsize"] = []
visibilities["binsize"] = []

for i in range(len(visibilities["file"])):
    visibilities["gridsize"].append(visibilities["npix"][i])
    visibilities["binsize"].append(1./(visibilities["npix"][i]*\
            visibilities["pixelsize"][i]*arcsec))

# Make sure the spectra dictionary has a "weight" entry.

if not "weight" in spectra:
    spectra["weight"] = [1. for i in range(len(spectra["file"]))]

# Set up the places where we will put all of the data.

visibilities["data"] = []
visibilities["data1d"] = []
visibilities["image"] = []
spectra["data"] = []
spectra["binned"] = []
images["data"] = []

# Check that all of the parameters are correct.

parameters = modeling.check_parameters(parameters, \
        nvis=len(visibilities["file"]))

# Decide whether to use an exponentially tapered 

if args.withexptaper:
    parameters["disk_type"]["value"] = "exptaper"

######################################
# Read in the millimeter visibilities.
######################################

for j in range(len(visibilities["file"])):
    # Read the raw data.

    data = uv.Visibilities()
    data.read(visibilities["file"][j])

    # Center the data. => need to update!

    data = uv.center(data, [visibilities["x0"][j], visibilities["y0"][j], 1.])

    """NEW: interpolate model to baselines instead of averaging the data to the
            model grid?

    visibilities["data"].append(data)
    """

    # Average the data to a more manageable size.

    visibilities["data"].append(uv.grid(data, \
            gridsize=visibilities["gridsize"][j], \
            binsize=visibilities["binsize"][j]))

    # Scale the weights of the visibilities to force them to be fit well.

    visibilities["data"][j].weights *= visibilities["weight"][j]

    # Average the visibilities radially.

    visibilities["data1d"].append(uv.average(data, gridsize=20, radial=True, \
            log=True, logmin=data.uvdist[numpy.nonzero(data.uvdist)].min()*\
            0.95, logmax=data.uvdist.max()*1.05))

    # Clean up the data because we don't need it any more.

    del data

    # Read in the image.

    visibilities["image"].append(im.readimfits(visibilities["image_file"][j]))

######################
# Read in the spectra.
######################

for j in range(len(spectra["file"])):
    spectra["data"].append(sp.Spectrum())
    spectra["data"][j].read(spectra["file"][j])

    # Adjust the weight of the SED, as necessary.

    spectra["data"][j].unc /= spectra["weight"][j]**0.5

    # Merge the SED with the binned Spitzer spectrum.

    if spectra["bin?"][j]:
        wave = numpy.linspace(spectra["data"][j].wave.min(), \
                spectra["data"][j].wave.max(), spectra["nbins"][j])
        flux = numpy.interp(wave, spectra["data"][j].wave, \
                spectra["data"][j].flux)

        good = flux > 0.

        wave = wave[good]
        flux = flux[good]

        spectra["binned"].append(sp.Spectrum(wave, flux))
    else:
        spectra["binned"].append(spectra["data"][j])

    # Merge all the spectra together in one big SED.

    try:
        spectra["total"].wave = numpy.concatenate((spectra["total"].wave, \
                spectra["binned"][j].wave))
        spectra["total"].flux = numpy.concatenate((spectra["total"].flux, \
                numpy.log10(spectra["binned"][j].flux)))
        spectra["total"].unc = numpy.concatenate((spectra["total"].unc, \
                numpy.repeat(0.1, spectra["binned"][j].wave.size)))

        order = numpy.argsort(spectra["total"].wave)

        spectra["total"].wave = spectra["total"].wave[order]
        spectra["total"].flux = spectra["total"].flux[order]
        spectra["total"].unc = spectra["total"].unc[order]
    except:
        spectra["total"] = sp.Spectrum(spectra["binned"][j].wave, \
                numpy.log10(spectra["binned"][j].flux), \
                numpy.repeat(0.1, spectra["binned"][j].wave.size))

#####################
# Read in the images.
#####################

for j in range(len(images["file"])):
    images["data"].append(im.Image())
    images["data"][j].read(images["file"][j])

################################################################################
#
# Fit the model to the data.
#
################################################################################

# Set up the emcee run.

ndim = 0
keys = []

for key in sorted(parameters.keys()):
    if not parameters[key]["fixed"]:
        ndim += 1
        keys.append(key)

# Make the labels nice with LaTeX.

labels = ["$"+key.replace("_","_{").replace("log","\log ")+"}$" \
        if key[0:3] == "log" else "$"+key.replace("h_large","h,large")+\
        "$" for key in keys]

# If we are resuming an MCMC simulation, read in the necessary info, otherwise
# set up the info.

if args.resume:
    pos = numpy.load("pos.npy")
    chain = numpy.load("chain.npy")
    state = None
    nsteps = chain[0,:,0].size

    if args.resetprob:
        prob = None
        prob_list = numpy.empty((nwalkers,0))
    else:
        prob_list = numpy.load("prob.npy")
        if len(prob_list.shape) == 1:
            prob_list = prob_list.reshape((nwalkers,1))
        prob = prob_list[:,-1]
else:
    pos = []
    for j in range(nwalkers):
        m_env = numpy.random.uniform(-6., \
                parameters["logM_env"]["limits"][1],1)[0]

        r_env = numpy.random.uniform(max(parameters["logR_env"]["limits"][0],\
                0.5*m_env+4.), parameters["logR_env"]["limits"][1],1)[0]
        r_disk = numpy.random.uniform(numpy.log10(5.),\
                min(numpy.log10(500.), r_env, \
                parameters["logR_disk"]["limits"][1]),1)[0]
        r_in = numpy.random.uniform(parameters["logR_in"]["limits"][0],\
                numpy.log10((10.**r_disk)/2),1)[0]

        r_cav = numpy.random.uniform(r_in, numpy.log10(0.75*10.**r_disk),1)[0]

        r_gap1 = numpy.random.uniform(numpy.log10(10.**r_in+\
                parameters["w_gap1"]["limits"][0]/2), \
                numpy.log10(0.75*10.**r_disk),1)[0]

        w_gap1 = numpy.random.uniform(parameters["w_gap1"]["limits"][0], \
                min(parameters["w_gap1"]["limits"][1],\
                2*(10.**r_gap1-10.**r_in)), 1)[0]

        temp_pos = []

        for key in sorted(parameters.keys()):
            if parameters[key]["fixed"]:
                pass
            elif key == "logR_in":
                temp_pos.append(r_in)
            elif key == "logR_disk":
                temp_pos.append(r_disk)
            elif key == "logR_env":
                temp_pos.append(r_env)
            elif key == "logR_cav":
                temp_pos.append(r_cav)
            elif key == "logR_gap1":
                temp_pos.append(r_gap1)
            elif key == "w_gap1":
                temp_pos.append(w_gap1)
            elif key == "logM_disk":
                temp_pos.append(numpy.random.uniform(-6.,\
                        parameters[key]["limits"][1],1)[0])
            elif key == "logM_env":
                temp_pos.append(m_env)
            elif key == "h_0":
                temp_pos.append(numpy.random.uniform(\
                        parameters[key]["limits"][0], 0.2, 1)[0])
            elif key[0:8] == "flux_unc":
                temp_pos.append(numpy.random.normal(\
                        parameters[key]["value"], 0.001, 1)[0])
            else:
                temp_pos.append(numpy.random.uniform(\
                        parameters[key]["limits"][0], \
                        parameters[key]["limits"][1], 1)[0])

        pos.append(temp_pos)

    prob = None
    prob_list = numpy.empty((nwalkers, 0))
    chain = numpy.empty((nwalkers, 0, ndim))
    state = None
    nsteps = 0

# Set up the MCMC simulation.

if args.action == "run":
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, \
            args=(visibilities, images, spectra, parameters, False), \
            pool=pool)

# Run a few burner steps.

if args.action == "plot":
    nsteps = max_nsteps-1

while nsteps < max_nsteps:
    if args.action == "run":
        for i in range(steps_per_iter):
            pos, prob, state = sampler.run_mcmc(pos, 1, lnprob0=prob, \
                    rstate0=state)

            chain = numpy.concatenate((chain, sampler.chain), axis=1)

            prob_list = numpy.concatenate((prob_list, prob.\
                    reshape((nwalkers,1))), axis=1)

            # Plot the steps of the walkers.

            for j in range(ndim):
                fig, ax = plt.subplots(nrows=1, ncols=1)

                for k in range(nwalkers):
                    ax.plot(chain[k,:,j])

                plt.savefig("steps_{0:s}.png".format(keys[j]))

                plt.close(fig)

            # Save walker positions in case the code stps running for some 
            # reason.

            numpy.save("pos", pos)
            numpy.save("prob", prob_list)
            numpy.save("chain", chain)

            # Augment the nuber of steps and reset the sampler for the next run.

            nsteps += 1

            sampler.reset()

    # Get the best fit parameters and uncertainties from the last 10 steps.

    samples = chain[:,-nplot:,:].reshape((-1, ndim))

    # Make the cuts specified by the user.

    for command in trim:
        command = command.split(" ")

        for i, key in enumerate(keys):
            if key == command[0]:
                if command[1] == '<':
                    good = samples[:,i] > float(command[2])
                else:
                    good = samples[:,i] < float(command[2])

                samples = samples[good,:]

    # Get the best fit parameters.

    params = numpy.median(samples, axis=0)
    sigma = samples.std(axis=0)

    # Write out the results.

    f = open("fit.txt", "w")
    f.write("Best fit parameters:\n\n")
    for j in range(len(keys)):
        f.write("{0:s} = {1:f} +/- {2:f}\n".format(keys[j], params[j], \
                sigma[j]))
    f.write("\n")
    f.close()

    os.system("cat fit.txt")

    # Plot histograms of the resulting parameters.

    fig = corner.corner(samples, labels=labels, truths=params)

    plt.savefig("fit.pdf")

    # Make a dictionary of the best fit parameters.

    params = dict(zip(keys, params))

    ############################################################################
    #
    # Plot the results.
    #
    ############################################################################

    # Plot the best fit model over the data.

    fig, ax = plt.subplots(nrows=2*len(visibilities["file"]), ncols=3)

    # Create a high resolution model for averaging.

    m = modeling.run_disk_model(visibilities, images, spectra, params, \
            parameters, plot=True, ncpus=ncpus, ncpus_highmass=ncpus_highmass, \
            with_hyperion=args.withhyperion, timelimit=args.timelimit, \
            source=source, nice=nice, run_thermal=False)

    # Plot the millimeter data/models.

    for j in range(len(visibilities["file"])):
        # Plot the visibilities.

        plotting.plot_1D_visibilities(visibilities, m, parameters, params, \
                index=j, fig=(fig, ax[2*j,0]))

        # Plot the 2D visibilities.

        plotting.plot_2D_visibilities(visibilities, m, parameters, params, \
                index=j, fig=(fig, ax[2*j+1,0:2]))

        # Plot the model image.

        plotting.plot_continuum_image(visibilities, m, parameters, params, \
                index=j, fig=(fig, ax[2*j,1]))

    # Plot the SED.

    plotting.plot_SED(spectra, m, SED=args.SED, fig=(fig, ax[0,2]))

    # Plot the scattered light image.

    for j in range(len(images["file"])):
        plotting.plot_scattered_light(visibilities, m, parameters, params, \
                index=j, fig=(fig, ax[1,2]))

    # Turn off axes when they aren't being used.

    if len(images["file"]) == 0:
        ax[1,2].set_axis_off()

    for j in range(len(visibilities["file"])):
        if j > 0:
            ax[2*j,2].set_axis_off()
            ax[2*j+1,2].set_axis_off()

    # Adjust the plot.

    fig.set_size_inches((12.5,8*len(visibilities["file"])))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.08, \
            wspace=0.25, hspace=0.2)

    # Save the figure.

    fig.savefig("model.pdf")

    plt.close(fig)

    if args.action == "plot":
        nsteps = numpy.inf

# Now we can close the pool and end the code.

if args.action == "run":
    if withmpi:
        pool.close()
