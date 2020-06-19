#!/usr/bin/env python3

from pdspy.constants.physics import c, m_p, G
from matplotlib.backends.backend_pdf import PdfPages
import pdspy.plotting as plotting
import pdspy.modeling.mpi_pool
import pdspy.interferometry as uv
import pdspy.spectroscopy as sp
import pdspy.modeling as modeling
import pdspy.imaging as im
import pdspy.utils as utils
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
import astropy.stats
import schwimmbad
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
parser.add_argument('-p', '--resetprob', action='store_true')
parser.add_argument('-a', '--action', type=str, default="run")
parser.add_argument('-n', '--ncpus', type=int, default=1)
parser.add_argument('-e', '--withexptaper', action='store_true')
parser.add_argument('-v', '--plot_vis', action='store_true')
parser.add_argument('-c', '--withcontsub', action='store_true')
parser.add_argument('-b', '--trim', type=str, default="")
parser.add_argument('-i', '--nice', action='store_true')
parser.add_argument('-l', '--nicelevel', type=int, default=19)
args = parser.parse_args()

# Check whether we are using MPI.

withmpi = comm.Get_size() > 1

# Set the number of cpus to use.

ncpus = args.ncpus

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

def lnlike(p, visibilities, parameters, plot):
    
    # Set up the params dictionary.

    keys = []
    for key in sorted(parameters.keys()):
        if not parameters[key]["fixed"]:
            keys.append(key)

    params = dict(zip(keys, p))

    # Run the model.

    m = modeling.run_flared_model(visibilities, params, parameters, plot, \
            ncpus=ncpus, source=source, plot_vis=args.plot_vis, nice=nice)

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

    # Return the sum of the chisq.

    return numpy.array(chisq).sum()

# Define a prior function.

def lnprior(p, parameters, priors):
    
    # Set up the params dictionary.

    keys = []
    for key in sorted(parameters.keys()):
        if not parameters[key]["fixed"]:
            keys.append(key)

    params = dict(zip(keys, p))

    # Check the parameter limits.

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

    # Check that the cavity actually falls within the disk.

    if not parameters["logR_cav"]["fixed"]:
        if R_in <= 10.**params["logR_cav"] <= R_disk:
            pass
        else:
            return -numpy.inf

    # Check that the midplane temperature is below the atmosphere temperature.

    if ("logTmid0" in params) or ("logTmid0" in parameters):
        if "logTmid0" in params:
            Tmid0 = 10.**params["logTmid0"]
        else:
            Tmid0 = 10.**parameters["logTmid0"]["value"]

        if "logTatm0" in params:
            Tatm0 = 10.**params["logTatm0"]
        else:
            Tatm0 = 10.**parameters["logTatm0"]["value"]

        if Tmid0 < Tatm0:
            pass
        else:
            return -numpy.inf

    # Everything was correct, so continue on and check the priors.

    lnprior = 0.

    # The prior on parallax (distance).

    if (not parameters["dpc"]["fixed"]) and ("parallax" in priors):
        parallax_mas = 1. / params["dpc"] * 1000

        lnprior += -0.5 * (parallax_mas - priors["parallax"]["value"])**2 / \
                priors["parallax"]["sigma"]**2
    elif (not parameters["dpc"]["fixed"]) and ("dpc" in priors):
        lnprior += -0.5 * (params["dpc"] - priors["dpc"]["value"])**2 / \
                priors["dpc"]["sigma"]**2

    # A prior on stellar mass from the IMF.

    if (not parameters["logM_star"]["fixed"]) and ("Mstar" in priors):
        Mstar = 10.**params["logM_star"]

        if priors["Mstar"]["value"] == "chabrier":
            if Mstar <= 1.:
                lnprior += numpy.log(0.158 * 1./(numpy.log(10.) * Mstar) * \
                        numpy.exp(-(numpy.log10(Mstar) - numpy.log10(0.08))**2/\
                        (2*0.69**2)))
            else:
                lnprior += numpy.log(4.43e-2 * Mstar**-1.3 * \
                        1./(numpy.log(10.) * Mstar))

    return lnprior

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
        pool = schwimmbad.MPIPool()

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

config = utils.load_config()

# Decide whether to use an exponentially tapered 

if args.withexptaper:
    config.parameters["disk_type"]["value"] = "exptaper"

# Decide whether to do continuum subtraction or not.

if args.withcontsub:
    config.parameters["docontsub"]["value"] = True

# Read in the data.

visibilities, images, spectra = utils.load_data(config, model="flared")

################################################################################
#
# Fit the model to the data.
#
################################################################################

# Set up the emcee run.

ndim = 0
keys = []

for key in sorted(config.parameters.keys()):
    if not config.parameters[key]["fixed"]:
        ndim += 1
        keys.append(key)

# Make the labels nice with LaTeX.

labels = ["$"+key.replace("T0_env","T_0,env").replace("T0","T_0").\
        replace("turb_env","turb,env").replace("in_env","in,env").\
        replace("_","_{").replace("log","\log ")+"}$" if key[0:3] == \
        "log" else "$"+key+"$" for key in keys]

# If we are resuming an MCMC simulation, read in the necessary info, otherwise
# set up the info.

if args.resume:
    pos = numpy.load("pos.npy")
    chain = numpy.load("chain.npy")
    state = None
    nsteps = chain[0,0,:,0].size

    if args.resetprob:
        prob = None
        prob_list = numpy.empty((config.ntemps,config.nwalkers,0))
    else:
        prob_list = numpy.load("prob.npy")
        if len(prob_list.shape) == 2:
            prob_list = prob_list.reshape((config.ntemps,config.nwalkers,1))
        prob = prob_list[:,:,-1]
else:
    pos = []
    for i in range(config.ntemps):
        temp_pos = []
        for j in range(config.nwalkers):
            temp_pos.append(utils.propose_point_emcee(config.parameters, \
                    model="flared"))
        pos.append(temp_pos)

    prob = None
    prob_list = numpy.empty((config.ntemps, config.nwalkers, 0))
    chain = numpy.empty((config.ntemps, config.nwalkers, 0, ndim))
    state = None
    nsteps = 0

# Set up the MCMC simulation.

if args.action == "run":
    sampler = emcee.PTSampler(config.ntemps, config.nwalkers, ndim, lnlike, \
            lnprior, loglargs=(visibilities, config.parameters, False), \
            logpargs=(config.parameters, config.priors), pool=pool)

# If we are plotting, make sure that nsteps < max_nsteps.

if args.action == "plot":
    nsteps = config.max_nsteps - 1

# Run a few burner steps.

while nsteps < config.max_nsteps:
    if args.action == "run":
        for i in range(config.steps_per_iter):
            pos, prob, state = sampler.run_mcmc(pos, 1, lnprob0=prob, \
                    rstate0=state)

            chain = numpy.concatenate((chain, sampler.chain), axis=2)

            prob_list = numpy.concatenate((prob_list, prob.\
                    reshape((config.ntemps, config.nwalkers, 1))), axis=2)

            # Plot the steps of the walkers.

            for j in range(ndim):
                nrows, ncols = config.ntemps//5+(config.ntemps%5>0), 5

                fig, ax = plt.subplots(nrows=nrows, ncols=ncols, \
                        figsize=(ncols*2, nrows*2))

                for l in range(config.ntemps):
                    indx, indy = l//5, l%5

                    for k in range(config.nwalkers):
                        ax[indx,indy].plot(chain[l,k,:,j])

                fig.tight_layout()

                fig.savefig("steps_{0:s}.png".format(keys[j]))

                plt.close(fig)

            # Save walker positions in case the code stps running for some 
            # reason.

            numpy.save("pos", pos)
            numpy.save("prob", prob_list)
            numpy.save("chain", chain)

            # Augment the nuber of steps and reset the sampler for the next run.

            nsteps += 1

            sampler.reset()

    # Get the best fit parameters and uncertainties.

    samples = chain[0,:,-config.nplot:,:].reshape((-1, ndim))

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

    params = numpy.median(samples, axis=0)
    sigma = astropy.stats.mad_std(samples, axis=0)

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

    # Create a high resolution model for averaging.

    m = modeling.run_flared_model(visibilities, params, config.parameters, \
            plot=True, ncpus=ncpus, source=source, plot_vis=args.plot_vis, \
            nice=nice)

    # Open up a pdf file to plot into.

    pdf = PdfPages("model.pdf")

    # Loop through the visibilities and plot.

    for j in range(len(visibilities["file"])):
        # Plot the best fit model over the data.

        fig, ax = plt.subplots(nrows=visibilities["nrows"][j], \
                ncols=visibilities["ncols"][j], sharex=True, sharey=True)

        # Make a plot of the channel maps.

        plotting.plot_channel_maps(visibilities, m, config.parameters, params, \
                index=j, plot_vis=args.plot_vis, fig=(fig,ax), image="data", \
                contours="model")
        
        # Adjust the plot and save it.

        fig.set_size_inches((10,9))
        fig.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.07, \
                wspace=0.0,hspace=0.0)

        # Adjust the figure and save.

        pdf.savefig(fig)

        plt.clf()

    # Close the pdf.

    pdf.close()

    # If we're just plotting make sure we don't loop forever.

    if args.action == "plot":
        nsteps = numpy.inf

# Now we can close the pool.

if args.action == "run":
    if withmpi:
        pool.close()
