#!/usr/bin/env python3

from matplotlib.backends.backend_pdf import PdfPages
import pdspy.plotting as plotting
import pdspy.modeling.mpi_pool
import pdspy.modeling as modeling
import pdspy.utils as utils
import matplotlib.pyplot as plt
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
        replace("Tatm0","T_atm,0").replace("Tmid0","T_mid,0").\
        replace("turb_env","turb,env").replace("in_env","in,env").\
        replace("_","_{").replace("log","\log ")+"}$" if key[0:3] == \
        "log" else "$"+key+"$" for key in keys]

# If we are resuming an MCMC simulation, read in the necessary info, otherwise
# set up the info.

if args.resume:
    pos = numpy.load("pos.npy")
    chain = numpy.load("chain.npy")
    state = None
    nsteps = chain[0,:,0].size

    if args.resetprob:
        prob = None
        prob_list = numpy.empty((config.nwalkers,0))
    else:
        prob_list = numpy.load("prob.npy")
        if len(prob_list.shape) == 1:
            prob_list = prob_list.reshape((config.nwalkers,1))
        prob = prob_list[:,-1]
else:
    pos = []
    for j in range(config.nwalkers):
        pos.append(utils.propose_point_emcee(config.parameters, model="flared"))

    prob = None
    prob_list = numpy.empty((config.nwalkers, 0))
    chain = numpy.empty((config.nwalkers, 0, ndim))
    state = None
    nsteps = 0

# Set up the MCMC simulation.

if args.action == "run":
    sampler = emcee.EnsembleSampler(config.nwalkers, ndim, utils.emcee.lnprob, \
            args=(visibilities, images, spectra, config.parameters, \
            config.priors, False), kwargs={"model":"flared", "ncpus":ncpus, \
            "timelimit":3600, "ncpus_highmass":ncpus, \
            "with_hyperion":False, "source":source, "nice":nice, \
            "verbose":False}, pool=pool)

# If we are plotting, make sure that nsteps < max_nsteps.

if args.action == "plot":
    nsteps = config.max_nsteps - 1

# Run a few burner steps.

while nsteps < config.max_nsteps:
    if args.action == "run":
        for i in range(config.steps_per_iter):
            pos, prob, state = sampler.run_mcmc(pos, 1, lnprob0=prob, \
                    rstate0=state)

            chain = numpy.concatenate((chain, sampler.chain), axis=1)

            prob_list = numpy.concatenate((prob_list, prob.\
                    reshape((config.nwalkers,1))), axis=1)

            # Plot the steps of the walkers.

            for j in range(ndim):
                fig, ax = plt.subplots(nrows=1, ncols=1)

                for k in range(config.nwalkers):
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

    # Get the best fit parameters and uncertainties.

    samples = chain[:,-config.nplot:,:].reshape((-1, ndim))

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

    plt.close(fig)

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
