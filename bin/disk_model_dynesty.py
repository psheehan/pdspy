#!/usr/bin/env python3

import pdspy.modeling as modeling
import pdspy.plotting as plotting
import pdspy.utils as utils
import dynesty.plotting as dyplot
import dynesty.results as dyres
import dynesty.utils as dyfunc
import dynesty
import matplotlib.pyplot as plt
import schwimmbad
import argparse
import numpy
import sys
import os
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
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-f', '--ftcode', type=str, default="galario")
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

# Deprecated option to include an expontial taper on the disk.

if args.withexptaper:
    config.parameters["disk_type"]["value"] = "exptaper"

# Read in the data.

visibilities, images, spectra = utils.load_data(config)

################################################################################
#
# Fit the model to the data.
#
################################################################################

# Set up the emcee run.

ndim = 0
periodic = []
keys = []

for key in sorted(config.parameters.keys()):
    if not config.parameters[key]["fixed"]:
        ndim += 1
        keys.append(key)

        if key == "pa":
            periodic.append(ndim-1)

# Make the labels nice with LaTeX.

labels = ["$"+key.replace("_","_{").replace("log","\log ")+"}$" \
        if key[0:3] == "log" else "$"+key.replace("h_large","h,large")+\
        "$" for key in keys]

# If we are resuming an MCMC simulation, read in the necessary info, otherwise
# set up the info.

if args.resume:
    sampler = utils.dynesty.load_sampler("sampler.p", pool=pool, dynamic=True)

    res = sampler.results
else:
    sampler = dynesty.DynamicNestedSampler(utils.dynesty.lnlike, \
            utils.dynesty.ptform, ndim, logl_args=(visibilities, images, \
            spectra, config.parameters, False), logl_kwargs={"model":"disk", \
            "ncpus":ncpus, "timelimit":args.timelimit, \
            "ncpus_highmass":ncpus_highmass, "with_hyperion":args.withhyperion,\
            "source":source, "nice":nice, "verbose":args.verbose, \
            "ftcode":args.ftcode}, \
            ptform_args=(config.parameters, config.priors), periodic=periodic, \
            pool=pool, sample="rwalk", walks=config.walks)

# Run a few burner steps.

if args.action == "run":
    if not sampler.base:
        for it, results in enumerate(sampler.sample_initial(dlogz=config.dlogz,\
                nlive=config.nlive_init, save_samples=True, \
                resume=args.resume)):
            # Save the state of the sampler (delete the pool first).

            utils.dynesty.save_sampler("sampler.p", sampler, pool=pool, \
                    dynamic=True)

            # Print out the status of the sampler.

            dyres.print_fn(results, sampler.it - 1, sampler.ncall, \
                    dlogz=config.dlogz, logl_max=numpy.inf)

            # Manually calculate the stopping criterion.

            logz_remain = numpy.max(sampler.sampler.live_logl) + \
                    sampler.sampler.saved_logvol[-1]
            delta_logz = numpy.logaddexp(sampler.sampler.saved_logz[-1], \
                    logz_remain) - sampler.sampler.saved_logz[-1]

            # Every 1000 steps stop and make plots of the status.

            if (sampler.it - 1) % 1000 == 0 and delta_logz >= config.dlogz:
                # Add the live points and get the results.

                sampler.sampler.add_final_live()

                res = sampler.sampler.results

                # Make plots of the current status of the fit.

                utils.dynesty.plot_status(res, ptform=sampler.prior_transform, \
                        labels=labels, periodic=periodic)

                # If we haven't reached the stopping criteria yet, remove the
                # live points.

                sampler.sampler._remove_live_points()

        # Gather the results and make one final plot of the status.

        res = sampler.results

        utils.dynesty.plot_status(res, ptform=sampler.prior_transform, \
                labels=labels, periodic=periodic)

    for i in range(sampler.batch, config.maxbatch):
        # Get the correct bounds to use for the batch.

        logl_bounds = dynesty.dynamicsampler.weight_function(sampler.results)
        lnz, lnzerr = sampler.results.logz[-1], sampler.results.logzerr[-1]

        # Sample the batch.

        for results in sampler.sample_batch(logl_bounds=logl_bounds, \
                nlive_new=config.nlive_batch):
            # Print out the results.

            (worst, ustar, vstar, loglstar, nc,
                     worst_it, boundidx, bounditer, eff) = results

            results = (worst, ustar, vstar, loglstar, numpy.nan, numpy.nan,
                    lnz, lnzerr**2, numpy.nan, nc, worst_it, boundidx,
                    bounditer, eff, numpy.nan)

            dyres.print_fn(results, sampler.it - 1, sampler.ncall,\
                    nbatch=sampler.batch+1, stop_val=5, \
                    logl_min=logl_bounds[0], logl_max=logl_bounds[1])

        # Merge the new samples in.

        sampler.combine_runs()

        # Save the status of the sampler after each batch.

        utils.dynesty.save_sampler("sampler.p", sampler, pool=pool, \
                dynamic=True)

        # Get the results.

        res = sampler.results

        # Make plots of the current status of the fit.

        utils.dynesty.plot_status(res, ptform=sampler.prior_transform, \
                labels=labels, periodic=periodic)

# If we are just plotting, a few minor things to do.

elif args.action == "plot":
    # Add the final live points if needed and get the results.

    if not sampler.base:
        if not sampler.sampler.added_live:
            sampler.sampler.add_final_live()

        res = sampler.sampler.results
    else:
        res = sampler.results

    # Make the traceplots and the bound plots.

    utils.dynesty.plot_status(res, ptform=sampler.prior_transform, \
            labels=labels, periodic=periodic)

# Generate a plot of the weighted samples.

fig, ax = plt.subplots(ndim-1, ndim-1, figsize=(10,10))

dyplot.cornerpoints(res, cmap="plasma", kde=False, fig=(fig,ax), labels=labels)

fig.savefig("cornerpoints.png")

# Generate a corner plot from Dynesty.

fig, ax = plt.subplots(ndim, ndim, figsize=(15,15))

dyplot.cornerplot(res, color="blue", show_titles=True, max_n_ticks=3, \
        quantiles=None, fig=(fig, ax), labels=labels)

fig.savefig("cornerplot.png")

# Convert the results to a more traditional set of samples that you would
# get from an MCMC program.

samples, weights = res.samples, numpy.exp(res.logwt - res.logz[-1])

samples = dyfunc.resample_equal(samples, weights)

# Save pos, prob, chain.

numpy.save("samples.npy", samples)

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
        config.parameters, plot=True, ncpus=ncpus, \
        ncpus_highmass=ncpus_highmass, with_hyperion=args.withhyperion, \
        timelimit=args.timelimit, source=source, nice=nice, \
        ftcode=args.ftcode)

# Plot the millimeter data/models.

for j in range(len(visibilities["file"])):
    # Plot the visibilities.

    plotting.plot_1D_visibilities(visibilities, m, config.parameters, \
            params, index=j, fig=(fig, ax[2*j,0]))

    # Plot the 2D visibilities.

    plotting.plot_2D_visibilities(visibilities, m, config.parameters, \
            params, index=j, fig=(fig, ax[2*j+1,0:2]))

    # Plot the model image.

    plotting.plot_continuum_image(visibilities, m, config.parameters, \
            params, index=j, fig=(fig, ax[2*j,1]))

# Plot the SED.

plotting.plot_SED(spectra, m, SED=args.SED, fig=(fig, ax[0,2]))

# Plot the scattered light image.

for j in range(len(images["file"])):
    plotting.plot_scattered_light(visibilities, m, config.parameters, \
            params, index=j, fig=(fig, ax[1,2]))

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
