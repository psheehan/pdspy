#!/usr/bin/env python3

from pdspy.utils.dynesty import lnlike, ptform
from matplotlib.backends.backend_pdf import PdfPages
import pdspy.plotting as plotting
import pdspy.modeling as modeling
import pdspy.utils as utils
import dynesty.plotting as dyplot
import dynesty.results as dyres
import dynesty.utils as dyfunc
import dynesty
import matplotlib.pyplot as plt
import astropy.stats
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
parser.add_argument('-p', '--resetprob', action='store_true')
parser.add_argument('-a', '--action', type=str, default="run")
parser.add_argument('-n', '--ncpus', type=int, default=1)
parser.add_argument('-e', '--withexptaper', action='store_true')
parser.add_argument('-v', '--plot_vis', action='store_true')
parser.add_argument('-c', '--withcontsub', action='store_true')
parser.add_argument('-b', '--trim', type=str, default="")
parser.add_argument('-i', '--nice', action='store_true')
parser.add_argument('-l', '--nicelevel', type=int, default=19)
parser.add_argument('-f', '--ftcode', type=str, default="galario")
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
periodic = []
keys = []

for key in sorted(config.parameters.keys()):
    if not config.parameters[key]["fixed"]:
        ndim += 1
        keys.append(key)

        if key == "pa":
            periodic.append(ndim-1)

# Make the labels nice with LaTeX.

labels = ["$"+key.replace("T0_env","T_0,env").replace("T0","T_0").\
        replace("Tatm0","T_atm,0").replace("Tmid0","T_mid,0").\
        replace("turb_env","turb,env").replace("in_env","in,env").\
        replace("_","_{").replace("log","\log ")+"}$" if key[0:3] == \
        "log" else "$"+key+"$" for key in keys]

# Set up the MCMC simulation.

if args.resume:
    sampler = utils.dynesty.load_sampler("sampler.p", pool=pool, dynamic=True)

    # For backwards compatibility
    sampler.loglikelihood.args = [visibilities, images, spectra, \
            config.parameters, False]
    sampler.loglikelihood.kwargs = {"model":"flared", \
            "ncpus":ncpus, "source":source, "nice":nice, "ftcode":args.ftcode}
    sampler.prior_transform.kwargs = {"model":"flared"}

    res = sampler.results
else:
    sampler = dynesty.DynamicNestedSampler(utils.dynesty.lnlike, \
            utils.dynesty.ptform, ndim, logl_args=(visibilities, images, \
            spectra, config.parameters, False), logl_kwargs={"model":"flared", \
            "ncpus":ncpus, "source":source, "nice":nice,
            "ftcode":args.ftcode}, ptform_args=(\
            config.parameters, config.priors), ptform_kwargs={"model":\
            "flared"}, periodic=periodic, pool=pool, sample="rwalk", \
            walks=config.walks)

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

# Now calculate the best fit parameters.

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
        nice=nice, ftcode=args.ftcode)

# Open up a pdf file to plot into.

pdf = PdfPages("model.pdf")

# Loop through the visibilities and plot.

for j in range(len(visibilities["file"])):
    # Plot the best fit model over the data.

    fig, ax = plt.subplots(nrows=visibilities["nrows"][j], \
            ncols=visibilities["ncols"][j], sharex=True, sharey=True)

    # Make a plot of the channel maps.

    plotting.plot_channel_maps(visibilities, m, config.parameters, params, \
            index=j, plot_vis=args.plot_vis, fig=(fig,ax))
    
    # Adjust the plot and save it.

    fig.set_size_inches((10,9))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.07, \
            wspace=0.0,hspace=0.0)

    # Adjust the figure and save.

    pdf.savefig(fig)

    plt.clf()

# Close the pdf.

pdf.close()

# Now we can close the pool.

if args.action == "run":
    if withmpi:
        pool.close()
