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
import dynesty.plotting as dyplot
import dynesty.results as dyres
import dynesty.utils as dyfunc
import dynesty
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
import schwimmbad
import scipy.stats
import scipy.integrate
import argparse
import pickle
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

def ptform(u, parameters, priors):

    # Set up the params dictionary.

    keys = []
    for key in sorted(parameters.keys()):
        if not parameters[key]["fixed"]:
            keys.append(key)

    uparams = dict(zip(keys, u))
    pparams = dict(zip(keys, [0 for i in range(len(u))]))

    # Get the correct order for setting parameters (as some depend on others

    ordered_keys, index = numpy.unique(["logR_env","logR_disk","logR_in", \
            "logTmid0"]+list(parameters.keys()), return_index=True)
    ordered_keys = ordered_keys[numpy.argsort(index)]

    # Now loop through the parameters and transform the ones that aren't fixed.

    for key in ordered_keys:
        if not parameters[key]["fixed"]:
            # R_disk has to be smaller than R_env.
            if key == "logR_disk":
                if "logR_env" in pparams:
                    logR_env = pparams["logR_env"]
                else:
                    logR_env = parameters["logR_env"]["value"]

                pparams[key] = uparams[key] * (min(logR_env, \
                        parameters[key]["limits"][1]) - \
                        parameters[key]["limits"][0]) + \
                        parameters[key]["limits"][0]
            # R_in has to be smaller than R_disk.
            elif key == "logR_in":
                if "logR_disk" in pparams:
                    logR_disk = pparams["logR_disk"]
                else:
                    logR_disk = parameters["logR_disk"]["value"]

                pparams[key] = uparams[key] * (min(logR_disk, \
                        parameters[key]["limits"][1]) - \
                        parameters[key]["limits"][0]) + \
                        parameters[key]["limits"][0]
            # R_cav should be between R_in and R_disk.
            elif key == "logR_cav":
                if "logR_disk" in pparams:
                    logR_disk = pparams["logR_disk"]
                else:
                    logR_disk = parameters["logR_disk"]["value"]

                if "logR_in" in pparams:
                    logR_in = pparams["logR_in"]
                else:
                    logR_in = parameters["logR_in"]["value"]


                pparams[key] = uparams[key] * (min(pparams["logR_disk"], \
                        parameters[key]["limits"][1]) - \
                        max(pparams["logR_in"],parameters[key]["limits"][0])) +\
                        max(pparams["logR_in"],parameters[key]["limits"][0])
            # Tmid0 should be less than Tatm0.
            elif key == "logTmid0":
                pparams[key] = uparams[key] * (min(pparams["logTatm0"], \
                        parameters[key]["limits"][1]) - \
                        parameters[key]["limits"][0]) + \
                        parameters[key]["limits"][0]
            # Make the position angle cyclic.
            elif key == "pa":
                pparams[key] = (uparams[key] % 1.) * 360.
            # If we have a prior on the parallax, use that to get dpc.
            elif key == "dpc" and "parallax" in priors:
                m = priors["parallax"]["value"]
                s = priors["parallax"]["sigma"]
                low = 1. / parameters["dpc"]["limits"][1] * 1000
                high = 1. / parameters["dpc"]["limits"][0] * 1000
                low_n, high_n = (low - m) /s, (high - m) /s

                parallax = scipy.stats.truncnorm.ppf(uparams[key], low_n, \
                        high_n, loc=m, scale=s)

                pparams[key] = 1./parallax * 1000
            # If we have a prior on the stellar mass from the IMF.
            elif key == "logM_star" and "Mstar" in priors:
                imf = imf_gen(a=10.**parameters[key]["limits"][0], \
                        b=10.**parameters[key]["limits"][1])

                pparams[key] = numpy.log10(imf.ppf(uparams[key]))
            # If we have a prior on a parameter, draw the parameter from a 
            # normal distribution.
            elif key in priors:
                m = priors[key]["value"]
                s = priors[key]["sigma"]
                low = parameters[key]["limits"][0]
                high = parameters[key]["limits"][1]
                low_n, high_n = (low - m) /s, (high - m) /s

                pparams[key] = scipy.stats.truncnorm.ppf(uparams[key], low_n, \
                        high_n, loc=m, scale=s)
            # If none of the above apply, then draw from a uniform prior between
            # the provided limits.
            else:
                pparams[key] = uparams[key] * (parameters[key]["limits"][1] - \
                        parameters[key]["limits"][0]) + \
                        parameters[key]["limits"][0]

    # Now get the list of parameter values and return.

    p = [pparams[key] for key in sorted(keys)]

    return p

# Define a few useful classes for generating samples from the IMF.

def chabrier_imf(Mstar):
    if Mstar <= 1.:
        return 0.158 * 1./(numpy.log(10.) * Mstar) * \
                numpy.exp(-(numpy.log10(Mstar)-numpy.log10(0.08))**2/ \
                (2*0.69**2))
    else:
        return 4.43e-2 * Mstar**-1.3 * 1./(numpy.log(10.) * Mstar)

class imf_gen(scipy.stats.rv_continuous):
    def __init__(self, a=None, b=None):
        self.norm = scipy.integrate.quad(chabrier_imf, a, b)[0]

        super().__init__(a=a, b=b)

    def _pdf(self, x):
        return chabrier_imf(x) / self.norm

# Functions for saving the state of the Dynesty Sampler and loading a saved 
# state.

def save_sampler(name, sampler, pool=None):

    # Clear the random state, as it cannot be pickled.
    sampler.rstate = None

    # Clear the MPI pool, as it also cannot be pickled.
    sampler.pool = None
    sampler.M = map

    # Save
    pickle.dump(sampler, open(name, "wb"))

    # Restore everything to the way it was before.
    sampler.rstate = numpy.random
    sampler.pool = pool
    if pool != None:
        sampler.M = pool.map
    else:
        sampler.M = map

def load_sampler(name, pool=None):
    # Load the sampler from the pickle file.
    sampler = pickle.load(open("sampler.p","rb"))

    # Add back in the random state.
    sampler.rstate = numpy.random

    # Add the pool correctly.
    sampler.pool = pool
    if pool != None:
        sampler.M = pool.map
        sampler.queue_size = pool.size
    else:
        sampler.M = map

    return sampler

# A function to make useful plots as the sampling is running.

def plot_status(res, labels=None, periodic=None):
    # Generate a plot of the trace.

    try:
        fig, ax = dyplot.traceplot(res, show_titles=True, trace_cmap="viridis",\
                connect=True, connect_highlight=range(5), labels=labels)
    except:
        # If it hasn't converged enough...
        fig, ax = dyplot.traceplot(res, show_titles=True, trace_cmap="viridis",\
                connect=True, connect_highlight=range(5), labels=labels, \
                kde=False)

    fig.savefig("traceplot.png")

    plt.close(fig)

    # Generate a bounds cornerplot.

    fig, ax = dyplot.cornerbound(res, it=res.niter-1, periodic=periodic, \
            prior_transform=sampler.prior_transform, show_live=True, \
            labels=labels)

    fig.savefig("boundplot.png")

    plt.close(fig)

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
        replace("turb_env","turb,env").replace("in_env","in,env").\
        replace("_","_{").replace("log","\log ")+"}$" if key[0:3] == \
        "log" else "$"+key+"$" for key in keys]

# Set up the MCMC simulation.

if args.resume:
    sampler = load_sampler("sampler.p", pool=pool)

    res = sampler.results
else:
    sampler = dynesty.NestedSampler(lnlike, ptform, ndim, \
            nlive=config.nlive_init, \
            logl_args=(visibilities, config.parameters, False), \
            ptform_args=(config.parameters, config.priors), periodic=periodic, \
            pool=pool, sample="rwalk", walks=config.walks)

# Run a few burner steps.

if args.action == "run":
    for it, results in enumerate(sampler.sample_initial(dlogz=\
            config.dlogz)):
        # Save the state of the sampler (delete the pool first).

        save_sampler("sampler.p", sampler, pool=pool)

        # Print out the status of the sampler.

        dyres.print_fn(results, sampler.it - 1, sampler.ncall, \
                dlogz=config.dlogz, logl_max=numpy.inf)

        # Manually calculate the stopping criterion.

        logz_remain = numpy.max(sampler.live_logl) + sampler.saved_logvol[-1]
        delta_logz = numpy.logaddexp(sampler.saved_logz[-1], logz_remain) - \
                sampler.saved_logz[-1]

        # Every 1000 steps stop and make plots of the status.

        if (sampler.it - 1) % 1000 == 0 and delta_logz >= config.dlogz:
            # Add the live points and get the results.

            sampler.add_final_live()

            res = sampler.results

            # Make plots of the current status of the fit.

            plot_status(res, labels=labels, periodic=periodic)

            # If we haven't reached the stopping criteria yet, remove the 
            # live points.

            sampler._remove_live_points()

    # Gather the results and make one final plot of the status.

    res = sampler.results

    plot_status(res, labels=labels, periodic=periodic)

# If we are just plotting, a few minor things to do.

elif args.action == "plot":
    # Add the final live points if needed and get the results.

    if not sampler.added_live:
        sampler.add_final_live()

        res = sampler.sampler.results
    else:
        res = sampler.results

    # Make the traceplots and the bound plots.

    plot_status(res, labels=labels, periodic=periodic)

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
