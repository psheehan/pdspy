#!/usr/bin/env python3

from pdspy.constants.physics import c, m_p, G
from matplotlib.backends.backend_pdf import PdfPages
import pdspy.modeling.mpi_pool
import pdspy.interferometry as uv
import pdspy.spectroscopy as sp
import pdspy.modeling as modeling
import pdspy.imaging as im
import dynesty.plotting as dyplot
import dynesty.results as dyres
import dynesty.utils as dyfunc
import dynesty
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
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

# Define a useful class for plotting.

class Transform:
    def __init__(self, xmin, xmax, dx, fmt):
        self.xmin = xmin
        self.xmax = xmax
        self.dx = dx
        self.fmt = fmt

    def __call__(self, x, p):
        return self.fmt% ((x-(self.xmax-self.xmin+1)/2)*self.dx)

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
        pool = pdspy.modeling.mpi_pool.MPIPool(largedata=False)

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

# Set up the places where we will put all of the data.

visibilities["data"] = []
visibilities["data1d"] = []
visibilities["image"] = []

# Check that all of the parameters are correct.

parameters = modeling.check_parameters(parameters)

# Make sure "fmt" is in the visibilities dictionary.

if not "fmt" in visibilities:
    visibilities["fmt"] = ['4.1f' for i in range(len(visibilities["file"]))]

# Decide whether to use an exponentially tapered 

if args.withexptaper:
    parameters["disk_type"]["value"] = "exptaper"

# Decide whether to do continuum subtraction or not.

if args.withcontsub:
    parameters["docontsub"]["value"] = True

# Define the priors as an empty dictionary, if no priors were specified.

if not 'priors' in globals():
    priors = {}

######################################
# Read in the millimeter visibilities.
######################################

for j in range(len(visibilities["file"])):
    # Read the raw data.

    data = uv.Visibilities()
    data.read(visibilities["file"][j])

    # Center the data. => need to update!

    data = uv.center(data, [visibilities["x0"][j], visibilities["y0"][j], 1.])

    # Add the data to the dictionary structure.

    visibilities["data"].append(data)

    # Scale the weights of the visibilities to force them to be fit well.

    visibilities["data"][j].weights *= visibilities["weight"][j]

    # Average the visibilities radially.

    visibilities["data1d"].append(uv.average(data, gridsize=20, radial=True, \
            log=True, logmin=data.uvdist[numpy.nonzero(data.uvdist)].min()*\
            0.95, logmax=data.uvdist.max()*1.05, mode="spectralline"))

    # Read in the image.

    visibilities["image"].append(im.readimfits(visibilities["image_file"][j]))

################################################################################
#
# Fit the model to the data.
#
################################################################################

# Set up the emcee run.

ndim = 0
periodic = []
keys = []

for key in sorted(parameters.keys()):
    if not parameters[key]["fixed"]:
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
    sampler = pickle.load(open("sampler.p","rb"))

    sampler.rstate = numpy.random
    sampler.pool = pool
    sampler.M = pool.map
    sampler.queue_size = pool.size
else:
    sampler = dynesty.NestedSampler(lnlike, ptform, ndim, nlive=nlive, \
            logl_args=(visibilities, parameters, False), \
            ptform_args=(parameters, priors), periodic=periodic, pool=pool, \
            sample="rwalk", walks=walks)

# Run a few burner steps.

if args.action == "run":
    for it, results in enumerate(sampler.sample(dlogz=dlogz)):
        # Save the state of the sampler (delete the pool first).

        sampler.pool = None
        sampler.M = map
        pickle.dump(sampler, open("sampler.p","wb"))
        sampler.pool = pool
        sampler.M = pool.map

        # Print out the status of the sampler.

        dyres.print_fn(results, sampler.it - 1, sampler.ncall, dlogz=dlogz, \
                logl_max=numpy.inf)

        # Manually calculate the stopping criterion.

        logz_remain = numpy.max(sampler.live_logl) + sampler.saved_logvol[-1]
        delta_logz = numpy.logaddexp(sampler.saved_logz[-1], logz_remain) - \
                sampler.saved_logz[-1]

        # Every 1000 steps stop and make plots of the status.

        if (sampler.it - 1) % 1000 == 0 or delta_logz < 0.05:
            # Add the live points and get the results.

            sampler.add_final_live()

            res = sampler.results

            # Generate a plot of the trace.

            try:
                fig, ax = dyplot.traceplot(res, show_titles=True, \
                        trace_cmap="viridis", connect=True, \
                        connect_highlight=range(5), labels=labels)
            except:
                # If it hasn't converged enough...
                fig, ax = dyplot.traceplot(res, show_titles=True, \
                        trace_cmap="viridis", connect=True, \
                        connect_highlight=range(5), labels=labels, \
                        kde=False)

            fig.savefig("traceplot.png")

            plt.close(fig)

            # Generate a bounds cornerplot.

            fig, ax = dyplot.cornerbound(res, it=res.niter-1, periodic=[5,7], \
                    prior_transform=sampler.prior_transform, show_live=True, \
                    labels=labels)

            fig.savefig("boundplot.png")

            plt.close(fig)

            # If we haven't reached the stopping criteria yet, remove the live 
            # points.

            if delta_logz > 0.05:
                sampler._remove_live_points()

# If we are just plotting, a few minor things to do.

elif args.action == "plot":
    # Add the final live points.

    sampler.add_final_live()

    # Get the results.

    res = sampler.results

    # And make the traceplot one last time.

    try:
        fig, ax = dyplot.traceplot(res, show_titles=True, \
                trace_cmap="viridis", connect=True, \
                connect_highlight=range(5), labels=labels)
    except:
        # If it hasn't converged enough...
        fig, ax = dyplot.traceplot(res, show_titles=True, \
                trace_cmap="viridis", connect=True, \
                connect_highlight=range(5), labels=labels, \
                kde=False)

    fig.savefig("traceplot.png")

    plt.close(fig)

# Generate a plot of the weighted samples.

fig, ax = plt.subplots(11, 11, figsize=(10,10))

dyplot.cornerpoints(res, cmap="plasma", kde=False, fig=(fig,ax), labels=labels)

fig.savefig("cornerpoints.png")

# Generate a corner plot from Dynesty.

fig, ax = plt.subplots(12, 12, figsize=(15,15))

dyplot.cornerplot(res, color="blue", show_titles=True, max_n_ticks=3, \
        quantiles=None, fig=(fig, ax), labels=labels)

fig.savefig("cornerplot.png")

# Convert the results to a more traditional set of samples that you would
# get from an MCMC program.

samples, weights = res.samples, numpy.exp(res.logwt - res.logz[-1])

samples = dyfunc.resample_equal(samples, weights)

# Save pos, prob, chain.

numpy.save("samples.npy")

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

keys = []
for key in sorted(parameters.keys()):
    if not parameters[key]["fixed"]:
        keys.append(key)

params = dict(zip(keys, params))

############################################################################
#
# Plot the results.
#
############################################################################

# Create a high resolution model for averaging.

m = modeling.run_flared_model(visibilities, params, parameters, plot=True, \
        ncpus=ncpus, source=source, plot_vis=args.plot_vis, nice=nice)

# Open up a pdf file to plot into.

pdf = PdfPages("model.pdf")

# Loop through the visibilities and plot.

for j in range(len(visibilities["file"])):
    # Plot the best fit model over the data.

    fig, ax = plt.subplots(nrows=visibilities["nrows"][j], \
            ncols=visibilities["ncols"][j], sharex=True, sharey=True)

    # Calculate the velocity for each image.

    if args.plot_vis:
        v = c * (float(visibilities["freq"][j])*1.0e9 - \
                visibilities["data"][j].freq)/ \
                (float(visibilities["freq"][j])*1.0e9)
    else:
        v = c * (float(visibilities["freq"][j])*1.0e9 - \
                visibilities["image"][j].freq)/ \
                (float(visibilities["freq"][j])*1.0e9)

    # Plot the image.

    vmin = numpy.nanmin(visibilities["image"][j].image)
    vmax = numpy.nanmax(visibilities["image"][j].image)

    for k in range(visibilities["nrows"][j]):
        for l in range(visibilities["ncols"][j]):
            ind = k*visibilities["ncols"][j] + l + visibilities["ind0"][j]

            # Turn off the axis if ind >= nchannels

            if ind >= v.size:
                ax[k,l].set_axis_off()
                continue

            # Get the centroid position.

            ticks = visibilities["image_ticks"][j]

            if "x0" in params:
                xmin, xmax = int(round(visibilities["image_npix"][j]/2 + \
                        visibilities["x0"][j]/\
                        visibilities["image_pixelsize"][j]+ \
                        params["x0"]/visibilities["image_pixelsize"][j]+ \
                        ticks[0]/visibilities["image_pixelsize"][j])), \
                        int(round(visibilities["image_npix"][j]/2+\
                        visibilities["x0"][j]/\
                        visibilities["image_pixelsize"][j]+ \
                        params["x0"]/visibilities["image_pixelsize"][j]+ \
                        ticks[-1]/visibilities["image_pixelsize"][j]))
            else:
                xmin, xmax = int(round(visibilities["image_npix"][j]/2 + \
                        visibilities["x0"][j]/\
                        visibilities["image_pixelsize"][j]+ \
                        parameters["x0"]["value"]/\
                        visibilities["image_pixelsize"][j]+ \
                        ticks[0]/visibilities["image_pixelsize"][j])), \
                        int(round(visibilities["image_npix"][j]/2+\
                        visibilities["x0"][j]/\
                        visibilities["image_pixelsize"][j]+ \
                        parameters["x0"]["value"]/\
                        visibilities["image_pixelsize"][j]+ \
                        ticks[-1]/visibilities["image_pixelsize"][j]))
            if "y0" in params:
                ymin, ymax = int(round(visibilities["image_npix"][j]/2-\
                        visibilities["y0"][j]/\
                        visibilities["image_pixelsize"][j]- \
                        params["y0"]/visibilities["image_pixelsize"][j]+ \
                        ticks[0]/visibilities["image_pixelsize"][j])), \
                        int(round(visibilities["image_npix"][j]/2-\
                        visibilities["y0"][j]/\
                        visibilities["image_pixelsize"][j]- \
                        params["y0"]/visibilities["image_pixelsize"][j]+ \
                        ticks[-1]/visibilities["image_pixelsize"][j]))
            else:
                ymin, ymax = int(round(visibilities["image_npix"][j]/2-\
                        visibilities["y0"][j]/\
                        visibilities["image_pixelsize"][j]- \
                        parameters["y0"]["value"]/\
                        visibilities["image_pixelsize"][j]+ \
                        ticks[0]/visibilities["image_pixelsize"][j])), \
                        int(round(visibilities["image_npix"][j]/2-\
                        visibilities["y0"][j]/\
                        visibilities["image_pixelsize"][j]- \
                        parameters["y0"]["value"]/\
                        visibilities["image_pixelsize"][j]+ \
                        ticks[-1]/visibilities["image_pixelsize"][j]))

            # Plot the image.

            if args.plot_vis:
                ax[k,l].errorbar(visibilities["data1d"][j].uvdist, \
                        visibilities["data1d"][j].amp[:,ind], yerr=1./\
                        visibilities["data1d"][j].weights[:,ind]**0.5, \
                        fmt="bo")
            else:
                ax[k,l].imshow(visibilities["image"][j].image[ymin:ymax,\
                        xmin:xmax,ind,0], origin="lower", \
                        interpolation="nearest", vmin=vmin, vmax=vmax)

            # Now make the centroid the map center for the model.

            xmin, xmax = int(round(visibilities["image_npix"][j]/2+1 + \
                    ticks[0]/visibilities["image_pixelsize"][j])), \
                    int(round(visibilities["image_npix"][j]/2+1 +\
                    ticks[-1]/visibilities["image_pixelsize"][j]))
            ymin, ymax = int(round(visibilities["image_npix"][j]/2+1 + \
                    ticks[0]/visibilities["image_pixelsize"][j])), \
                    int(round(visibilities["image_npix"][j]/2+1 + \
                    ticks[-1]/visibilities["image_pixelsize"][j]))

            # Plot the model image.

            if args.plot_vis:
                ax[k,l].plot(m.visibilities[visibilities["lam"][j]].uvdist,\
                        m.visibilities[visibilities["lam"][j]].amp[:,ind], \
                        "g-")
            else:
                levels = numpy.array([0.05, 0.25, 0.45, 0.65, 0.85, 0.95])*\
                        m.images[visibilities["lam"][j]].image.max()

                ax[k,l].contour(m.images[visibilities["lam"][j]].\
                        image[ymin:ymax,xmin:xmax,ind,0], levels=levels)

            # Add the velocity to the map.

            txt = ax[k,l].annotate(r"$v=%{0:s}$ km s$^{{-1}}$".format(\
                    visibilities["fmt"][j]) % (v[ind]/1e5),\
                    xy=(0.1,0.8), xycoords='axes fraction')

            #txt.set_path_effects([PathEffects.withStroke(linewidth=2, \
            #        foreground='w')])

            # Fix the axes labels.

            if args.plot_vis:
                ax[-1,l].set_xlabel("U-V Distance [k$\lambda$]")
            else:
                transform = ticker.FuncFormatter(Transform(xmin, xmax, \
                        visibilities["image_pixelsize"][j], '%.1f"'))

                ax[k,l].set_xticks(visibilities["image_npix"][j]/2+\
                        ticks[1:-1]/visibilities["image_pixelsize"][j]-xmin)
                ax[k,l].set_yticks(visibilities["image_npix"][j]/2+\
                        ticks[1:-1]/visibilities["image_pixelsize"][j]-ymin)

                ax[k,l].get_xaxis().set_major_formatter(transform)
                ax[k,l].get_yaxis().set_major_formatter(transform)

                ax[-1,l].set_xlabel("$\Delta$RA")

                # Show the size of the beam.

                bmaj = visibilities["image"][j].header["BMAJ"] / \
                        abs(visibilities["image"][j].header["CDELT1"])
                bmin = visibilities["image"][j].header["BMIN"] / \
                        abs(visibilities["image"][j].header["CDELT1"])
                bpa = visibilities["image"][j].header["BPA"]

                ax[k,l].add_artist(patches.Ellipse(xy=(12.5,17.5), \
                        width=bmaj, height=bmin, angle=(bpa+90), \
                        facecolor="white", edgecolor="black"))

                ax[k,l].set_adjustable('box')

        if args.plot_vis:
            ax[k,0].set_ylabel("Amplitude [Jy]")

            ax[k,l].set_xscale("log", nonposx='clip')
        else:
            ax[k,0].set_ylabel("$\Delta$Dec")

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
