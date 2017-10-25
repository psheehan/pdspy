#!/usr/bin/env python3

from pdspy.constants.physics import c, m_p, G
from pdspy.constants.physics import k as k_b
from pdspy.constants.astronomy import M_sun, AU
import pdspy.modeling.mpi_pool
import pdspy.interferometry as uv
import pdspy.spectroscopy as sp
import pdspy.modeling as modeling
import pdspy.imaging as im
import pdspy.misc as misc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
import pdspy.table
import pdspy.dust as dust
import pdspy.gas as gas
import pdspy.mcmc as mc
import scipy.signal
import argparse
import numpy
import time
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
args = parser.parse_args()

source = args.object

if source == None:
    print("--object must be specified")
    sys.exit()

if args.action not in ['run','plot']:
    print("Please select a valid action")
    sys.exit(0)

if args.action == 'plot':
    args.resume = True

################################################################################
#
# Create a function which returns a model of the data.
#
################################################################################

def model(visibilities, params, parameters, plot=False):
    # Set the values of all of the parameters.

    p = {}
    for key in parameters:
        if parameters[key]["fixed"]:
            if isinstance(parameters[key]["value"], str):
                if parameters[parameters[key]["value"]]["fixed"]:
                    value = parameters[parameters[key]["value"]]["value"]
                else:
                    value = params[parameters[key]["value"]]
            else:
                value = parameters[key]["value"]
        else:
            value = params[key]

        if key[0:3] == "log":
            p[key[3:]] = 10.**value
        else:
            p[key] = value

    # Add these to parameters:

    t0 = 10.**params[3]
    q = params[4]
    a_turb = 10.**params[5]
    v_sys = params[6]

    # Make sure alpha and beta are defined.

    t_rdisk = p["T0"] * (p["R_disk"] / 1.)**-p["q"]
    p["h_0"] = ((k_b*(p["R_disk"]*AU)**3*t_rdisk) / (G*p["M_star"]*M_sun * \
            2.37*m_p))**0.5 / AU
    p["beta"] = 0.5 * (3 - p["q"])
    p["alpha"] = p["gamma"] + p["beta"]

    # Shift the wavelengths by the velocities.

    b = p["v_sys"]*1.0e5 / c
    lam = c / data.freq / 1.0e-4
    wave = lam * numpy.sqrt((1. - b) / (1. + b))

    # Set up the dust.

    dustopac = "draine_3mm.hdf5"

    ddust = dust.Dust()
    ddust.set_properties_from_file(os.environ["HOME"]+\
            "/Documents/Projects/DiskMasses/Modeling/Dust/"+dustopac)

    # Set up the gas.

    gases = []
    abundance = []

    co = gas.Gas()
    co.set_properties_from_lambda(os.environ["HOME"]+\
            "/Documents/Projects/DiskMasses/Modeling/Gas/co.dat")
    gases.append(co)
    abundance.append(1.5e-4)

    # Make sure we are in a temp directory to not overwrite anything.

    original_dir = os.environ["PWD"]
    os.mkdir("/tmp/temp_ROXs12_{0:d}".format(comm.Get_rank()))
    os.chdir("/tmp/temp_ROXs12_{0:d}".format(comm.Get_rank()))

    # Write the parameters to a text file so it is easy to keep track of them.

    f = open("params.txt","w")
    for i in range(len(params)):
        f.write("params[{0:d}] = {1:f}\n".format(i, params[i]))
    f.close()

    # Set up the model. 

    m = modeling.YSOModel()
    m.add_star(mass=p["M_star"], luminosity=p["L_star"],temperature=p["T_star"])
    m.set_spherical_grid(p["R_in"], max(5*p["R_disk"],300), 100, 51, 2, \
            code="radmc3d")
    m.add_pringle_disk(mass=p["M_disk"], rmin=p["R_in"], rmax=p["R_disk"], \
            plrho=p["alpha"], h0=p["h_0"], plh=p["beta"], dust=ddust, \
            t0=p["T0"], plt=p["q"], gas=gases, abundance=abundance,\
            aturb=p["a_turb"])
    m.grid.set_wavelength_grid(0.1,1.0e5,500,log=True)

    # Run the images/visibilities/SEDs.

    for j in range(len(visibilities["file"])):
        m.set_camera_wavelength(wave)

        m.run_visibilities(name=visibilities["lam"][j], nphot=1e5, \
                npix=visibilities["npix"][j], lam=None, \
                pixelsize=visibilities["pixelsize"][j], tgas_eq_tdust=True, \
                scattering_mode_max=0, incl_dust=False, incl_lines=True, \
                loadlambda=True, incl=p["i"], pa=p["pa"], dpc=p["dpc"], \
                code="radmc3d", verbose=False)

        m.visibilities[visibilities["lam"][j]] = uv.center(\
                m.visibilities[visibilities["lam"][j]], [p["x0"], p["y0"], 1])

        m.visibilities[visibilities["lam"][j]] = uv.interpolate_model(\
                visibilities["lam"][j].u, visibilities["lam"][j].v, \
                visibilities["lam"][j].freq, \
                m.visibilities[visibilities["lam"][j]])

        if plot:
            m.run_image(name=visibilities["lam"][j], nphot=1e5, \
                    npix=visibilities["image_npix"][j], lam=None, \
                    pixelsize=visibilities["image_pixelsize"][j], \
                    tgas_eq_tdust=True, scattering_mode_max=0, \
                    incl_dust=False, incl_lines=True, loadlambda=True, \
                    incl=p["i"], pa=p["pa"], dpc=p["dpc"], code="radmc3d", \
                    verbose=False)

            x, y = numpy.meshgrid(numpy.linspace(-256,255,512), \
                    numpy.linspace(-256,255,512))

            beam = misc.gaussian2d(x, y, 0., 0., image.header["BMAJ"]/2.355/\
                    image.header["CDELT2"], image.header["BMIN"]/2.355/\
                    image.header["CDELT2"], (90-image.header["BPA"])*\
                    numpy.pi/180., 1.0)

            m.images["CO2-1"].image[:,:,ind,0] = scipy.signal.fftconvolve(\
                    m.images["CO2-1"].image[:,:,ind,0], beam, mode="same")

        os.system("rm params.txt")
        os.chdir(original_dir)
        os.system("rmdir /tmp/temp_ROXs12_{0:d}".format(comm.Get_rank()))

    return m

# Define a likelihood function.

def lnlike(params, visibilities, parameters, plot):

    m = model(visibilities, params, parameters, plot)

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

def lnprior(params, parameters):
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

    # Everything was correct, so continue on.

    return 0.0

# Define a probability function.

def lnprob(p, visibilities, parameters, plot):

    keys = []
    for key in sorted(parameters.keys()):
        if not parameters[key]["fixed"]:
            keys.append(key)

    params = dict(zip(keys, p))

    lp = lnprior(params, parameters)

    if not numpy.isfinite(lp):
        return -numpy.inf

    return lp + lnlike(params, visibilities, parameters, plot)

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

os.system("rm -r /tmp/temp_ROXs12_*")

################################################################################
#
# Set up a pool for parallel runs.
#
################################################################################

if args.action == "run":
    #pool = emcee.utils.MPIPool()
    pool = pdspy.modeling.mpi_pool.MPIPool(largedata=False)

    if not pool.is_master():
        pool.wait()
        sys.exit(0)

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
spectra["data"] = []
spectra["binned"] = []
images["data"] = []

######################################
# Read in the millimeter visibilities.
######################################

for j in range(len(visibilities["file"])):
    # Read the raw data.

    data = uv.Visibilities()
    data.read(visibilities["file"][j])

    # Center the data. => need to update!

    if parameters["x0"]["fixed"]:
        data = uv.center(data, [parameters["x0"]["value"], \
                parameters["y0"]["value"], 1.])

    # Add the data to the dictionary structure.

    visibilities["data"].append(data)

    # Scale the weights of the visibilities to force them to be fit well.

    visibilities["data"][j].weights *= visibilities["weight"][j]

    # Average the visibilities radially.

    visibilities["data1d"].append(uv.average(data, gridsize=20, radial=True, \
            log=True, logmin=data.uvdist[numpy.nonzero(data.uvdist)].min()*\
            0.95, logmax=data.uvdist.max()*1.05), mode="spectralline")

    # Read in the image.

    visibilities["image"].append(im.readimfits(visibilities["image_file"][j]))

################################################################################
#
# Fit the model to the data.
#
################################################################################

# Set up the emcee run.

ndim = 0
for key in parameters:
    if not parameters[key]["fixed"]:
        ndim += 1

# If we are resuming an MCMC simulation, read in the necessary info, otherwise
# set up the info.

if args.resume:
    pos = numpy.load("pos.npy")
    chain = numpy.load("chain.npy")
    state = None
    nsteps = chain[0,:,0].size

    if args.resetprob:
        prob = None
    else:
        prob = numpy.load("prob.npy")
else:
    pos = []
    for j in range(nwalkers):
        r_env = numpy.random.uniform(2.5,4.,1)[0]
        r_disk = numpy.random.uniform(numpy.log10(30.),\
                numpy.log10(10.**r_env),1)[0]
        r_in = numpy.random.uniform(numpy.log10(0.1),\
                numpy.log10((10.**r_disk)/2),1)[0]

        r_cav = numpy.random.uniform(r_in, numpy.log10(0.75*10.**r_disk),1)[0]

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
            elif key == "logM_disk":
                temp_pos.append(numpy.random.uniform(-5.,-3.,1)[0])
            elif key == "logM_env":
                temp_pos.append(numpy.random.uniform(-5.,-3.,1)[0])
            else:
                temp_pos.append(numpy.random.uniform(\
                        parameters[key]["limits"][0], \
                        parameters[key]["limits"][1], 1)[0])

        pos.append(temp_pos)

    prob = None
    chain = numpy.empty((nwalkers, 0, ndim))
    state = None
    nsteps = 0

# Set up the MCMC simulation.

if args.action == "run":
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, \
            args=(visibilities, parameters, False), pool=pool)

# Run a few burner steps.

while nsteps < 10000:
    if args.action == "run":
        pos, prob, state = sampler.run_mcmc(pos, steps_per_iter, lnprob0=prob, \
                rstate0=state)

        chain = numpy.concatenate((chain, sampler.chain), axis=1)

        # Get keys of the parameters that are varying.

        keys = []
        for key in sorted(parameters.keys()):
            if not parameters[key]["fixed"]:
                keys.append(key)

        # Plot the steps of the walkers.

        for j in range(ndim):
            fig, ax = plt.subplots(nrows=1, ncols=1)

            for k in range(nwalkers):
                ax.plot(chain[k,:,j])

            plt.savefig("steps_{0:s}.pdf".format(keys[j]))

            plt.close(fig)

        # Save walker positions in case the code stps running for some reason.

        numpy.save("pos", pos)
        numpy.save("prob", prob)
        numpy.save("chain", chain)

        # Augment the nuber of steps and reset the sampler for the next run.

        nsteps += steps_per_iter

        sampler.reset()

    # Get the best fit parameters and uncertainties.

    samples = chain[:,-nplot:,:].reshape((-1, ndim))

    params = numpy.median(samples, axis=0)
    sigma = samples.std(axis=0)

    # Write out the results.

    if args.action == "run":
        # Write out the results.

        f = open("fit.txt", "w")
        f.write("Best fit parameters:\n\n")
        for j in range(len(keys)):
            f.write("{0:s} = {1:f} +/- {2:f}\n".format(keys[j], params[j], \
                    sigma[j]))
        f.write("\n")
        f.close()

        os.system("cat flared_ROXs12/flared_fit.txt")

        # Plot histograms of the resulting parameters.

        fig = corner.corner(samples, labels=keys, truths=params)

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

    # Plot the best fit model over the data.

    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)

    # Create a high resolution model for averaging.

    m = model(visibilities, params, parameters, plot=True)

    for j in range(len(visibilities["file"])):
        # Calculate the velocity for each image.

        v = c * (visibilities["nu0"][j] - visibilities["image"][j].nu) / \
                visibilities["nu0"][j]

        # Plot the image.

        vmin = numpy.nanmin(visibilities["image"][j].image)
        vmax = numpy.nanmax(visibilities["image"][j].image)

        for k in range(5):
            for l in range(5):
                ind = k*5 + l + 1

                # Get the centroid position.

                ticks = visibilities["image_ticks"][j]

                xmin, xmax = int(visibilities["image_npix"][j]/2 + \
                        params["x0"]/visibilities["image_pixelsize"][j]+ \
                        round(ticks[0]/visibilities["image_pixelsize"][j])), \
                        round(visibilities["image_npix"][j]/2+\
                        params["x0"]/visibilities["image_pixelsize"][j]+ \
                        int(ticks[-1]/visibilities["image_pixelsize"][j]))
                ymin, ymax = int(visibilities["image_npix"][j]/2+\
                        params["y0"]/visibilities["image_pixelsize"][j]+ \
                        round(ticks[0]/visibilities["image_pixelsize"][j])), \
                        round(visibilities["image_npix"][j]/2+\
                        params["y0"]/visibilities["image_pixelsize"][j]+ \
                        int(ticks[-1]/visibilities["image_pixelsize"][j]))

                # Plot the image.

                ax[k,l].imshow(visibilities["image"][j].image[ymin:ymax,\
                        xmin:xmax,ind,0], origin="lower", \
                        interpolation="nearest", vmin=vmin, vmax=vmax)

                # Now make the centroid the map center for the model.

                xmin, xmax = int(visibilities["image_npix"][j]/2 + \
                        round(ticks[0]/visibilities["image_pixelsize"][j])), \
                        round(visibilities["image_npix"][j]/2+\
                        int(ticks[-1]/visibilities["image_pixelsize"][j]))
                ymin, ymax = int(visibilities["image_npix"][j]/2+\
                        round(ticks[0]/visibilities["image_pixelsize"][j])), \
                        round(visibilities["image_npix"][j]/2+\
                        int(ticks[-1]/visibilities["image_pixelsize"][j]))

                # Plot the model image.

                levels = numpy.array([0.05, 0.25, 0.45, 0.65, 0.85, 0.95]) * \
                        m.images[visibilities["lam"][j]].image.max()

                ax[i,j].contour(m.images[visibilities["lam"][j]].\
                        image[ymin:ymax,xmin:xmax,ind,0], levels=levels)

                # Add the velocity to the map.

                txt = ax[k,l].annotate(r"$v=%4.1f$ km s$^{-1}$" % (v[ind]/1e5),\
                        xy=(0.1,0.8), xycoords='axes fraction')

                #txt.set_path_effects([PathEffects.withStroke(linewidth=2, \
                #        foreground='w')])

                # Fix the axes labels.

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
                        abs(image.header["CDELT1"])
                bmin = visibilities["image"][j].header["BMIN"] / \
                        abs(image.header["CDELT1"])
                bpa = visibilities["image"][j].header["BPA"]

                ax[k,l].add_artist(patches.Ellipse(xy=(12.5,17.5), width=bmaj, \
                        height=bmin, angle=(bpa+90), facecolor="white", \
                        edgecolor="black"))

                ax[k,l].set_adjustable('box-forced')

            ax[k,0].set_ylabel("$\Delta$Dec")

    # Adjust the plot and save it.

    fig.set_size_inches((10,9))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.07, \
            wspace=0.0,hspace=0.0)

    # Adjust the figure and save.

    fig.savefig("model.pdf")

    plt.clf()

    if args.action == "plot":
        nsteps = numpy.inf

# Now we can close the pool.

if args.action == "run":
    pool.close()
