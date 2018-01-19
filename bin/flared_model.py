#!/usr/bin/env python3

from pdspy.constants.physics import c, m_p, G
from pdspy.constants.physics import k as k_b
from pdspy.constants.astronomy import M_sun, AU
from matplotlib.backends.backend_pdf import PdfPages
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
parser.add_argument('-n', '--ncpus', type=int, default=1)
parser.add_argument('-e', '--withexptaper', action='store_true')
parser.add_argument('-v', '--plot_vis', action='store_true')
parser.add_argument('-c', '--withcontsub', action='store_true')
args = parser.parse_args()

# Check whether we are using MPI.

withmpi = comm.Get_size() > 1

# Set the number of cpus to use.

ncpus = args.ncpus

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
            if parameters[key]["value"] in parameters.keys():
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

    # Make sure alpha and beta are defined.

    if p["disk_type"] == "exptaper":
        t_rdisk = p["T0"] * (p["R_disk"] / 1.)**-p["q"]
        p["h_0"] = ((k_b*(p["R_disk"]*AU)**3*t_rdisk) / (G*p["M_star"]*M_sun * \
                2.37*m_p))**0.5 / AU
    else:
        p["h_0"] = ((k_b * AU**3 * p["T0"]) / (G*p["M_star"]*M_sun * \
                2.37*m_p))**0.5 / AU
    p["beta"] = 0.5 * (3 - p["q"])
    p["alpha"] = p["gamma"] + p["beta"]

    # Shift the wavelengths by the velocities.

    b = p["v_sys"]*1.0e5 / c
    lam = c / visibilities["data"][0].freq / 1.0e-4
    wave = lam * numpy.sqrt((1. - b) / (1. + b))

    # Set up the dust.

    dustopac = "pollack_new.hdf5"

    dust_gen = dust.DustGenerator(dust.__path__[0]+"/data/"+dustopac)

    ddust = dust_gen(p["a_max"] / 1e4, p["p"])
    edust = dust_gen(1.0e-4, 3.5)

    # Set up the gas.

    gases = []
    abundance = []

    index = 1
    while index > 0:
        if "gas_file"+str(index) in p:
            g = gas.Gas()
            g.set_properties_from_lambda(gas.__path__[0]+"/data/"+\
                    p["gas_file"+str(index)])

            gases.append(g)
            abundance.append(p["abundance"+str(index)])

            index += 1
        else:
            index = -1

    # Make sure we are in a temp directory to not overwrite anything.

    original_dir = os.environ["PWD"]
    os.mkdir("/tmp/temp_{1:s}_{0:d}".format(comm.Get_rank(), source))
    os.chdir("/tmp/temp_{1:s}_{0:d}".format(comm.Get_rank(), source))

    # Write the parameters to a text file so it is easy to keep track of them.

    f = open("params.txt","w")
    for key in p:
        f.write("{0:s} = {1}\n".format(key, p[key]))
    f.close()

    # Set up the model. 

    m = modeling.YSOModel()
    m.add_star(mass=p["M_star"], luminosity=p["L_star"],temperature=p["T_star"])

    if p["envelope_type"] == "ulrich":
        m.set_spherical_grid(p["R_in"], p["R_env"], 100, 51, 2, code="radmc3d")
    else:
        m.set_spherical_grid(p["R_in"], max(5*p["R_disk"],300), 100, 51, 2, \
                code="radmc3d")

    if p["disk_type"] == "exptaper":
        m.add_pringle_disk(mass=p["M_disk"], rmin=p["R_in"], rmax=p["R_disk"], \
                plrho=p["alpha"], h0=p["h_0"], plh=p["beta"], dust=ddust, \
                t0=p["T0"], plt=p["q"], gas=gases, abundance=abundance,\
                aturb=p["a_turb"])
    else:
        m.add_disk(mass=p["M_disk"], rmin=p["R_in"], rmax=p["R_disk"], \
                plrho=p["alpha"], h0=p["h_0"], plh=p["beta"], dust=ddust, \
                t0=p["T0"], plt=p["q"], gas=gases, abundance=abundance,\
                aturb=p["a_turb"])

    if p["envelope_type"] == "ulrich":
        m.add_ulrich_envelope(mass=p["M_env"], rmin=p["R_in"], rmax=p["R_env"],\
                cavpl=p["ksi"], cavrfact=p["f_cav"], dust=edust, \
                t0=p["T0_env"], tpl=p["q_env"], gas=gases, abundance=abundance,\
                aturb=p["a_turb_env"])
    else:
        pass

    m.grid.set_wavelength_grid(0.1,1.0e5,500,log=True)

    # Run the images/visibilities/SEDs.

    for j in range(len(visibilities["file"])):
        m.set_camera_wavelength(wave)

        if p["docontsub"]:
            m.run_image(name=visibilities["lam"][j], nphot=1e5, \
                    npix=visibilities["npix"][j], lam=None, \
                    pixelsize=visibilities["pixelsize"][j], tgas_eq_tdust=True,\
                    scattering_mode_max=0, incl_dust=True, incl_lines=True, \
                    loadlambda=True, incl=p["i"], pa=p["pa"], dpc=p["dpc"], \
                    code="radmc3d", verbose=False, writeimage_unformatted=True,\
                    setthreads=ncpus)

            m.run_image(name="cont", nphot=1e5, \
                    npix=visibilities["npix"][j], lam=None, \
                    pixelsize=visibilities["pixelsize"][j], tgas_eq_tdust=True,\
                    scattering_mode_max=0, incl_dust=True, incl_lines=False, \
                    loadlambda=True, incl=p["i"], pa=p["pa"], dpc=p["dpc"], \
                    code="radmc3d", verbose=False, writeimage_unformatted=True,\
                    setthreads=ncpus)

            m.images[visibilities["lam"][j]].image -= m.images["cont"].image
        else:
            m.run_image(name=visibilities["lam"][j], nphot=1e5, \
                    npix=visibilities["npix"][j], lam=None, \
                    pixelsize=visibilities["pixelsize"][j], tgas_eq_tdust=True,\
                    scattering_mode_max=0, incl_dust=False, incl_lines=True, \
                    loadlambda=True, incl=p["i"], pa=p["pa"], dpc=p["dpc"], \
                    code="radmc3d", verbose=False, writeimage_unformatted=True,\
                    setthreads=ncpus)

        m.visibilities[visibilities["lam"][j]] = uv.interpolate_model(\
                visibilities["data"][j].u, visibilities["data"][j].v, \
                visibilities["data"][j].freq, \
                m.images[visibilities["lam"][j]], dRA=-p["x0"], dDec=-p["y0"], \
                nthreads=ncpus)

        if plot:
            lam = c / visibilities["image"][0].freq / 1.0e-4
            wave = lam * numpy.sqrt((1. - b) / (1. + b))

            m.set_camera_wavelength(wave)

            if p["docontsub"]:
                m.run_image(name=visibilities["lam"][j], nphot=1e5, \
                        npix=visibilities["image_npix"][j], lam=None, \
                        pixelsize=visibilities["image_pixelsize"][j], \
                        tgas_eq_tdust=True, scattering_mode_max=0, \
                        incl_dust=True, incl_lines=True, loadlambda=True, \
                        incl=p["i"], pa=-p["pa"], dpc=p["dpc"], code="radmc3d",\
                        verbose=False, setthreads=ncpus)

                m.run_image(name="cont", nphot=1e5, \
                        npix=visibilities["image_npix"][j], lam=None, \
                        pixelsize=visibilities["image_pixelsize"][j], \
                        tgas_eq_tdust=True, scattering_mode_max=0, \
                        incl_dust=True, incl_lines=False, loadlambda=True, \
                        incl=p["i"], pa=-p["pa"], dpc=p["dpc"], code="radmc3d",\
                        verbose=False, setthreads=ncpus)

                m.images[visibilities["lam"][j]].image -= m.images["cont"].image
            else:
                m.run_image(name=visibilities["lam"][j], nphot=1e5, \
                        npix=visibilities["image_npix"][j], lam=None, \
                        pixelsize=visibilities["image_pixelsize"][j], \
                        tgas_eq_tdust=True, scattering_mode_max=0, \
                        incl_dust=False, incl_lines=True, loadlambda=True, \
                        incl=p["i"], pa=-p["pa"], dpc=p["dpc"], code="radmc3d",\
                        verbose=False, setthreads=ncpus)

            x, y = numpy.meshgrid(numpy.linspace(-256,255,512), \
                    numpy.linspace(-256,255,512))

            beam = misc.gaussian2d(x, y, 0., 0., \
                    visibilities["image"][j].header["BMAJ"]/2.355/\
                    visibilities["image"][j].header["CDELT2"], \
                    visibilities["image"][j].header["BMIN"]/2.355/\
                    visibilities["image"][j].header["CDELT2"], \
                    (90-visibilities["image"][j].header["BPA"])*\
                    numpy.pi/180., 1.0)

            for ind in range(len(wave)):
                m.images[visibilities["lam"][j]].image[:,:,ind,0] = \
                        scipy.signal.fftconvolve(\
                        m.images[visibilities["lam"][j]].image[:,:,ind,0], \
                        beam, mode="same")

            if args.plot_vis:
                m.visibilities[visibilities["lam"][j]] = \
                        uv.average(m.visibilities[visibilities["lam"][j]], \
                        gridsize=20, radial=True, log=True, \
                        logmin=m.visibilities[visibilities["lam"][j]].uvdist[\
                        numpy.nonzero(m.visibilities[visibilities["lam"][j]].\
                        uvdist)].min()*0.95, logmax=m.visibilities[\
                        visibilities["lam"][j]].uvdist.max()*1.05, \
                        mode="spectralline")

    os.system("rm params.txt")
    os.chdir(original_dir)
    os.system("rmdir /tmp/temp_{1:s}_{0:d}".format(comm.Get_rank(), source))

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

os.system("rm -r /tmp/temp_{0:s}_{1:d}".format(source, comm.Get_rank()))

################################################################################
#
# Set up a pool for parallel runs.
#
################################################################################

if args.action == "run":
    if withmpi:
        #pool = emcee.utils.MPIPool()
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

# Make sure "fmt" is in the visibilities dictionary.

if not "fmt" in visibilities:
    visibilities["fmt"] = ['4.1f' for i in range(len(visibilities["file"]))]

# Decide whether to use an exponentially tapered 

if not "disk_type" in parameters:
    parameters["disk_type"] = {"fixed":True, "value":"truncated", \
            "limits":[0.,0.]}

if args.withexptaper:
    parameters["disk_type"]["value"] = "exptaper"

# Make sure the code doesn't break if envelope_type isn't specified.

if not "envelope_type" in parameters:
    parameters["envelope_type"] = {"fixed":True, "value":"none", \
            "limits":[0.,0.]}

# Decide whether to do continuum subtraction or not.

if not "docontsub" in parameters:
    parameters["docontsub"] = {"fixed":True, "value":False, "limits":[0.,0.]}

if args.withcontsub:
    parameters["docontsub"]["value"] = True

# Make sure the code is backwards compatible to a time when only a single gas
# file was being supplied.

if "gas_file" in parameters:
    parameters["gas_file1"] = parameters["gas_file"]
    parameters["logabundance1"] = parameters["logabundance"]

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
        r_env = numpy.random.uniform(parameters["logR_env"]["limits"][0],\
                parameters["logR_env"]["limits"][1],1)[0]
        r_disk = numpy.random.uniform(numpy.log10(5.),\
                min(r_env, parameters["logR_disk"]["limits"][1]),1)[0]
        r_in = numpy.random.uniform(parameters["logR_in"]["limits"][0],\
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

# If we are plotting, make sure that nsteps < max_nsteps.

if args.action == "plot":
    nsteps = max_nsteps - 1

# Run a few burner steps.

while nsteps < max_nsteps:
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

        os.system("cat fit.txt")

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

    # Create a high resolution model for averaging.

    m = model(visibilities, params, parameters, plot=True)

    # Open up a pdf file to plot into.

    pdf = PdfPages("model.pdf")

    # Loop through the visibilities and plot.

    for j in range(len(visibilities["file"])):
        # Plot the best fit model over the data.

        fig, ax = plt.subplots(nrows=visibilities["nrows"][j], \
                ncols=visibilities["ncols"][j], sharex=True, sharey=True)

        # Calculate the velocity for each image.

        v = c * (float(visibilities["freq"][j])*1.0e9 - \
                visibilities["image"][j].freq)/(float(visibilities["freq"][j])*\
                1.0e9)

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

                    ax[k,l].set_adjustable('box-forced')

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

    # If we're just plotting make sure we don't loop forever.

    if args.action == "plot":
        nsteps = numpy.inf

# Now we can close the pool.

if args.action == "run":
    if withmpi:
        pool.close()
