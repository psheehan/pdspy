#!/usr/bin/env python3

from pdspy.constants.astronomy import arcsec
import pdspy.interferometry as uv
import pdspy.spectroscopy as sp
import pdspy.modeling as modeling
import pdspy.imaging as im
import pdspy.misc as misc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pdspy.table
import pdspy.dust as dust
import pdspy.mcmc as mc
import scipy.signal
import subprocess
import argparse
import signal
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
parser.add_argument('-c', '--withhyperion', action='store_true')
parser.add_argument('-p', '--resetprob', action='store_true')
parser.add_argument('-a', '--action', type=str, default="run")
parser.add_argument('-n', '--ncpus', type=int, default=1)
parser.add_argument('-m', '--ncpus_highmass', type=int, default=8)
parser.add_argument('-e', '--withexptaper', action='store_true')
parser.add_argument('-t', '--timelimit', type=int, default=7200)
args = parser.parse_args()

# Check whether we are using MPI.

withmpi = comm.Get_size() > 1

# Set the number of cpus to use.

ncpus = args.ncpus
ncpus_highmass = args.ncpus_highmass

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

def model(visibilities, images, spectra, params, parameters, plot=False):

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

    # Make sure alpha is defined.

    p["alpha"] = p["gamma"] + p["beta"]
    p["alpha_large"] = p["gamma"] + p["beta_large"]

    # If we're using a Pringle disk, make sure the scale height is set correctly

    if p["disk_type"] == "exptaper":
        p["h_0"] *= p["R_disk"]**p["beta"]

    # Get the needed values of the gaps.

    p["R_in_gap1"] = p["R_gap1"] - p["w_gap1"]/2
    p["R_out_gap1"] = p["R_gap1"] + p["w_gap1"]/2
    p["R_in_gap2"] = p["R_gap2"] - p["w_gap2"]/2
    p["R_out_gap2"] = p["R_gap2"] + p["w_gap2"]/2
    p["R_in_gap3"] = p["R_gap3"] - p["w_gap3"]/2
    p["R_out_gap3"] = p["R_gap3"] + p["w_gap3"]/2

    # Set up the dust.

    dustopac = "pollack_new.hdf5"

    dust_gen = dust.DustGenerator(dust.__path__[0]+"/data/"+dustopac)

    ddust = dust_gen(p["a_max"] / 1e4, p["p"])
    edust = dust_gen(1.0e-4, 3.5)

    # Make sure we are in a temp directory to not overwrite anything.

    original_dir = os.environ["PWD"]
    os.mkdir("/tmp/temp_{1:s}_{0:d}".format(comm.Get_rank(), source))
    os.chdir("/tmp/temp_{1:s}_{0:d}".format(comm.Get_rank(), source))

    # Write the parameters to a text file so it is easy to keep track of them.

    f = open("params.txt","w")
    for key in p:
        f.write("{0:s} = {1}\n".format(key, p[key]))
    f.close()

    # Set up the model and run the thermal simulation.

    if p["M_disk"] > 0.001 or p["R_disk"] < 50 or p["M_env"] > 0.001 or \
            p["R_env"] < 500 or p["h_0"] > 0.25:
        if args.withhyperion:
            nphi = 201
            code = "hyperion"
            nprocesses = ncpus_highmass
        else:
            nphi = 101
            code = "radmc3d"
            nprocesses = ncpus_highmass
    else:
        nphi = 101
        code = "radmc3d"
        nprocesses = ncpus

    m = modeling.YSOModel()
    m.add_star(mass=p["M_star"],luminosity=p["L_star"],temperature=p["T_star"])
    m.set_spherical_grid(p["R_in"], p["R_env"], 100, nphi, 2, code=code)

    if p["disk_type"] == "exptaper":
        m.add_pringle_disk(mass=p["M_disk"]*p["f_M_large"], rmin=p["R_in"], \
                rmax=p["R_disk"], plrho=p["alpha_large"], \
                h0=p["h_0"]*p["f_h_large"], plh=p["beta_large"], dust=ddust, \
                gap_rin=[p["R_in"],p["R_in_gap1"],p["R_in_gap2"],\
                p["R_in_gap3"]], gap_rout=[p["R_cav"],p["R_out_gap1"],\
                p["R_out_gap2"],p["R_out_gap3"]], gap_delta=[p["delta_cav"],\
                p["delta_gap1"],p["delta_gap2"],p["delta_gap3"]])
        if p["f_M_large"] < 1:
            m.add_pringle_disk(mass=p["M_disk"]*(1-p["f_M_large"]), \
                    rmin=p["R_in"], rmax=p["R_disk"], plrho=p["alpha"], \
                    h0=p["h_0"], plh=p["beta"], dust=edust, gap_rin=[p["R_in"],\
                    p["R_in_gap1"],p["R_in_gap2"],p["R_in_gap3"]], \
                    gap_rout=[p["R_cav"],p["R_out_gap1"],p["R_out_gap2"],\
                    p["R_out_gap3"]], gap_delta=[p["delta_cav"],\
                    p["delta_gap1"], p["delta_gap2"],p["delta_gap3"]])
    else:
        m.add_disk(mass=p["M_disk"]*p["f_M_large"], rmin=p["R_in"], \
                rmax=p["R_disk"], plrho=p["alpha_large"], \
                h0=p["h_0"]*p["f_h_large"], plh=p["beta_large"], dust=ddust, \
                gap_rin=[p["R_in"],p["R_in_gap1"],p["R_in_gap2"],\
                p["R_in_gap3"]], gap_rout=[p["R_cav"],p["R_out_gap1"],\
                p["R_out_gap2"],p["R_out_gap3"]], gap_delta=[p["delta_cav"],\
                p["delta_gap1"],p["delta_gap2"],p["delta_gap3"]])
        if p["f_M_large"] < 1:
            m.add_disk(mass=p["M_disk"]*(1-p["f_M_large"]), rmin=p["R_in"], \
                    rmax=p["R_disk"], plrho=p["alpha"], h0=p["h_0"], \
                    plh=p["beta"], dust=edust, gap_rin=[p["R_in"],\
                    p["R_in_gap1"],p["R_in_gap2"],p["R_in_gap3"]], \
                    gap_rout=[p["R_cav"],p["R_out_gap1"],p["R_out_gap2"],\
                    p["R_out_gap3"]], gap_delta=[p["delta_cav"],\
                    p["delta_gap1"], p["delta_gap2"],p["delta_gap3"]])

    if p["envelope_type"] == "ulrich":
        m.add_ulrich_envelope(mass=p["M_env"], rmin=p["R_in"], rmax=p["R_env"],\
                cavpl=p["ksi"], cavrfact=p["f_cav"], dust=edust)
    else:
        pass

    m.grid.set_wavelength_grid(0.1,1.0e5,500,log=True)

    # Run the thermal simulation.

    if code == "hyperion":
        m.run_thermal(code="hyperion", nphot=2e5, mrw=True, pda=True, \
                niterations=20, mpi=True, nprocesses=nprocesses, verbose=False)

        # Convert model to radmc-3d format.

        m.make_hyperion_symmetric()

        m.convert_hyperion_to_radmc3d()
    else:
        try:
            t1 = time.time()
            m.run_thermal(code="radmc3d", nphot=1e6, modified_random_walk=True,\
                    mrw_gamma=2, mrw_tauthres=10, mrw_count_trigger=100, \
                    verbose=False, setthreads=nprocesses, \
                    timelimit=args.timelimit)
            t2 = time.time()
            f = open(original_dir + "/times.txt", "a")
            f.write("{0:f}\n".format(t2-t1))
            f.close()
        # Catch a timeout error from models running for too long...
        except subprocess.TimeoutExpired:
            t2 = time.time()
            f = open(original_dir + "/times.txt", "a")
            f.write("{0:f}\n".format(t2-t1))
            f.close()

            os.system("mv params.txt {0:s}/params_timeout_{1:s}".format(\
                    original_dir, time.strftime("%Y-%m-%d-%H:%M:%S", \
                    time.gmtime())))
            os.system("rm *.inp *.out *.dat *.uinp")
            os.chdir(original_dir)
            os.rmdir("/tmp/temp_{1:s}_{0:d}".format(comm.Get_rank(), source))

            return 0.
        # Catch a strange RADMC3D error and re-run. Not an ideal fix, but 
        # perhaps the simplest option.
        except:
            t1 = time.time()
            m.run_thermal(code="radmc3d", nphot=1e6, modified_random_walk=True,\
                    mrw_gamma=2, mrw_tauthres=10, mrw_count_trigger=100, \
                    verbose=False, setthreads=nprocesses, \
                    timelimit=args.timelimit)
            t2 = time.time()
            f = open(original_dir + "/times.txt", "a")
            f.write("{0:f}\n".format(t2-t1))
            f.close()

    # Run the images/visibilities/SEDs. If plot == "concat" then we are doing
    # a fit and we need less. Otherwise we are making a plot of the best fit 
    # model so we need to generate a few extra things.

    # Run the visibilities.

    for j in range(len(visibilities["file"])):
        m.run_visibilities(name=visibilities["lam"][j], nphot=1e5, \
                npix=visibilities["npix"][j], \
                pixelsize=visibilities["pixelsize"][j], \
                lam=visibilities["lam"][j], incl=p["i"], \
                pa=p["pa"], dpc=p["dpc"], code="radmc3d", \
                mc_scat_maxtauabs=5, verbose=False, setthreads=nprocesses)

        """NEW: Interpolate model to native baselines?
        m.visibilities[visibilities["lam"][j]] = uv.interpolate_model(\
                visibilities["data"].u, visibilities["data"].v, \
                visibilities["data"].freq, \
                m.visibilities[visibilities["lam"][j]])
        """

        m.visibilities[visibilities["lam"][j]] = uv.center(\
                m.visibilities[visibilities["lam"][j]], [p["x0"], \
                p["y0"], 1.])

        if plot:
            # Run a high resolution version of the visibilities.

            m.run_visibilities(name=visibilities["lam"][j]+"_high", nphot=1e5, \
                    npix=2048, pixelsize=0.05, lam=visibilities["lam"][j], \
                    incl=p["i"], pa=p["pa"], dpc=p["dpc"], \
                    code="radmc3d", mc_scat_maxtauabs=5, verbose=False, \
                    setthreads=nprocesses)

            m.visibilities[visibilities["lam"][j]+"_high"] = uv.center(\
                    m.visibilities[visibilities["lam"][j]+"_high"], \
                    [p["x0"], p["y0"], 1.])

            # Run a millimeter image.

            m.run_image(name=visibilities["lam"][j], nphot=1e5, \
                    npix=visibilities["image_npix"][j], \
                    pixelsize=visibilities["image_pixelsize"][j], \
                    lam=visibilities["lam"][j], incl=p["i"], \
                    pa=-p["pa"], dpc=p["dpc"], code="radmc3d", \
                    mc_scat_maxtauabs=5, verbose=False, setthreads=nprocesses)

            x, y = numpy.meshgrid(numpy.linspace(-256,255,512), \
                    numpy.linspace(-256,255,512))

            beam = misc.gaussian2d(x, y, 0., 0., \
                    visibilities["image"][j].header["BMAJ"]/2.355/\
                    visibilities["image"][j].header["CDELT2"], \
                    visibilities["image"][j].header["BMIN"]/2.355/\
                    visibilities["image"][j].header["CDELT2"], \
                    (90-visibilities["image"][j].header["BPA"])*numpy.pi/180., \
                    1.0)

            m.images[visibilities["lam"][j]].image = scipy.signal.fftconvolve(\
                    m.images[visibilities["lam"][j]].image[:,:,0,0], beam, \
                    mode="same").reshape(m.images[visibilities["lam"][j]].\
                    image.shape)

    # Run the images.

    for j in range(len(images["file"])):
        m.run_image(name=images["lam"][j], nphot=1e5, \
                npix=images["npix"][j], pixelsize=images["pixelsize"][j], \
                lam=images["lam"][j], incl=p["i"], \
                pa=p["pa"], dpc=p["dpc"], code="radmc3d", \
                mc_scat_maxtauabs=5, verbose=False, setthreads=nprocesses)

        # Convolve with the beam.

        x, y = numpy.meshgrid(numpy.linspace(-256,255,512), \
                numpy.linspace(-256,255,512))

        beam = misc.gaussian2d(x, y, 0., 0., 1., 1., 0., 1.0)

        m.images[images["lam"][j]].image = scipy.signal.fftconvolve(\
                m.images[images["lam"][j]].image[:,:,0,0], beam, mode="same").\
                reshape(m.images[images["lam"][j]].image.shape)

    # Run the SED.

    if "total" in spectra:
        if plot:
            m.set_camera_wavelength(numpy.logspace(-1,4,50))
        else:
            m.set_camera_wavelength(spectra["total"].wave)

        m.run_sed(name="SED", nphot=1e4, loadlambda=True, incl=p["i"],\
                pa=p["pa"], dpc=p["dpc"], code="radmc3d", \
                camera_scatsrc_allfreq=True, mc_scat_maxtauabs=5, \
                verbose=False, setthreads=nprocesses)

        m.spectra["SED"].flux = dust.redden(m.spectra["SED"].wave, \
                m.spectra["SED"].flux, p["Ak"], law="mcclure")

    # Clean up everything and return.

    os.system("rm params.txt")
    os.chdir(original_dir)
    os.rmdir("/tmp/temp_{1:s}_{0:d}".format(comm.Get_rank(), source))

    return m

# Define a likelihood function.

def lnlike(params, visibilities, images, spectra, parameters, plot):

    m = model(visibilities, images, spectra, params, parameters, plot)

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
                m.images[images["lam"][j]].image) / images["data"][j].unc**2)))

    # Calculate the chisq for the SED.

    if "total" in spectra:
        chisq.append(-0.5 * (numpy.sum((spectra["total"].flux - \
                m.spectra["SED"].flux) / spectra["total"].unc**2)))

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

def lnprob(p, visibilities, images, spectra, parameters, plot):

    keys = []
    for key in sorted(parameters.keys()):
        if not parameters[key]["fixed"]:
            keys.append(key)

    params = dict(zip(keys, p))

    lp = lnprior(params, parameters)

    if not numpy.isfinite(lp):
        return -numpy.inf

    return lp + lnlike(params, visibilities, images, spectra, parameters, \
            plot)

# Define a useful class for plotting.

class Transform:
    def __init__(self, xmin, xmax, dx, fmt):
        self.xmin = xmin
        self.xmax = xmax
        self.dx = dx
        self.fmt = fmt

    def __call__(self, x, p):
        return self.fmt% ((x-(self.xmax-self.xmin)/2)*self.dx)

# Define a function to scale an image to look nice.

def scale_image(image, mode="linear"):
    vmin = image.image.min()
    vmax = numpy.percentile(image.image, 95)

    a = 1000.
    b = (image.image - vmin) / (vmax - vmin)

    if mode == "linear":
        c = b
    elif mode == "arcsinh":
        c = numpy.arcsinh(10*b)/3.
    elif mode == "log":
        c = numpy.log10(a*b+1)/numpy.log10(a)
    else:
        print("Not a valid mode!")

    return c

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

# Set up the places where we will put all of the data.

visibilities["data"] = []
visibilities["data1d"] = []
visibilities["image"] = []
spectra["data"] = []
spectra["binned"] = []
images["data"] = []

# Decide whether to use an exponentially tapered 

if not "disk_type" in parameters:
    parameters["disk_type"] = {"fixed":True, "value":"truncated", \
            "limits":[0.,0.]}

if args.withexptaper:
    parameters["disk_type"]["value"] = "exptaper"

# Make sure the code doesn't break if envelope_type isn't specified.

if not "envelope_type" in parameters:
    parameters["envelope_type"] = {"fixed":True, "value":"ulrich", \
            "limits":[0.,0.]}

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

    # Merge the SED with the binned Spitzer spectrum.

    if spectra["bin?"]:
        wave = numpy.linspace(spectra["data"][j].wave.min(), \
                spectra["data"][j].wave.max(), spectra["nbins"][j])
        flux = numpy.interp(wave, spectra["data"][j].wave, \
                spectra["data"][j].flux)

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

    # Adjust the weight of the SED, as necessary.

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
        r_disk = numpy.random.uniform(numpy.log10(30.),\
                numpy.log10(10.**r_env),1)[0]
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
            elif key == "logM_disk":
                temp_pos.append(numpy.random.uniform(-6.,\
                        parameters[key]["limits"][1],1)[0])
            elif key == "logM_env":
                temp_pos.append(numpy.random.uniform(-6.,\
                        parameters[key]["limits"][1],1)[0])
            elif key == "h_0":
                temp_pos.append(numpy.random.uniform(\
                        parameters[key]["limits"][0], 0.2, 1)[0])
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
            args=(visibilities, images, spectra, parameters, False), \
            pool=pool)

# Run a few burner steps.

while nsteps < max_nsteps:
    if args.action == "run":
        for i in range(steps_per_iter):
            pos, prob, state = sampler.run_mcmc(pos, 1, lnprob0=prob, \
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

            # Save walker positions in case the code stps running for some 
            # reason.

            numpy.save("pos", pos)
            numpy.save("prob", prob)
            numpy.save("chain", chain)

            # Augment the nuber of steps and reset the sampler for the next run.

            nsteps += 1

            sampler.reset()

    # Get the best fit parameters and uncertainties from the last 10 steps.

    samples = chain[:,-nplot:,:].reshape((-1, ndim))

    params = numpy.median(samples, axis=0)
    sigma = samples.std(axis=0)

    # Print out the status of the fit.

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

    # Plot the best fit model over the data.

    fig, ax = plt.subplots(nrows=2*len(visibilities["file"]), ncols=3)

    # Create a high resolution model for averaging.

    m = model(visibilities, images, spectra, params, parameters, plot=True)

    # Plot the millimeter data/models.

    for j in range(len(visibilities["file"])):
        # Create a high resolution model for averaging.

        m1d = uv.average(m.visibilities[visibilities["lam"][j]+"_high"], \
                gridsize=10000, binsize=3500, radial=True)

        # Plot the visibilities.

        ax[2*j,0].errorbar(visibilities["data1d"][j].uvdist/1000, \
                visibilities["data1d"][j].amp, \
                yerr=numpy.sqrt(1./visibilities["data1d"][j].weights),\
                fmt="bo", markersize=8, markeredgecolor="b")

        # Plot the best fit model

        ax[2*j,0].plot(m1d.uvdist/1000, m1d.amp, "g-")

        # Plot the 2D visibilities.

        ticks = visibilities["ticks"][j]

        xmin, xmax = int(round(visibilities["npix"][j]/2+ticks[0]/\
                (visibilities["binsize"][j]/1000))), \
                int(round(visibilities["npix"][j]/2+ticks[6]/\
                (visibilities["binsize"][j]/1000)))
        ymin, ymax = int(round(visibilities["npix"][j]/2+ticks[0]/\
                (visibilities["binsize"][j]/1000))), \
                int(round(visibilities["npix"][j]/2+ticks[6]/\
                (visibilities["binsize"][j]/1000)))

        vmin = min(0, visibilities["data1d"][j].real.min())
        vmax = visibilities["data1d"][j].real.max()

        ax[2*j+1,0].imshow(visibilities["data"][j].real.reshape(\
                (visibilities["npix"][j],visibilities["npix"][j]))\
                [xmin:xmax,xmin:xmax][:,::-1], origin="lower", \
                interpolation="nearest", vmin=vmin, vmax=vmax)
        ax[2*j+1,0].contour(m.visibilities[visibilities["lam"][j]].real.\
                reshape((visibilities["npix"][j],visibilities["npix"][j]))\
                [xmin:xmax,xmin:xmax][:,::-1])

        vmin = -visibilities["data1d"][j].real.max()
        vmax =  visibilities["data1d"][j].real.max()

        ax[2*j+1,1].imshow(visibilities["data"][j].imag.reshape(\
                (visibilities["npix"][j],visibilities["npix"][j]))\
                [xmin:xmax,xmin:xmax][:,::-1], origin="lower", \
                interpolation="nearest", vmin=vmin, vmax=vmax)
        ax[2*j+1,1].contour(m.visibilities[visibilities["lam"][j]].imag.\
                reshape((visibilities["npix"][j],visibilities["npix"][j]))\
                [xmin:xmax,xmin:xmax][:,::-1])

        transform1 = ticker.FuncFormatter(Transform(xmin, xmax, \
                visibilities["binsize"][j]/1000, '%.0f'))

        ax[2*j+1,0].set_xticks(visibilities["npix"][j]/2+ticks[1:-1]/\
                (visibilities["binsize"][j]/1000)-xmin)
        ax[2*j+1,0].set_yticks(visibilities["npix"][j]/2+ticks[1:-1]/\
                (visibilities["binsize"][j]/1000)-ymin)
        ax[2*j+1,0].get_xaxis().set_major_formatter(transform1)
        ax[2*j+1,0].get_yaxis().set_major_formatter(transform1)

        ax[2*j+1,1].set_xticks(visibilities["npix"][j]/2+ticks[1:-1]/\
                (visibilities["binsize"][j]/1000)-xmin)
        ax[2*j+1,1].set_yticks(visibilities["npix"][j]/2+ticks[1:-1]/\
                (visibilities["binsize"][j]/1000)-ymin)
        ax[2*j+1,1].get_xaxis().set_major_formatter(transform1)
        ax[2*j+1,1].get_yaxis().set_major_formatter(transform1)

        # Create a model image to contour over the image.

        model_image = m.images[visibilities["lam"][j]]

        # Plot the image.

        ticks = visibilities["image_ticks"][j]

        if "x0" in params:
            xmin, xmax = int(round(visibilities["image_npix"][j]/2 + \
                    visibilities["x0"][j]/visibilities["image_pixelsize"][j]+ \
                    params["x0"]/visibilities["image_pixelsize"][j]+ \
                    ticks[0]/visibilities["image_pixelsize"][j])), \
                    int(round(visibilities["image_npix"][j]/2+\
                    visibilities["x0"][j]/visibilities["image_pixelsize"][j]+ \
                    params["x0"]/visibilities["image_pixelsize"][j]+ \
                    ticks[-1]/visibilities["image_pixelsize"][j]))
        else:
            xmin, xmax = int(round(visibilities["image_npix"][j]/2 + \
                    visibilities["x0"][j]/visibilities["image_pixelsize"][j]+ \
                    parameters["x0"]["value"]/visibilities["image_pixelsize"][j]+ \
                    ticks[0]/visibilities["image_pixelsize"][j])), \
                    int(round(visibilities["image_npix"][j]/2+\
                    visibilities["x0"][j]/visibilities["image_pixelsize"][j]+ \
                    parameters["x0"]["value"]/visibilities["image_pixelsize"][j]+ \
                    ticks[-1]/visibilities["image_pixelsize"][j]))
        if "y0" in params:
            ymin, ymax = int(round(visibilities["image_npix"][j]/2-\
                    visibilities["y0"][j]/visibilities["image_pixelsize"][j]- \
                    params["y0"]/visibilities["image_pixelsize"][j]+ \
                    ticks[0]/visibilities["image_pixelsize"][j])), \
                    int(round(visibilities["image_npix"][j]/2-\
                    visibilities["y0"][j]/visibilities["image_pixelsize"][j]- \
                    params["y0"]/visibilities["image_pixelsize"][j]+ \
                    ticks[-1]/visibilities["image_pixelsize"][j]))
        else:
            ymin, ymax = int(round(visibilities["image_npix"][j]/2-\
                    visibilities["y0"][j]/visibilities["image_pixelsize"][j]- \
                    parameters["y0"]["value"]/visibilities["image_pixelsize"][j]+ \
                    ticks[0]/visibilities["image_pixelsize"][j])), \
                    int(round(visibilities["image_npix"][j]/2-\
                    visibilities["y0"][j]/visibilities["image_pixelsize"][j]- \
                    parameters["y0"]["value"]/visibilities["image_pixelsize"][j]+ \
                    ticks[-1]/visibilities["image_pixelsize"][j]))

        ax[2*j,1].imshow(visibilities["image"][j].image\
                [ymin:ymax,xmin:xmax,0,0], origin="lower", \
                interpolation="nearest")

        xmin, xmax = int(round(visibilities["image_npix"][j]/2+1 + \
                ticks[0]/visibilities["image_pixelsize"][j])), \
                int(round(visibilities["image_npix"][j]/2+1 + \
                ticks[-1]/visibilities["image_pixelsize"][j]))
        ymin, ymax = int(round(visibilities["image_npix"][j]/2+1 + \
                ticks[0]/visibilities["image_pixelsize"][j])), \
                int(round(visibilities["image_npix"][j]/2+1 + \
                ticks[-1]/visibilities["image_pixelsize"][j]))

        ax[2*j,1].contour(model_image.image[ymin:ymax,xmin:xmax,0,0])

        transform = ticker.FuncFormatter(Transform(xmin, xmax, \
                visibilities["image_pixelsize"][j], '%.1f"'))

        ax[2*j,1].set_xticks(visibilities["image_npix"][j]/2+1+\
                ticks[1:-1]/visibilities["image_pixelsize"][j]-xmin)
        ax[2*j,1].set_yticks(visibilities["image_npix"][j]/2+1+\
                ticks[1:-1]/visibilities["image_pixelsize"][j]-ymin)
        ax[2*j,1].get_xaxis().set_major_formatter(transform)
        ax[2*j,1].get_yaxis().set_major_formatter(transform)

    # Plot the SED.

    for j in range(len(spectra["file"])):
        if spectra["bin?"][j]:
            ax[0,2].plot(spectra["data"][j].wave, spectra["data"][j].flux, "b-")
        else:
            ax[0,2].errorbar(spectra["data"][j].wave, spectra["data"][j].flux, \
                    fmt="bo", yerr=spectra["data"][j].unc, markeredgecolor="b")

    ax[0,2].plot(m.spectra["SED"].wave, m.spectra["SED"].flux, "g-")

    # Plot the scattered light image.

    for j in range(len(images["file"])):
        c = scale_image(images["data"][j], mode="arcsinh")

        ax[1,2].imshow(c[:,:,0,0], origin="lower", interpolation="nearest", \
                cmap="gray")

        transform3 = ticker.FuncFormatter(Transform(0, images["npix"][j], \
                images["pixelsize"][j], '%.1f"'))

        ticks = images["ticks"][j]

        ax[1,2].set_xticks(images["npix"][j]/2+ticks/images["pixelsize"][j])
        ax[1,2].set_yticks(images["npix"][j]/2+ticks/images["pixelsize"][j])
        ax[1,2].get_xaxis().set_major_formatter(transform3)
        ax[1,2].get_yaxis().set_major_formatter(transform3)

        # Create a model image to contour over the image.

        c = scale_image(m.images[images["lam"][j]], mode=images["plot_mode"][j])

        levels = numpy.array([0.05,0.25,0.45,0.65,0.85,1.0]) * \
                (c.max() - c.min()) + c.min()

        ax[1,2].contour(c[:,:,0,0], colors='gray', levels=levels)
    if len(images["file"]) == 0:
        ax[1,2].set_axis_off()

    # Adjust the plot and save it.

    ax[0,2].axis([0.1,1.0e4,1e-6,1e3])

    ax[0,2].set_xscale("log", nonposx='clip')
    ax[0,2].set_yscale("log", nonposy='clip')

    ax[0,2].set_xlabel("$\lambda$ [$\mu$]")
    ax[0,2].set_ylabel(r"$F_{\nu}$ [Jy]")

    for j in range(len(visibilities["file"])):
        ax[2*j,0].axis([1,visibilities["data1d"][j].uvdist.max()/1000*3,0,\
                visibilities["data1d"][j].amp.max()*1.1])

        ax[2*j,0].set_xscale("log", nonposx='clip')

        ax[2*j,0].set_xlabel("U-V Distance [k$\lambda$]")
        ax[2*j,0].set_ylabel("Amplitude [Jy]")

        ax[2*j,1].set_xlabel("$\Delta$RA")
        ax[2*j,1].set_ylabel("$\Delta$Dec")

        ax[2*j+1,0].set_xlabel("U [k$\lambda$]")
        ax[2*j+1,0].set_ylabel("V [k$\lambda$]")

        ax[2*j+1,1].set_xlabel("U [k$\lambda$]")
        ax[2*j+1,1].set_ylabel("V [k$\lambda$]")

    ax[1,2].set_xlabel("$\Delta$RA")
    ax[1,2].set_ylabel("$\Delta$Dec")

    for j in range(len(visibilities["file"])):
        if j > 0:
            ax[2*j,2].set_axis_off()
            ax[2*j+1,2].set_axis_off()

    fig.set_size_inches((12.5,8*len(visibilities["file"])))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.08, \
            wspace=0.25, hspace=0.2)

    # Adjust the figure and save.

    fig.savefig("model.pdf")

    plt.close(fig)

    if args.action == "plot":
        nsteps = numpy.inf

# Now we can close the pool and end the code.

if args.action == "run":
    if withmpi:
        pool.close()
