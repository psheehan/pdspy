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
parser.add_argument('-s', '--scatteredlight', action='store_true')
parser.add_argument('-e', '--withextinction', action='store_true')
parser.add_argument('-g', '--withgraindist', action='store_true')
parser.add_argument('-h', '--withhyperion', action='store_true')
parser.add_argument('-p', '--resetprob', action='store_true')
parser.add_argument('-a', '--action', type=str, default="run")
args = parser.parse_args()

source = args.object
ncpus = comm.Get_size()

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

def model(x, y, params, good=None, output="concat", npix=256, pixelsize=0.1, \
        lam="1300", scattered_light=False, with_extinction=False, dpc=140, \
        sed_lam=None, with_graindist=False):
    # Stellar parameters.

    T_star = params[0]
    L_star = 10.**params[1]

    # Disk parameters.

    M_disk = 10.**params[2]
    R_in = 10.**params[3]
    R_disk = 10.**params[4]
    h_0 = params[5]
    gamma = params[6]
    beta = params[7]
    alpha = gamma + beta

    # Envelope parameters.

    M_env = 10.**params[8]
    R_env = 10.**params[9]
    R_c = 10.**params[10]
    f_cav = params[11]
    ksi = params[12]

    # Dust parameters.

    a_max = 10.**params[13]
    if with_graindist:
        p = params[14]
    else:
        p = 3.5

    inclination = params[15]
    position_angle = params[16]

    Av = params[17]

    # Viewing parameters.

    x0 = params[18]
    y0 = params[19]
    dpc = params[20]

    # Set up the dust.

    dustopac = "pollack_new.hdf5"

    dust_gen = dust.DustGenerator(os.environ["HOME"]+\
            "/Documents/Projects/DiskMasses/Modeling/Dust/"+dustopac)

    ddust = dust_gen(a_max / 1e4, p)
    edust = dust_gen(1.0e-4, 3.5)

    # Make sure we are in a temp directory to not overwrite anything.

    original_dir = os.environ["PWD"]
    os.mkdir("/tmp/temp_{1:s}_{0:d}".format(comm.Get_rank(), source))
    os.chdir("/tmp/temp_{1:s}_{0:d}".format(comm.Get_rank(), source))

    # Write the parameters to a text file so it is easy to keep track of them.

    f = open("params.txt","w")
    for i in range(len(params)):
        f.write("params[{0:d}] = {1:f}\n".format(i, params[i]))
    f.close()

    # Set up the model and run the thermal simulation.

    if params[1] > -3.0 or 10.**params[3] < 50 or params[6] > -3.0 or \
            10.**params[7] < 500:
        if args.withhyperion:
            m = modeling.YSOModel()
            m.add_star(luminosity=L_star, temperature=T_star)
            m.set_spherical_grid(R_in, R_env, 100, 201, 2, code="hyperion")
            m.add_disk(mass=M_disk, rmin=R_in, rmax=R_disk, plrho=alpha, \
                    h0=h_0, plh=beta, dust=ddust)
            m.add_ulrich_envelope(mass=M_env, rmin=R_in, rmax=R_env, cavpl=ksi,\
                    cavrfact=f_cav, dust=edust)
            m.grid.set_wavelength_grid(0.1,1.0e5,500,log=True)

            # Run the thermal simulation.

            m.run_thermal(code="hyperion", nphot=2e5, mrw=True, pda=True, \
                    niterations=20, mpi=True, nprocesses=ncpus, verbose=False)

            # Convert model to radmc-3d format.

            m.make_hyperion_symmetric()

            m.convert_hyperion_to_radmc3d()
        else:
            m = modeling.YSOModel()
            m.add_star(luminosity=L_star, temperature=T_star)
            m.set_spherical_grid(R_in, R_env, 100, 101, 2, code="radmc3d")
            m.add_disk(mass=M_disk, rmin=R_in, rmax=R_disk, plrho=alpha, \
                    h0=h_0, plh=beta, dust=ddust)
            m.add_ulrich_envelope(mass=M_env, rmin=R_in, rmax=R_env, cavpl=ksi,\
                    cavrfact=f_cav, dust=edust)
            m.grid.set_wavelength_grid(0.1,1.0e5,500,log=True)

            # Run the thermal simulation.

            m.run_thermal(code="radmc3d", nphot=1e6, modified_random_walk=True,\
                    mrw_gamma=2, mrw_tauthres=10, mrw_count_trigger=100, \
                    verbose=False, set_threads=20)
    else:
        m = modeling.YSOModel()
        m.add_star(luminosity=L_star, temperature=T_star)
        m.set_spherical_grid(R_in, R_env, 100, 101, 2, code="radmc3d")
        m.add_disk(mass=M_disk, rmin=R_in, rmax=R_disk, plrho=alpha, h0=h_0, \
                plh=beta, dust=ddust)
        m.add_ulrich_envelope(mass=M_env, rmin=R_in, rmax=R_env, cavpl=ksi, \
                cavrfact=f_cav, dust=edust)
        m.grid.set_wavelength_grid(0.1,1.0e5,500,log=True)

        # Run the thermal simulation.

        m.run_thermal(code="radmc3d", nphot=1e6, modified_random_walk=True, \
                mrw_gamma=2, mrw_tauthres=10, mrw_count_trigger=100, \
                verbose=False)

    # Run the images/visibilities/SEDs. If output == "concat" then we are doing
    # a fit and we need less. Otherwise we are making a plot of the best fit 
    # model so we need to generate a few extra things.

    if output == "concat":
        # Run the visibilities.

        for j in range(len(visibilities["file"])):
            m.run_visibilities(name=visibilities["lam"][j], nphot=1e5, \
                    npix=visibilities["npix"][j], \
                    pixelsize=visibilities["pixelsize"][j], \
                    lam=visibilities["lam"][j], incl=inclination, \
                    pa=position_angle, dpc=dpc, code="radmc3d", \
                    mc_scat_maxtauabs=5, verbose=False)

        # Run the images.

        for j in range(len(images["file"])):
            m.run_image(name=images["lam"][j], nphot=1e5, \
                    npix=images["npix"][j], pixelsize=images["pixelsize"][j], \
                    lam=images["lam"][j], incl=inclination, \
                    pa=position_angle, dpc=dpc, code="radmc3d", \
                    mc_scat_maxtauabs=5, verbose=False)

        # Run the SED.

        m.set_camera_wavelength(sed_lam)

        m.run_sed(name="SED", nphot=1e4, loadlambda=True, incl=inclination, \
                pa=position_angle, dpc=dpc, code="radmc3d", \
                camera_scatsrc_allfreq=True, mc_scat_maxtauabs=5, \
                verbose=False)

        if with_extinction:
            m.spectra["SED"].flux = dust.redden(m.spectra["SED"].wave, \
                    m.spectra["SED"].flux, Av, law="mcclure")

        # Concatenate everything in the right way.

        if scattered_light:
            z_model = numpy.concatenate((m.images["scattered_light"].\
                    image.reshape((128**2,)), numpy.log10(m.spectra["SED"].\
                    flux)))
        else:
            z_model = numpy.log10(m.spectra["SED"].flux)
        for j in range(len(lam)):
            z_model = numpy.concatenate((z_model, \
                    m.visibilities[lam[j]].real[:,0], \
                    m.visibilities[lam[j]].imag[:,0]))

        # Clean up everything and return.

        os.system("rm params.txt")
        os.chdir(original_dir)
        os.rmdir("/tmp/temp_{1:s}_{0:d}".format(comm.Get_rank(), source))

        return z_model[good]
    else:
        # Run the high resolution visibilities.

        for j in range(len(visibilities["file"])):
            # Run a high resolution version of the visibilities.

            m.run_visibilities(name=visibilities["lam"][j], nphot=1e5, \
                    npix=2048, pixelsize=0.05, lam=lam[j], incl=inclination, \
                    pa=position_angle, dpc=dpc, code="radmc3d", \
                    mc_scat_maxtauabs=5, verbose=False)

            # Run the visibilities they were done for the fit to show in 2D

            m.run_visibilities(name=visibilities["lam"][j]+"2D", nphot=1e5, \
                    npix=visibilities["npix"][j], \
                    pixelsize=visibilities["pixelsize"][j], \
                    lam=visibilities["lam"][j], incl=inclination, \
                    pa=-position_angle, dpc=dpc, code="radmc3d", \
                    mc_scat_maxtauabs=5, verbose=False)

            # Run a millimeter image.

            m.run_image(name=visibilities["lam"][j], nphot=1e5, \
                    npix=visibilities["image_npix"][j], \
                    pixelsize=visibilities["image_pixelsize"][j], \
                    lam=visibilities["lam"][j], incl=inclination, \
                    pa=position_angle, dpc=dpc, code="radmc3d", \
                    mc_scat_maxtauabs=5, verbose=False)

        # Run the scattered light image. 

        for j in range(len(images["file"])):
            m.run_image(name=images["lam"][j], nphot=1e5, \
                    npix=images["npix"][j], pixelsize=images["pixelsize"][j], \
                    lam=images["lam"][j], incl=inclination, pa=position_angle, \
                    dpc=dpc, code="radmc3d", mc_scat_maxtauabs=5, verbose=False)

        # Run the SED

        m.set_camera_wavelength(numpy.logspace(-1,4,50))

        m.run_sed(name="SED", nphot=1e4, loadlambda=True, incl=inclination, \
                pa=position_angle, dpc=dpc, code="radmc3d", \
                camera_scatsrc_allfreq=True, mc_scat_maxtauabs=5, \
                verbose=False)

        if with_extinction:
            m.spectra["SED"].flux = dust.redden(m.spectra["SED"].wave, \
                    m.spectra["SED"].flux, Av, law="mcclure")

        # Clean up everything and return.

        os.system("rm params.txt")
        os.chdir(original_dir)
        os.rmdir("/tmp/temp_{1:s}_{0:d}".format(comm.Get_rank(), source))

        return m

# Define a likelihood function.

def lnlike(p, x, y, z, zerr, good, output, npix, pixelsize, lam, \
        scattered_light, with_extinction, dpc, sed_lam, with_graindist):
    m = model(x, y, p, good, output, npix, pixelsize, lam, scattered_light, \
            with_extinction, dpc, sed_lam, with_graindist)

    return -0.5*(numpy.sum((z - m)**2 / zerr**2))

def lnprior(p):
    if -1 < p[0] < 1.3 and p[1] <= -2.5 and \
            0.1 <= 10.**p[2] < 10.**p[3] < 10.**p[7] and \
            0.01 <= p[4] and -0.5 <= p[5] <= 2 and p[6] < -2.0 and \
            0.0 <= p[8] <= 1.0 and 0.5 <= p[9] <= 1.5 and \
            0.0 <= p[10] <= 90. and 0. <= p[11] <= 180. and \
            0. <= p[12] <= 5. and 0.5 <= p[13] <= 1.5 and \
            0. <= p[14] <= 2. and 2.5 <= p[15] <= 4.5:
        return 0.0

    return -numpy.inf

def lnprob(p, x, y, z, zerr, good, output, npix, pixelsize, lam, \
        scattered_light, with_extinction, dpc, sed_lam, with_graindist):
    lp = lnprior(p)

    if not numpy.isfinite(lp):
        return -numpy.inf

    return lp + lnlike(p, x, y, z, zerr, good, output, npix, pixelsize, lam, \
            scattered_light, with_extinction, dpc, sed_lam, with_graindist)

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
# Set up a pool for parallel runs.
#
################################################################################

if args.action == "run":
    pool = emcee.utils.MPIPool(loadbalance=True)

    if not pool.is_master():
        pool.wait()
        sys.exit(0)

################################################################################
#
# In case we are restarting this from the same job submission, delete any
# temporary directories associated with this run.
#
################################################################################

os.system("rm -r /tmp/temp_{0:s}_*".format(source))

################################################################################
#
# Read in the data.
#
################################################################################

# Define some useful values depending on the source. => now in config.py

"""OLD:
if source in ['I04016','I04108B','I04158','I04166','I04169','I04181A', \
        'I04181B','I04263','I04295','I04302','I04365']:
    binsize = [8057.218995847603]
    pixelsize = [0.1]
    freq = ["230GHz"]
    lam = ["1300"]
    npix = [256]
    gridsize = npix
    dpc = 140
elif source in ['CRBR12','Elias21','Elias29','GSS30-IRS3','GY91','IRS48',\
        'IRS63','LFAM26','WL12','WL17']:
    binsize = [26857.396652825348,40286.09497923802]
    pixelsize = [0.03,0.02]
    freq = ["345GHz","100GHz"]
    lam = ["870","3100"]
    npix = [256,256]
    gridsize = npix
    dpc = 137
"""

# Import the configuration file information.

from config import *

# Read in centroid parameters. => specify in config.py?

"""
f = open("{0:s}/{0:s}_gaussian_fit.txt".format(source),"r")
lines = f.readlines()
f.close()

params = {}

for j in range(len(freq)):
    key = lines[9*j].split(' ')[-1].split(':')[0]

    x0 = float(lines[9*j+2].split(' ')[2])
    y0 = float(lines[9*j+3].split(' ')[2])

    params[key] = [x0, y0, 1.]

if source in ['I04181B']:
    params['230GHz'] = [0.,0.,1.]
"""

# Read in the millimeter visibilities.

visibilities["data"] = []
visibilities["data1d"] = []
visibilities["image"] = []
images["data"] = []

for j in range(len(visibilities["file"])):
    # Read the raw data.

    data = uv.Visibilities()
    data.read(visiblities["file"][j])

    # Center the data. => need to update!

    data = uv.center(data, params[freq[j]])

    # Average the data to a more manageable size.

    visibilities["data"].append(uv.grid(data, \
            gridsize=visibilities["gridsize"][j], \
            binsize=visibilities["binsize"][j]))

    # Scale the weights of the visibilities to force them to be fit well.

    visibilities["data"][j].weights *= visibilities["weight"][j]

    # Average the visibilities radially.

    visibilities["data1d"].append(uv.average(data, gridsize=20, radial=True, \
            log=True, logmin=data.uvdist[numpy.nonzero(data.uvdist)].min()*\
            0.95, logmax=data.uvdist.max()*1.05)

    # Clean up the data because we don't need it any more.

    del data

    # Read in the image => need to update.

    visibilities["image"][j] = im.readimfits(visibilities["image_file"][j])

# Read in the SEDs

sed = sp.Spectrum()
sed.read("../Data/{0:s}/Literature/{0:s}_SED.hdf5".format(source))

sed_av = sp.Spectrum()
sed_av.read("../Data/{0:s}/Literature/{0:s}_SED_averaged.hdf5".format(source))

sed_log = sp.Spectrum(sed_av.wave, numpy.log10(sed_av.flux), \
        sed_av.unc/sed_av.flux)
sed_log.flux[sed_log.flux == -numpy.inf] == 0.

# Read in the Spitzer spectrum.

if os.path.exists("../Data/{0:s}/IRS/{0:s}_IRS_spectrum.hdf5".format(source)):
    spitzer = sp.Spectrum()
    spitzer.read("../Data/{0:s}/IRS/{0:s}_IRS_spectrum.hdf5".format(source))

    # Merge the SED with the binned Spitzer spectrum.

    wave = numpy.linspace(5., 35., 25)
    flux = numpy.interp(wave, spitzer.wave, spitzer.flux)

    sed_log.wave = numpy.concatenate((sed_log.wave, wave))
    sed_log.flux = numpy.concatenate((sed_log.flux, numpy.log10(flux)))
    sed_log.unc = numpy.concatenate((sed_log.unc, numpy.repeat(0.1, wave.size)))

    order = numpy.argsort(sed_log.wave)

    sed_log.wave = sed_log.wave[order]
    sed_log.flux = sed_log.flux[order]
    sed_log.unc = sed_log.unc[order]
else:
    spitzer = None

# Adjust the weight of the SED, as necessary.

"""OLD:
if source in ['IRS63','LFAM26']:
    sed_log.unc /= 3.
elif source in ['WL12']:
    sed_log.unc /= 10.
"""

# Read in the images.

"""OLD:
if os.path.exists("../Data/{0:s}/HST/{0:s}_scattered_light.hdf5".\
        format(source)):
    scattered_light = im.Image()
    scattered_light.read("../Data/{0:s}/HST/{0:s}_scattered_light.hdf5".\
            format(source))
else:
    if args.scatteredlight:
        print("No scattered light image available... Exiting.")
        sys.exit(0)
"""

for j in range(len(images["file"])):
    images["data"].append(im.Image())
    images["data"][j].read(images["file"][j])

################################################################################
#
# Fit the model to the data.
#
################################################################################

# Set up the inputs for the MCMC function.

x = vis[lam[0]].u
y = vis[lam[0]].v
if args.scatteredlight:
    z = numpy.concatenate((scattered_light.image.reshape((128**2,)), \
            sed_log.flux))
    zerr = numpy.concatenate((scattered_light.image.reshape((128**2,)), \
            sed_log.unc))
else:
    z = sed_log.flux
    zerr = sed_log.unc

for j in range(len(lam)):
    z = numpy.concatenate((z, vis[lam[j]].real[:,0], vis[lam[j]].imag[:,0]))
    zerr = numpy.concatenate((zerr, 1./numpy.sqrt(vis[lam[j]].weights[:,0]), \
            1./numpy.sqrt(vis[lam[j]].weights[:,0])))

good = numpy.isfinite(zerr)
z = z[good]
zerr = zerr[good]

# Set up the emcee run.

ndim, nwalkers = 16, 200

# If we are resuming an MCMC simulation, read in the necessary info, otherwise
# set up the info.

if args.resume:
    pos = numpy.load("{0:s}/pos.npy".format(source))
    chain = numpy.load("{0:s}/chain.npy".format(source))
    state = None
    nsteps = chain[0,:,0].size

    if args.resetprob:
        prob = None
    else:
        prob = numpy.load("{0:s}/prob.npy".format(source))

    # If we started running without p, make sure we add the necessary stuff.

    if chain.shape[2] == 15:
        extra = numpy.random.uniform(2.5, 4.5, nwalkers*nsteps).\
                reshape((nwalkers, nsteps, 1))

        chain = numpy.concatenate((chain, extra), axis=2)
        pos = chain[:,-1,:]
else:
    pos = []
    for i in range(nwalkers):
        r_env = numpy.random.uniform(2.5,4.,1)[0]
        r_disk = numpy.random.uniform(numpy.log10(30.),\
                numpy.log10(10.**r_env),1)[0]
        r_in = numpy.random.uniform(numpy.log10(0.1),\
                numpy.log10((10.**r_disk)/2),1)[0]

        if source in ['I04166','I04169','I04302','CRBR12','IRS63','LFAM26']:
            pa = numpy.random.uniform(0.,180.,1)[0]
        elif source in ['I04016','GSS30-IRS3']:
            pa = numpy.random.uniform(90.,270.,1)[0]
        else:
            pa = numpy.random.uniform(-90.,90.,1)[0]

        pos.append([numpy.random.uniform(-1., 1., 1)[0], \
                numpy.random.uniform(-5., -3., 1)[0], \
                r_in, r_disk, \
                numpy.random.uniform(0.01,0.15,1)[0], \
                numpy.random.uniform(-0.5,2.0,1)[0], \
                numpy.random.uniform(-5., -3., 1)[0], \
                r_env, \
                numpy.random.uniform(0, 1., 1)[0], \
                numpy.random.uniform(0.5, 1.5, 1)[0], \
                numpy.random.uniform(0, 90., 1)[0], \
                pa, \
                numpy.random.uniform(0., 5., 1)[0], \
                numpy.random.uniform(0.5, 1.5, 1)[0], \
                numpy.random.uniform(0., 2., 1)[0], \
                numpy.random.uniform(2.5, 4.5, 1)[0]])
    prob = None
    chain = numpy.empty((nwalkers, 0, ndim))
    state = None
    nsteps = 0

# Set up the MCMC simulation.

if args.action == "run":
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, \
            args=(x, y, z, zerr, good, "concat", npix, pixelsize, lam, \
            args.scatteredlight, args.withextinction, dpc, sed_log.wave, \
            args.withgraindist), pool=pool)

# Run a few burner steps.

while nsteps < 100000:
    if args.action == "run":
        pos, prob, state = sampler.run_mcmc(pos, 5, lnprob0=prob, rstate0=state)

        chain = numpy.concatenate((chain, sampler.chain), axis=1)

        # Plot the steps of the walkers.

        for i in range(ndim):
            fig, ax = plt.subplots(nrows=1, ncols=1)

            for j in range(nwalkers):
                ax.plot(chain[j,:,i])

            plt.savefig("{1:s}/disk_test_{0:d}.pdf".format(i, source))

            plt.close(fig)

        # Save walker positions in case the code stps running for some reason.

        numpy.save("{0:s}/pos".format(source), pos)
        numpy.save("{0:s}/prob".format(source), prob)
        numpy.save("{0:s}/chain".format(source), chain)

        # Augment the nuber of steps and reset the sampler for the next run.

        nsteps += 5

        sampler.reset()

    # Get the best fit parameters and uncertainties from the last 10 steps.

    samples = chain[:,-5:,:].reshape((-1, ndim))

    params = numpy.median(samples, axis=0)
    sigma = samples.std(axis=0)

    # Fix the position angle.

    params[11] = -params[11]

    # When plotting, change parameter values here.

    if args.action == "plot":
        pass

    # Print out the status of the fit.

    if args.action == "run":
        # Write out the results.

        f = open("{0:s}/{0:s}_disk_fit.txt".format(source), "w")
        f.write("Best fit to {0:s}:\n\n".format(source))
        f.write("Lstar = {0:f} +/- {1:f}\n".format(params[0], sigma[0]))
        f.write("log10(Mdisk) = {0:f} +/- {1:f}\n".format(params[1], sigma[1]))
        f.write("Rin = {0:f} +/- {1:f}\n".format(params[2], sigma[2]))
        f.write("Rdisk = {0:f} +/- {1:f}\n".format(params[3], sigma[3]))
        f.write("h0 = {0:f} +/- {1:f}\n".format(params[4], sigma[4]))
        f.write("gamma = {0:f} +/- {1:f}\n".format(params[5], sigma[5]))
        f.write("beta = {0:f} +/- {1:f}\n".format(params[13], sigma[13]))
        f.write("log10(Menv) = {0:f} +/- {1:f}\n".format(params[6], sigma[6]))
        f.write("Renv = {0:f} +/- {1:f}\n".format(params[7], sigma[7]))
        f.write("fcav = {0:f} +/- {1:f}\n".format(params[8], sigma[8]))
        f.write("ksi = {0:f} +/- {1:f}\n".format(params[9], sigma[9]))
        f.write("i = {0:f} +/- {1:f}\n".format(params[10], sigma[10]))
        f.write("pa = {0:f} +/- {1:f}\n".format(params[11], sigma[11]))
        f.write("log10(a_max) = {0:f} +/- {1:f}\n".format(params[12],sigma[12]))
        f.write("Av = {0:f} +/- {1:f}\n\n".format(params[14], sigma[14]))
        f.write("p = {0:f} +/- {1:f}\n\n".format(params[15], sigma[15]))
        f.close()

        os.system("cat {0:s}/{0:s}_disk_fit.txt".format(source))

        # Plot histograms of the resulting parameters.

        xlabels = ["$L_{star}$","$log_{10}(M_{disk})$","$R_{in}$","$R_{disk}$",\
                "$h_0$", "$\gamma$", "$log_{10}(M_{env})$", "$R_{env}$", \
                "$f_{cav}$", r"$\xi$", "$i$", "p.a.", "$log_{10}(a_{max})$", \
                r"$\beta$", "$A_v$", "p"]

        fig = corner.corner(samples, labels=xlabels, truths=params)

        plt.savefig("{0:s}/{0:s}_disk_fit.pdf".format(source))

    ############################################################################
    #
    # Plot the results.
    #
    ############################################################################

    # Plot the best fit model over the data.

    fig, ax = plt.subplots(nrows=2*len(lam), ncols=3)

    # Create a high resolution model for averaging.

    N = [image[lam[j]].image.shape[0] for j in range(len(lam))]
    dx = [image[lam[j]].header["CDELT2"] * numpy.pi / 180. / arcsec for j in \
            range(len(lam))]

    m = model(1, 1, params, output="data", npix=N, pixelsize=dx, \
            lam=lam, with_extinction=args.withextinction)

    # Plot the millimeter data/models.

    for j in range(len(lam)):
        # Create a high resolution model for averaging.

        m1d = uv.average(m.visibilities[lam[j]], gridsize=10000, binsize=3500, \
                radial=True)

        # Plot the visibilities.

        ax[2*j,0].errorbar(data_1d[lam[j]].uvdist/1000, data_1d[lam[j]].amp, \
                yerr=numpy.sqrt(1./data_1d[lam[j]].weights),\
                fmt="bo", markersize=8, markeredgecolor="b")

        # Plot the best fit model

        ax[2*j,0].plot(m1d.uvdist/1000, m1d.amp, "g-")

        # Plot the 2D visibilities.

        if source in ['I04016','I04108B','I04158','I04166','I04169','I04181A', \
                'I04181B','I04263','I04295','I04302','I04365']:
            ticks = numpy.array([-250,-200,-100,0,100,200,250])
        elif source in ['CRBR12','Elias21','Elias29','GSS30-IRS3','GY91', \
                'IRS63','LFAM26','WL12','WL17']:
            ticks = numpy.array([-1500,-1000,-500,0,500,1000,1500])

        xmin, xmax = int(npix[j]/2+ticks[0]/(binsize[j]/1000)), \
                int(npix[j]/2+ticks[6]/(binsize[j]/1000))
        ymin, ymax = int(npix[j]/2+ticks[0]/(binsize[j]/1000)), \
                int(npix[j]/2+ticks[6]/(binsize[j]/1000))

        vmin = min(0, data_1d[lam[j]].real.min())
        vmax = data_1d[lam[j]].real.max()

        ax[2*j+1,0].imshow(vis[lam[j]].real.reshape((npix[j],npix[j]))\
                [xmin:xmax,xmin:xmax][:,::-1], origin="lower", \
                interpolation="nearest", vmin=vmin, vmax=vmax)
        ax[2*j+1,0].contour(m.visibilities[lam[j]+"2D"].real.reshape(\
                (npix[j],npix[j]))[xmin:xmax,xmin:xmax][:,::-1])

        vmin = -data_1d[lam[j]].real.max()
        vmax = data_1d[lam[j]].real.max()

        ax[2*j+1,1].imshow(vis[lam[j]].imag.reshape((npix[j],npix[j]))\
                [xmin:xmax,xmin:xmax][:,::-1], origin="lower", \
                interpolation="nearest", vmin=vmin, vmax=vmax)
        ax[2*j+1,1].contour(m.visibilities[lam[j]+"2D"].imag.reshape(\
                (npix[j],npix[j]))[xmin:xmax,xmin:xmax][:,::-1], linewidth=0.2)

        transform1 = ticker.FuncFormatter(Transform(xmin, xmax, \
                binsize[j]/1000, '%.0f'))

        ax[2*j+1,0].set_xticks(npix[j]/2+ticks[1:-1]/(binsize[j]/1000)-xmin)
        ax[2*j+1,0].set_yticks(npix[j]/2+ticks[1:-1]/(binsize[j]/1000)-ymin)
        ax[2*j+1,0].get_xaxis().set_major_formatter(transform1)
        ax[2*j+1,0].get_yaxis().set_major_formatter(transform1)

        ax[2*j+1,1].set_xticks(npix[j]/2+ticks[1:-1]/(binsize[j]/1000)-xmin)
        ax[2*j+1,1].set_yticks(npix[j]/2+ticks[1:-1]/(binsize[j]/1000)-ymin)
        ax[2*j+1,1].get_xaxis().set_major_formatter(transform1)
        ax[2*j+1,1].get_yaxis().set_major_formatter(transform1)

        # Create a model image to contour over the image.

        model_image = m.images[lam[j]]

        x, y = numpy.meshgrid(numpy.linspace(-256,255,512), \
                numpy.linspace(-256,255,512))

        beam = misc.gaussian2d(x, y, 0., 0.,image[lam[j]].header["BMAJ"]/2.355/\
                image[lam[j]].header["CDELT2"], \
                image[lam[j]].header["BMIN"]/2.355/\
                image[lam[j]].header["CDELT2"], \
                (90-image[lam[j]].header["BPA"])*numpy.pi/180., 1.0)

        model_image.image = scipy.signal.fftconvolve(\
                model_image.image[:,:,0,0], beam, mode="same").\
                reshape(model_image.image.shape)

        # Plot the image.

        if source in ['I04158']:
            ticks = numpy.array([-6.0,-5.0,-2.5,0,2.5,5.0,6.0])
        elif source in ['I04166','I04169']:
            ticks = numpy.array([-2.5,-2.0,-1.0,0,1.0,2.0,2.5])
        elif source in ['I04181A','I04181B','I04295','I04365']:
            ticks = numpy.array([-6.0,-5.0,-2.5,0,2.5,5.0,6.0])
        elif source in ['CRBR12','Elias21','Elias29','GSS30-IRS3','GY91', \
                'IRS48','IRS63','LFAM26','WL12','WL17']:
            ticks = numpy.array([-0.75,-0.6,-0.3,0,0.3,0.6,0.75])
        else:
            ticks = numpy.array([-4.5,-4.0,-2.0,0,2.0,4.0,4.5])

        xmin, xmax = int(N[j]/2+round(ticks[0]/dx[j])), \
                round(N[j]/2+int(ticks[6]/dx[j]))
        ymin, ymax = int(N[j]/2+round(ticks[0]/dx[j])), \
                round(N[j]/2+int(ticks[6]/dx[j]))

        ax[2*j,1].imshow(image[lam[j]].image[xmin:xmax,ymin:ymax,0,0], \
                origin="lower", interpolation="nearest")

        ax[2*j,1].contour(model_image.image[xmin:xmax,ymin:ymax,0,0])

        transform = ticker.FuncFormatter(Transform(xmin, xmax, dx[j], '%.1f"'))

        ax[2*j,1].set_xticks(N[j]/2+ticks[1:-1]/dx[j]-xmin)
        ax[2*j,1].set_yticks(N[j]/2+ticks[1:-1]/dx[j]-ymin)
        ax[2*j,1].get_xaxis().set_major_formatter(transform)
        ax[2*j,1].get_yaxis().set_major_formatter(transform)

    # Plot the SED.

    ax[0,2].errorbar(sed.wave, sed.flux, fmt="bo", yerr=sed.unc, \
            markeredgecolor="b")

    ax[0,2].plot(m.spectra["SED"].wave, m.spectra["SED"].flux, "g-")

    if spitzer != None:
        ax[0,2].plot(spitzer.wave, spitzer.flux, "b-")

    # Plot the scattered light image.

    if os.path.exists("../Data/{0:s}/HST/{0:s}_scattered_light.hdf5".\
            format(source)):

        vmin = scattered_light.image.min()
        vmax = numpy.percentile(scattered_light.image, 95)
        a = 1000.
        
        b = (scattered_light.image - vmin) / (vmax - vmin)

        c = numpy.arcsinh(10*b)/3.

        ax[1,2].imshow(c[:,:,0,0], origin="lower", interpolation="nearest", \
                cmap="gray")

        transform3 = ticker.FuncFormatter(Transform(0, 128, 0.1, '%.1f"'))

        ticks = numpy.array([-6,-3,0,3,6])

        ax[1,2].set_xticks(128/2+ticks/0.1)
        ax[1,2].set_yticks(128/2+ticks/0.1)
        ax[1,2].get_xaxis().set_major_formatter(transform3)
        ax[1,2].get_yaxis().set_major_formatter(transform3)

        # Create a model image to contour over the image.

        model_image = m.images["scattered_light"]

        beam = misc.gaussian2d(x, y, 0., 0., 1., 1., 0., 1.0)

        model_image.image = scipy.signal.fftconvolve(\
                model_image.image[:,:,0,0], beam, mode="same").\
                reshape(model_image.image.shape)

        vmin = model_image.image.min()
        vmax = model_image.image.max()
        a = 1000.
        
        b = (model_image.image - vmin) / (vmax - vmin)

        if source in ['I04158','I04295']:
            # Log scaling.
            c = numpy.log10(a*b+1)/numpy.log10(a)
        else:
            # Linear scaling.
            c = b

        levels = numpy.array([0.05,0.25,0.45,0.65,0.85,1.0]) * \
                (c.max() - c.min()) + c.min()

        ax[1,2].contour(c[:,:,0,0], colors='gray', levels=levels)
    else:
        ax[1,2].set_axis_off()

    # Adjust the plot and save it.

    ax[0,2].axis([0.1,1.0e4,1e-6,1e3])

    ax[0,2].set_xscale("log", nonposx='clip')
    ax[0,2].set_yscale("log", nonposx='clip')

    ax[0,2].set_xlabel("$\lambda$ [$\mu$]")
    ax[0,2].set_ylabel(r"$F_{\nu}$ [Jy]")

    for j in range(len(lam)):
        ax[2*j,0].axis([1,data_1d[lam[j]].uvdist.max()/1000*3,0,\
                data_1d[lam[j]].amp.max()*1.1])

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

    for j in range(len(lam)):
        if j > 0:
            ax[2*j,2].set_axis_off()
            ax[2*j+1,2].set_axis_off()

    fig.set_size_inches((12.5,8*len(lam)))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.08, \
            wspace=0.25, hspace=0.2)

    # Adjust the figure and save.

    fig.savefig("{0:s}/{0:s}_disk_model.pdf".format(source))

    plt.close(fig)

    if args.action == "plot":
        nsteps = numpy.inf

# Now we can close the pool and end the code.

if args.action == "run":
    pool.close()
