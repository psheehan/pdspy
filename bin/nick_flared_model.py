#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
from pdspy.constants.physics import c, m_p, k, G
from pdspy.constants.astronomy import M_sun, AU
import pdspy.interferometry as uv
import pdspy.spectroscopy as sp
import pdspy.modeling as modeling
import pdspy.modeling.mpi_pool as largepool
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
_TIME_ = str(time.time())
import sys
import os
import emcee
import corner
from mpi4py import MPI
from pdspy.misc import logger
from pdspy.misc.colours import colours
from pdspy import version as pdspyv
pdspyv.assertion()

comm = MPI.COMM_WORLD

################################################################################
#
# Create a function which returns a model of the data.
#
################################################################################

def model(x, y, lam, params,info, output="concat"):
    save_dir=info.svd
    source_name=info.source
    # params[M_star,Mdisk,Rdisk,temp,]
    M_star = 10.**params[0]
    T_star = 4300.
    R_star = 1.7
    L_star = 1.0

    M_disk = 10.**params[1]
    R_in = 0.1
    R_disk = 10.**params[2]
    gamma = 1.

    t0 = 10.**params[3]
    q = params[4]

    t_rdisk = t0 * (R_disk / 1.)**-q
    h_0 = ((k*(R_disk*AU)**3*t_rdisk) / (G*M_star*M_sun * 2.37*m_p))**0.5 / AU
    beta = 0.5 * (3 - q)
    alpha = gamma + beta

    a_turb = 10.**params[5]

    v_sys = params[6] * 1.0e5
    b = v_sys / c
    wave = lam * numpy.sqrt((1. - b) / (1. + b))

    x0 = params[9]
    y0 = params[10]

    dpc = params[11]

    dustopac = "draine_3mm.hdf5"

    ddust = dust.Dust()
    ddust.set_properties_from_file(work_dir+"Dust/"+dustopac)

    gases = []
    abundance = []

    co = gas.Gas()
    co.set_properties_from_lambda(work_dir+"Gas/c17o.dat")
    gases.append(co)
    abundance.append(1.5e-4)

    inclination = params[7]
    position_angle = params[8]

    original_dir = os.environ["PWD"]
    os.mkdir("/tmp/temp_"+source_name+"_{0:d}".format(comm.Get_rank()))
    os.chdir("/tmp/temp_"+source_name+"_{0:d}".format(comm.Get_rank()))

    f = open("params.txt","w")
    for i in range(len(params)):
        f.write("params[{0:d}] = {1:f}\n".format(i, params[i]))
    f.close()
    
    m = modeling.YSOModel()
    m.add_star(mass=M_star, luminosity=L_star, temperature=T_star)
    m.set_spherical_grid(R_in, max(5*R_disk,300), 100, 51, 2, code="radmc3d")
    m.add_pringle_disk(mass=M_disk, rmin=R_in, rmax=R_disk, plrho=alpha,h0=h_0,\
            plh=beta, dust=ddust, t0=t0, plt=q, gas=gases, abundance=abundance,\
            aturb=a_turb)
    m.grid.set_wavelength_grid(0.1,1.0e5,500,log=True)

    # Run the images/visibilities/SEDs.
    verbosity = False

    if output == "concat":
        m.set_camera_wavelength(wave)

        m.run_visibilities(name="C17O", nphot=1e5, npix=1024, lam=None, \
                pixelsize=0.01, tgas_eq_tdust=True, scattering_mode_max=0, \
                incl_dust=False, incl_lines=True, loadlambda=True, \
                incl=inclination, pa=position_angle, dpc=230, code="radmc3d",\
                verbose=verbosity,writeimage_unformatted=True)

        m.visibilities["C17O"] = uv.center(m.visibilities["C17O"], \
                [x0, y0, 1])

        model_vis = uv.interpolate_model(x, y, c / lam / 1.0e-4, \
                m.visibilities["C17O"])

        z_model = numpy.concatenate((model_vis.real, model_vis.imag))

        os.chdir(original_dir)
        os.rmdir("/tmp/temp_"+source_name+"_{0:d}".format(comm.Get_rank()))

        return z_model
    else:
        m.set_camera_wavelength(wave)

        """
        m.run_image(name="CO2-1", nphot=1e5, npix=1024, lam=None, \
                pixelsize=0.01, tgas_eq_tdust=True, scattering_mode_max=0, \
                incl_dust=False, incl_lines=True, loadlambda=True, \
                incl=inclination, pa=position_angle, dpc=120, code="radmc3d", \
                verbose=False)
        """

        m.run_visibilities(name="C17O", nphot=1e5, npix=1024, lam=None, \
                pixelsize=0.01, tgas_eq_tdust=True, scattering_mode_max=0, \
                incl_dust=False, incl_lines=True, loadlambda=True, \
                incl=inclination, pa=position_angle, dpc=230, code="radmc3d",\
                verbose=verbosity)

        os.chdir(original_dir)
        os.system("rmdir /tmp/temp_"+source_name+"_{0:d}".format(comm.Get_rank()))

        return m

# Define a likelihood function.

def lnlike(p, x, y, lam, z, zerr,info):

    m = model(x, y, lam, p,info)
    return -0.5*(numpy.sum((z - m)**2 / zerr**2))

def lnprior(p):
    if p[1] < 0 and 0. <= p[2] and 0. < p[3] and 0. <= p[4] < 1.0 and \
            0.0 <= p[7] <= 180. and 0. <= p[8] <= 360. and 0. < p[11]:
        return 0.0

    return -numpy.inf

def lnprob(p, x, y, lam, z, zerr,info):
    lp = lnprior(p)

    if not numpy.isfinite(lp):
        return -numpy.inf

    return lp + lnlike(p, x, y, lam, z, zerr,info)

class Info:
    def __init__(self, work_dir,save,input_file,rest_freq,source_name):
        self.wrk = work_dir
        self.infile = input_file
        self.rf = rest_freq
        self.source = source_name
        self.svd= save

# main function
if __name__ == "__main__":
    ################################################################################
    #
    # Parse command line arguments.
    #
    ################################################################################
    description = 'MCMC Modelling code created by Patrick Sheehan (OU)'\
                  'Version: ' + __version__

    resume_help = 'resumes the job, need to have the saved files pos,chain,prob'
    resetprob_help = 'Resets the probability'
    action_help = 'Allows to either running (default) or plotting(resume) actions to be taken'
    save_help = 'Allows for saving to alternate directory, end it with /'
    infile_help='Name of the input file'
    source_help="Name of source,no spaces"
    freq_help = "Input frequency in Hz"
    log_help = "Input logname"
    v_help    = 'Integer 1-5 of verbosity level'

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file',type=str,help=infile_help,dest="infile")
    parser.add_argument('-r', '--resume', default=False,help=resume_help)
    parser.add_argument('-p', '--resetprob', action='store_true',help=resetprob_help)
    parser.add_argument('-a', '--action', type=str, default="run",help=action_help)
    parser.add_argument('-s', '--save-dir',type=str, default=os.environ['HOME']+'/radmc3d/',help=save_help,dest='save')
    parser.add_argument('-o', '--source',default=False,help=source_help,dest='source')
    parser.add_argument('-rf', '--rest-freq',default=False,help=freq_help,dest='restfreq')
    parser.add_argument('-l', '--logfile',help=log_help,dest='logger')
    parser.add_argument('-v','--verbosity', help=v_help,default=2,dest='verb',type=int)
    args = parser.parse_args()

    logfile = args.logger
    verbosity = args.verb

    # Set up message logger            
    if not logfile:
        logfile = ('{}_{}.log'.format(__file__[:-3],time.time()))
    if verbosity >= 3:
        logger = logger.Messenger(verbosity=verbosity, add_timestamp=True,logfile=logfile)
    else:
        logger = logger.Messenger(verbosity=verbosity, add_timestamp=False,logfile=logfile)
    logger.header1("Starting {}....".format(__file__[:-3]))

    logger.header2('This program will create and remove numerous temporary files for debugging.')

    actionlist = ['run','plot','runlarge']

    logger.header1('Selected action :{}'.format(args.action))

    if args.action not in actionlist:
        logger.message("Please select a valid action: {}".format(actionlist))
        sys.exit(0)

    if args.action == 'plot':
        args.resume = True

    source_name = args.source
    rest_freq = args.restfreq

    if not source_name:
        if PY == 2:
            source_name = raw_input("Input source name (no spaces):")
        elif PY == 3:
            source_name = input("Input source name (no spaces):")
        if source_name == "":
            source_name = time.time()

    while not rest_freq or rest_freq == "":
        try:
            if PY == 2:
                rest_freq = raw_input("Input rest_frequency (Hz):")
            elif PY == 3:
                rest_freq = input("Input rest_frequency (Hz):")
            rest_freq=float(rest_freq)
            break
        except ValueError:
            logger.message("Input numbers only")
            continue
    if rest_freq != "":
        rest_freq = float(rest_freq)

    save_dir = args.save + '/' + source_name+'/'
    work_dir = args.save + '/'
    input_file = args.infile

    if not os.path.isdir(save_dir):
        try:
            os.makedirs(save_dir)
        except FileExistsError:
            pass

    info=Info(work_dir,save_dir,input_file,rest_freq,source_name)

    ################################################################################
    #
    # In case we are restarting this from the same job submission, delete any
    # temporary directories associated with this run.
    #
    ################################################################################

    os.system("rm -rf /tmp/temp_"+source_name+"_*")

    ################################################################################
    #
    # Set up a pool for parallel runs.
    #
    ################################################################################

    if args.action == "run":
        pool = emcee.utils.MPIPool()

        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    elif args.action == "runlarge":
        pool = largepool.MPIPool(largedata=True)

        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    ################################################################################
    #
    # Read in the data.
    #
    ################################################################################

    # Read in the visibilities.

    logger.message("Input file read: {}".format(work_dir+input_file))

    import glob

    if glob.glob(work_dir+input_file):
        logger.message('Found file')
    else:
        logger.message('Failed file')

    logger.message("Reading in the data...")
    try:
        data = uv.readvis(work_dir+input_file)
    except IOError:
        logger.message("Failed read in...")
        sys.exit(0)
    logger.message("Data read in...")

    # Read in the image

    #image = im.readimfits("../Q4OU01/UT171012/Per-emb-50.0001.fitsData/CO2-1/FWTau_CO2-1.fits")

    # Parameters for centering the data.

    centering_params = [0.0, 0.0, 1]

    # Center the data.

    data = uv.center(data, centering_params)

    logger.message("Data: {}".format(data.real.shape))

    # Average the visibilities radially.

    data_1d = uv.average(data, gridsize=20, radial=True, log=True, \
            logmin=data.uvdist[numpy.nonzero(data.uvdist)].min()*0.95, \
            logmax=data.uvdist.max()*1.05, mode="spectralline")

    ################################################################################
    #
    # Fit the model to the data.
    #
    ################################################################################

    # Set up the inputs for the MCMC function.

    logger.message("Initialize MCMC...")
    x = data.u
    y = data.v
    lam = c / data.freq / 1.0e-4
    z = numpy.concatenate((data.real, data.imag))
    zerr = 1./numpy.sqrt(numpy.abs(numpy.concatenate((data.weights, data.weights))))

    # Set up the emcee run.

    ndim, nwalkers = 12, 200

    # [log10(M_disk), R_in, R_disk, h0, gamma, inclination, position_angle]

    if args.resume:
        pos = numpy.load(save_dir+"flared_pos.npy")
        chain = numpy.load(save_dir+"flared_chain.npy")
        state = None
        nsteps = chain[0,:,0].size

        if args.resetprob:
            prob = None
        else:
            prob = numpy.load(save_dir+"flared_prob.npy")
    else:
        pos = []
        for i in range(nwalkers): # have to make this more general
            r_disk = numpy.random.uniform(1.,2.5,1)[0]

            # params
            '''
            star mass    (log10(Msun))
            disk mass    (log10(Msun))
            disk rad    (AU)
            temp @ 1 au (K)
            Power Law
            Micro Turbulense
            Systemic Velocity (km/s)
            inclination
            Position Angle
            Offset in arcsec
            offset in arcsec
            distance (Parsec)
            '''

            pos.append([numpy.random.uniform(-2.,1.,1)[0], \
                    numpy.random.uniform(-6., -2., 1)[0], \
                    r_disk, \
                    numpy.random.uniform(2.,3.,1)[0], \
                    numpy.random.uniform(0.0,1.0,1)[0], \
                    numpy.random.uniform(-2.,0.,1)[0], \
                    numpy.random.uniform(-5.,5.,1)[0], \
                    numpy.random.uniform(0, 180., 1)[0], \
                    numpy.random.uniform(0, 360., 1)[0], \
                    numpy.random.uniform(-0.1,0.1,1)[0], \
                    numpy.random.uniform(-0.1,0.1,1)[0], \
                    numpy.random.uniform(220.,240.,1)[0]])
        prob = None
        chain = numpy.empty((nwalkers, 0, ndim))
        state = None
        nsteps = 0
    # Set up the MCMC simulation.

    if (args.action == "run") or (args.action == 'runlarge'):
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, \
                args=(x, y, lam, z, zerr,info), pool=pool)

    # Run a few burner steps.

    logger.message("Beginning MCMC...")
    time1 = time.time()
    stepsize = 20
    while nsteps < 10000:
        # progress 
        logger.message("Step: {}".format(nsteps))
        timedif = time.time()-time1
        if nsteps == 0:
            with open(work_dir + ".flared_model_"+source_name+".progress",'w') as f:
                f.write("Source: {}, step: {}, time: {}\n".format(source_name,nsteps,timedif))
        else:
            with open(work_dir + ".flared_model_"+source_name+".progress",'a') as f:
                f.write("Source: {}, step: {}, time: {}\n".format(source_name,nsteps,timedif))
        if (args.action == "run") or (args.action == "runlarge"):
            pos, prob, state = sampler.run_mcmc(pos, stepsize, lnprob0=prob,rstate0=state)

            chain = numpy.concatenate((chain, sampler.chain), axis=1)

            for i in range(ndim):
                fig, ax = plt.subplots(nrows=1, ncols=1)

                for j in range(nwalkers):
                    ax.plot(chain[j,:,i])

                plt.savefig(save_dir+"flared_test_{0:d}.pdf".format(i))

                plt.close(fig)

            numpy.save(save_dir+"flared_pos", pos)
            numpy.save(save_dir+"flared_prob", prob)
            numpy.save(save_dir+"flared_chain", chain)

            nsteps += stepsize

            sampler.reset()

        # Get the best fit parameters and uncertainties.

        samples = chain[:,-5:,:].reshape((-1, ndim))

        params = numpy.median(samples, axis=0)
        sigma = samples.std(axis=0)

        # Fix the position angle.

        params[8] = -params[8]

        # If we are just plotting, adjust parameters here.

        if args.action == "plot":
            pass

        # Write out the results.

        if (args.action == "run") or (args.action=='runlarge'):
            with open(save_dir+"flared_fit.txt", "w") as f:
                f.write("Best fit to "+source_name+"\n\n")
                f.write("log10(Mstar) = {0:f} +/- {1:f}\n".format(params[0], sigma[0]))
                f.write("log10(Mdisk) = {0:f} +/- {1:f}\n".format(params[1], sigma[1]))
                f.write("Rdisk = {0:f} +/- {1:f}\n".format(params[2], sigma[2]))
                f.write("T0 = {0:f} +/- {1:f}\n".format(params[3], sigma[3]))
                f.write("q = {0:f} +/- {1:f}\n".format(params[4], sigma[4]))
                f.write("a_turb = {0:f} +/- {1:f}\n\n".format(params[5], sigma[5]))
                f.write("v_sys = {0:f} +/- {1:f}\n\n".format(params[6], sigma[6]))
                f.write("i = {0:f} +/- {1:f}\n".format(params[7], sigma[7]))
                f.write("pa = {0:f} +/- {1:f}\n\n".format(params[8], sigma[8]))
                f.write("x0 = {0:f} +/- {1:f}\n\n".format(params[9], sigma[9]))
                f.write("y0 = {0:f} +/- {1:f}\n\n".format(params[10], sigma[10]))
                f.write("dpc = {0:f} +/- {1:f}\n\n".format(params[11], sigma[11]))

            os.system("cat "+save_dir+"flared_fit.txt")

            # Plot histograms of the resulting parameters.

            xlabels = ["$log_{10}(M_*)$","$log_{10}(M_{disk})$","$R_{disk}$", \
                    "$T_0$","$q$","$a_{turb}$","$v_{sys}$","$i$","p.a.","$x_0$", \
                    "$y_0$","dpc"]

            fig = corner.corner(samples, labels=xlabels, truths=params)

            plt.savefig(save_dir+"flared_fit.pdf")

        ############################################################################
        #
        # Plot the results.
        #
        ############################################################################

        # Plot the best fit model over the data.

        fig, ax = plt.subplots(nrows=5, ncols=12, sharex=True, sharey=True)

        # Create a high resolution model for avereaging.

        m = model(1, 1, c/data.freq/1.0e-4, params,info, output="data")

        m1d = uv.average(m.visibilities["C17O"], gridsize=10000, binsize=5000, \
                radial=True, mode="spectralline")

        # Calculate the velocity for each image.

        nu0 = float(rest_freq)
        nu = data.freq

        try:
            v = c * (nu0 - nu) / nu0
        except TypeError:
            v = float(c) * (float(nu0) - nu) / nu
        # Plot the image.

        """
        vmin = numpy.nanmin(m.images["CO2-1"].image)
        vmax = numpy.nanmax(m.images["CO2-1"].image)
        """

        for i in range(5):
            for j in range(10):
                ind = i*10 + j

                # Plot the visibilities.
                #logger.message("i: {}, j:{}, ind: {}".format(i,j,ind))

                ax[i,j].errorbar(data_1d.uvdist/1000, data_1d.amp[:,ind], \
                        yerr=numpy.sqrt(1./data_1d.weights[:,ind]),\
                        fmt="bo", markersize=5, markeredgecolor="b")

                # Plot the best fit model
                ax[i,j].plot(m1d.uvdist/1000, m1d.amp[:,ind], "g-")

                """
                xmin, xmax = 512-96, 512+96
                ymin, ymax = 512-96, 512+96

                ax[i,j].imshow(m.images["CO2-1"].image[ymin:ymax,xmin:xmax,ind,0],\
                        origin="lower", interpolation="nearest", vmin=vmin, \
                        vmax=vmax)
                """
                txt = ax[i,j].annotate(r"$v=%4.1f$ km s$^{-1}$" % (v[ind]/1e5), \
                        xy=(0.1,0.8), xycoords='axes fraction')

        # Adjust the plot and save it.

        ax[0,0].axis([10,data_1d.uvdist.max()/1000*1.1,0,data_1d.amp.max()*1.1])
        ax[0,0].set_xscale('log')

        # Adjust the plot and save it.

        fig.set_size_inches((15,6.4))
        fig.subplots_adjust(left=0.06, right=0.98, top=0.98, bottom=0.07, \
                wspace=0.0,hspace=0.0)

        # Adjust the figure and save.

        fig.savefig(save_dir+"flared_model.pdf")

        plt.clf()

        if args.action == "plot":
            nsteps = numpy.inf

    # Now we can close the pool.

    if args.action == "run" or args.action == "runlarge":
        pool.close()
