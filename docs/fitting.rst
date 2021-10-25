===========================================
Fitting data with radiative transfer models
===========================================

This section is for setting up your data and running a fit of a disk radiative transfer model to it.

Preparing your data to be in the correct format
"""""""""""""""""""""""""""""""""""""""""""""""

If you have CASA6 installed within your Python installation (currently requires Python 3.6), you can skip this step and move on to the next. Otherwise, if you do not have CASA6 installed in this distribution you can put your data into an HDF5 format. Here’s how:

1. Within CASA, use the exportuvfits to split every spectral window that you care about into a separate UV FITS file. Each MS file should go into a separate .vis file:
   ::

       filenameA.vis/  
       |---- filenameA.1.uv.fits  
       |---- filenameA.2.uv.fits  
       .  
       .  
       .  
       \---- filenameA.N.uv.fits  
       filenameB.vis/  
       |---- filenameB.1.uv.fits  
       |---- filenameB.2.uv.fits  
       .  
       .  
       .  
       \---- filenameB.N.uv.fits

   I’ll typically organize this by array configuration and band, so it may look like this:
   ::

       source_Band3_track1.vis  
       source_Band3_track2.vis  
       source_Band6_track1.vis  

2. Use the below code to turn the .vis files into HDF5 files. 
   ::

       import pdspy.interferometry as uv  
       import glob  

       files = glob.glob("*Band3*.vis")  

       data = []  
       for file in files:  
           data.append(uv.freqcorrect(uv.readvis(file)))  

       vis = uv.concatenate(data)  

       vis.write("Source_Band3.hdf5")

   It’ll grab all of the \*.vis files that match the wildcard at the beginning, so you can adjust that to decide which sets of files get grabbed. So in the above example you could run it once with \*Band3\*.vis to merge the Band 3 data into one file, and then \*Band5\*.vis to merge the Band 6 data into a single dataset.

Setting up a configuration file
"""""""""""""""""""""""""""""""

You can find a basic configuration file in the pdspy bin directory (`config_template.py <https://github.com/psheehan/pdspy/blob/master/bin/config_template.py>`_) as an example, and I think it should be close to what you’ll want for your application. The visibilities dictionary requests a bunch of information about the visibility data. An example of what that should look like is:

    ::

        import numpy

        ################################################################################
        #
        # Set up the list of datasets that will be fit.
        #
        ################################################################################

        # Define the necessary info for the visiblity datasets.

        visibilities = {
                "file":["path/to/file1"],
                "pixelsize":[0.1],
                "freq":["230GHz"],
                "lam":["1300"],
                "npix":[256],
                "weight":[10.],
                "x0":[0.],
                "y0":[0.],
                # Info for the image.
                "image_file":["path/to/image_file1"],
                "image_pixelsize":[0.05],
                "image_npix":[1024],
                # Info for the plots.
                "nrows":[5],
                "ncols":[5],
                "ind0":[1],
                "ticks":[numpy.array([-250,-200,-100,0,100,200,250])],
                "image_ticks":[numpy.array([-0.75,-0.5,0,0.5,0.75])],
                "fmt":['4.1f'],
                }

        # Something similar for images.

        images = {
                "file":["path/to/file1"],
                "npix":[128],
                "pixelsize":[0.1],
                "lam":["0.8"],
                "bmaj":[1.0],
                "bmin":[1.0],
                "bpa":[0.0],
                "ticks":[numpy.array([-6,-3,0,3,6])],
                "plot_mode":["linear"],
                }

        # Something similar for spectra.

        spectra = {
                "file":["path/to/file1"],
                "bin?":[False],
                "nbins":[25],
                "weight":[1.],
                }

        ################################################################################
        #
        # Set up a number of configuration parameters.
        #
        ################################################################################

        # emcee parameters.

        nwalkers = 6            # The number of walkers to use.
        steps_per_iter = 5      # The number of steps to do at one time.
        max_nsteps = 10         # The maximum total number of steps to take.
        nplot = 5               # The number of previous steps to plot.

        # dynesty parameters.

        nlive_init = 250        # The number of live points to use for Dynesty.
        nlive_batch = 250       # Number of live points per batch for dynamic nested 
                                # sampling
        maxbatch = 0            # Maximum number of batches to use.
        dlogz = 0.05            # Stopping threshold for nested sampling.
        walks = 25              # Number of random walk steps to use to generate a 
                                # sample

        ################################################################################
        #
        # Set up the list of parameters and default fixed values.
        #
        ################################################################################

        parameters = {
                # Stellar parameters.
                "logM_star":{"fixed":True, "value":0.0, "limits":[-1.,1.]},
                "T_star":{"fixed":True, "value":4000., "limits":[500.,10000.]},
                "logL_star":{"fixed":False, "value":0.0, "limits":[-1.,2.]},
                # Disk parameters.
                "disk_type":{"fixed":True, "value":"truncated", "limits":[0.,0.]},
                "logM_disk":{"fixed":False, "value":-4., "limits":[-10.,-2.5]},
                "logR_in":{"fixed":False, "value":-1., "limits":[-1.,4.]},
                "logR_disk":{"fixed":False, "value":2., "limits":[0.,4.]},
                "h_0":{"fixed":False, "value":0.1, "limits":[0.01,0.5]},
                "gamma":{"fixed":False, "value":1.0, "limits":[-0.5,2.0]},
                "beta":{"fixed":False, "value":1.0, "limits":[0.5,1.5]},
                # Disk temperature parameters (for flared_model_*)
                "logT0":{"fixed":True, "value":2.5, "limits":[1.,3.]},
                "q":{"fixed":True, "value":0.25, "limits":[0.,1.]},
                "loga_turb":{"fixed":True, "value":-1.0, "limits":[-1.5,1.]},
                # Envelope parameters.
                "envelope_type":{"fixed":True, "value":"ulrich", "limits":[0.,0.]},
                "logM_env":{"fixed":False, "value":-3., "limits":[-10., -2.]},
                "logR_in_env":{"fixed":True, "value":"logR_in", "limits":[-1., 4.]},
                "logR_env":{"fixed":False, "value":3., "limits": [2.,5.]},
                "logR_c":{"fixed":True, "value":"logR_disk", "limits":[-1.,4.]},
                "f_cav":{"fixed":False, "value":0.5, "limits":[0.,1.]},
                "ksi":{"fixed":False, "value":1.0, "limits":[0.5,1.5]},
                # Envelope temperature parameters (for flared_model_*)
                "logT0_env":{"fixed":True, "value":2.5, "limits":[1.,3.5]},
                "q_env":{"fixed":True, "value":0.25, "limits":[0.,1.5]},
                "loga_turb_env":{"fixed":True, "value":-1.0, "limits":[-1.5,1.]},
                # Dust parameters.
                "dust_file":{"fixed":True, "value":"pollack_new.hdf5", "limits":[0.,0.]},
                "loga_min":{"fixed":True, "value":-1.3, "limits":[0.,5.]},
                "loga_max":{"fixed":False, "value":0., "limits":[0.,5.]},
                "p":{"fixed":False, "value":3.5, "limits":[2.5,4.5]},
                "envelope_dust":{"fixed":True, "value":"pollack_new.hdf5", "limits":[0.,0.]},
                # Gas parameters.
                "gas_file1":{"fixed":True, "value":"co.dat", "limits":[0.,0.]},
                "logabundance1":{"fixed":True, "value":-4., "limits":[-6.,-2.]},
                # Viewing parameters.
                "i":{"fixed":False, "value":45., "limits":[0.,180.]},
                "pa":{"fixed":False, "value":0., "limits":[0.,360.]},
                "x0":{"fixed":True, "value":0., "limits":[-0.1,0.1]},
                "y0":{"fixed":True, "value":0., "limits":[-0.1,0.1]},
                "dpc":{"fixed":True, "value":140., "prior":"box", "sigma":0., "limits":[1.,1e6]},
                "v_sys":{"fixed":True, "value":5., "limits":[0.,10.]},
                }

        ################################################################################
        #
        # Set up the priors.
        #
        ################################################################################

        priors = {
                "parallax":{"value":140., "sigma":0.},
                "Mstar":{"value":"chabrier"},
                }

The things in particular you’ll want to update are:

**file:** Either the MS file for your dataset, or the HDF5 visibility files the were created above. Can list as many as you’d like, I just put in 2 as an example. (All of the entries in the visibilities dictionary should be lists with the same number of elements).

**freq/lam:** The frequency/wavelength of the observations. Freq should be a string, lam a number.

**x0/y0:** If the data is far off-center, these are initial corrections to approximately center the data. Positive x0 means east (i.e. to the left in a CASA image) and positive y0 is north (i.e. up in a CASA image).

**image_file:** every HDF5 file should have a corresponding FITS image to show the best fit model over. All of the other image_* parameters correspond to values from the image: pixelsize, npix

Then at the bottom the **parameters** dictionary gives you a giant list of parameters that can be turned on or off. When a parameter has fixed:True, then it is fixed at a value of value. If fixed:False, then it’s a free parameter constrained by limits. For a full list of parameters, see `here <https://github.com/psheehan/pdspy/blob/master/pdspy/modeling/base_parameters.py>`_

The **flux_unc\*** parameters at the bottom add a flux uncertainty to the observations, with sigma:0.1 = 10% uncertainty (but that can be changed), and a Gaussian prior. You can add as many of these as you have visibility files, so you can tune the flux uncertainty separately for each dataset.

Running a model
"""""""""""""""

Make sure /path/to/pdspy/bin is in your PATH so that you can see the disk_model.py function. There are currently two well tested tools to run models:

+ **disk_model_emcee3.py or disk_model_nested.py**: Used to fit ALMA continuum visibilities and broadband spectral energy distributions (SEDs) with full radiative transfer models.

+ **flared_model_emcee3.py or flared_model_nested.py**: Used to fit ALMA spectral line visibilities with protoplanetary disk models that include a vertically isothermal, power law temperature distribution. No radiative equilibrium calculation is done.

From there the most basic way to run any one of these models is in the directory with config.py and entering:
::

    disk_model_emcee3.py --object <Object Name>

If you want to run with parallel RADMC-3D, to speed up the code, you can run:
::

    disk_model_emcee3.py --object <Object Name> --ncpus N

Progress is saved, so if you want to resume a fit that stopped for some reason, you can add:
::

    disk_model_emcee3.py --object <Object Name> --ncpus N --resume

You can also use MPI to run multiple single core models at once:
::

    mpirun -np N disk_model_emcee3.py --object <Object Name> --ncpus 1

Or some combination of simultaneous models and parallel RADMC-3D:
::

    mpirun -np N disk_model_emcee3.py --object <Object Name> --ncpus M

(where NxM should be <= the number of cores on your computer). The last two commands for running the code (using MPI) make it adaptable so that it can be run on supercomputers as well, for an even bigger boost. If you want to do this, let me know and I can provide some more details of how to efficiently run over multiple supercomputer nodes.
