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
        "logL_star":{"fixed":True, "value":0.0, "limits":[-1.,2.]},
        # Disk parameters.
        "disk_type":{"fixed":True, "value":"truncated", "limits":[0.,0.]},
        "logM_disk":{"fixed":True, "value":-4., "limits":[-10.,-2.5]},
        "logR_in":{"fixed":True, "value":-1., "limits":[-1.,4.]},
        "logR_disk":{"fixed":True, "value":2., "limits":[0.,4.]},
        "h_0":{"fixed":True, "value":0.1, "limits":[0.01,0.5]},
        "gamma":{"fixed":True, "value":1.0, "limits":[-0.5,2.0]},
        "gamma_taper":{"fixed":True, "value":"gamma", "limits":[-0.5,2.0]},
        "beta":{"fixed":True, "value":1.0, "limits":[0.5,1.5]},
        "logR_cav":{"fixed":True, "value":1.0, "limits":[-1.,3.]},
        "logdelta_cav":{"fixed":True, "value":0.0, "limits":[-4.,0.]},
        "logR_gap1":{"fixed":True, "value":1.0, "limits":[-1.,3.]},
        "w_gap1":{"fixed":True, "value":10., "limits":[1.,100.]},
        "logdelta_gap1":{"fixed":True, "value":0.0, "limits":[-4.,0.]},
        "logR_gap2":{"fixed":True, "value":1.4, "limits":[-1.,3.]},
        "w_gap2":{"fixed":True, "value":10., "limits":[1.,100.]},
        "logdelta_gap2":{"fixed":True, "value":0.0, "limits":[-4.,0.]},
        "logR_gap3":{"fixed":True, "value":1.8, "limits":[-1.,3.]},
        "w_gap3":{"fixed":True, "value":10., "limits":[1.,100.]},
        "logdelta_gap3":{"fixed":True, "value":0.0, "limits":[-4.,0.]},
        "f_M_large":{"fixed":True, "value":0.8, "limits":[0.05, 1.]},
        "logalpha_settle":{"fixed":True, "value":-2., "limits":[-5., 0.]},
        # Disk temperature parameters.
        "logT0":{"fixed":True, "value":2.5, "limits":[1.,3.]},
        "q":{"fixed":True, "value":0.25, "limits":[0.,1.]},
        "loga_turb":{"fixed":True, "value":-1.0, "limits":[-1.5,1.]},
        # Dartois temperature properties.
        "logTmid0":{"fixed":True, "value":2.0, "limits":[1.,3.]}, 
        "logTatm0":{"fixed":True, "value":2.5, "limits":[1.,3.]}, 
        "zq0":{"fixed":True, "value":0.1, "limits":[0.01,0.5]},
        "pltgas":{"fixed":True, "value":0.5, "limits":[0.,1.]},
        "delta":{"fixed":True, "value":1.0, "limits":[0.5,1.5]},
        # Envelope parameters.
        "envelope_type":{"fixed":True, "value":"ulrich", "limits":[0.,0.]},
        "logM_env":{"fixed":True, "value":-3., "limits":[-10., -2.]},
        "logR_in_env":{"fixed":True, "value":"logR_in", "limits":[-1., 4.]},
        "logR_env":{"fixed":True, "value":3., "limits": [2.,5.]},
        "logR_c":{"fixed":True, "value":"logR_disk", "limits":[-1.,4.]},
        "f_cav":{"fixed":True, "value":0.5, "limits":[0.,1.]},
        "ksi":{"fixed":True, "value":1.0, "limits":[0.5,1.5]},
        "theta_open":{"fixed":True, "value":"45", "limits":[0.,90.]},
        "zoffset":{"fixed":True, "value":1, "limits":[0.,5.]},
        "gamma_env":{"fixed":True, "value":0., "limits":[-0.5,2.0]},
        # Envelope temperature parameters.
        "logT0_env":{"fixed":True, "value":2.5, "limits":[1.,3.5]},
        "q_env":{"fixed":True, "value":0.25, "limits":[0.,1.5]},
        "loga_turb_env":{"fixed":True, "value":-1.0, "limits":[-1.5,1.]},
        # Dust parameters.
        "dust_file":{"fixed":True, "value":"pollack_new.hdf5", "limits":[0.,0.]},
        "loga_min":{"fixed":True, "value":-1.3, "limits":[0.,5.]},
        "loga_max":{"fixed":True, "value":0., "limits":[0.,5.]},
        "p":{"fixed":True, "value":3.5, "limits":[2.5,4.5]},
        "na":{"fixed":True, "value":100, "limits":[0,1000]},
        "envelope_dust":{"fixed":True, "value":"pollack_new.hdf5", "limits":[0.,0.]},
        # Gas parameters.
        "gas_file1":{"fixed":True, "value":"co.dat", "limits":[0.,0.]},
        "logabundance1":{"fixed":True, "value":-4., "limits":[-6.,-2.]},
        # Viewing parameters.
        "i":{"fixed":True, "value":45., "limits":[0.,180.]},
        "pa":{"fixed":True, "value":0., "limits":[0.,360.]},
        "x0":{"fixed":True, "value":0., "limits":[-0.1,0.1]},
        "y0":{"fixed":True, "value":0., "limits":[-0.1,0.1]},
        "dpc":{"fixed":True, "value":140., "prior":"box", "sigma":0., "limits":[1.,1e6]},
        "Ak":{"fixed":True, "value":0., "limits":[0.,1.]},
        "v_sys":{"fixed":True, "value":5., "limits":[0.,10.]},
        "docontsub":{"fixed":True, "value":False, "limits":[0.,0.]},
        # Gas extinction parameters.
        "tau0":{"fixed":True, "value":0., "limits":[0.,10.]},
        "v_ext":{"fixed":True, "value":4., "limits":[2.,6.]},
        "sigma_vext":{"fixed":True, "value":1.0, "limits":[0.01,5.]},
        # Nuisance parameters.
        "flux_unc1":{"fixed":True, "value":1., "prior":"box", "sigma":0., "limits":[0.5,1.5]},
        "flux_unc2":{"fixed":True, "value":1., "prior":"box", "sigma":0., "limits":[0.5,1.5]},
        "flux_unc3":{"fixed":True, "value":1., "prior":"box", "sigma":0., "limits":[0.5,1.5]},
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
