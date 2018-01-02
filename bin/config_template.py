import numpy

################################################################################
#
# Set up the list of datasets that will be fit.
#
################################################################################

# Define the necessary info for the visiblity datasets.

visibilities = {
        "file":["path/to/file1"],
        "binsize":[8057.218995847603],
        "pixelsize":[0.1],
        "freq":["230GHz"],
        "lam":["1300"],
        "npix":[256],
        "gridsize":[256],
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
        }

################################################################################
#
# Set up a number of configuration parameters.
#
################################################################################

# The number of walkers to use.

nwalkers = 6

# The number of steps to do at one time.

steps_per_iter = 5

# The maximum total number of steps to take.

max_nsteps = 10

# The number of previous steps to plot.

nplot = 5

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
        "logM_disk":{"fixed":True, "value":-4., "limits":[-10.,-2.5]},
        "logR_in":{"fixed":True, "value":-1., "limits":[-1.,4.]},
        "logR_disk":{"fixed":True, "value":2., "limits":[0.,4.]},
        "h_0":{"fixed":True, "value":0.1, "limits":[0.01,0.5]},
        "gamma":{"fixed":True, "value":1.0, "limits":[-0.5,2.0]},
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
        "f_h_large":{"fixed":True, "value":0.5, "limits":[0.1, 1.]},
        "beta_large":{"fixed":True, "value":1., "limits":[0.5, 1.5]},
        # Disk temperature parameters.
        "logT0":{"fixed":True, "value":2.5, "limits":[1.,3.5]},
        "q":{"fixed":True, "value":0.25, "limits":[0.,1.5]},
        "loga_turb":{"fixed":True, "value":-1.0, "limits":[-3.,1.]},
        # Envelope parameters.
        "logM_env":{"fixed":True, "value":-3., "limits":[-10., -2.]},
        "logR_in_env":{"fixed":True, "value":"logR_in", "limits":[-1., 4.]},
        "logR_env":{"fixed":True, "value":3., "limits": [2.,5.]},
        "logR_c":{"fixed":True, "value":"logR_disk", "limits":[-1.,4.]},
        "f_cav":{"fixed":True, "value":0.5, "limits":[0.,1.]},
        "ksi":{"fixed":True, "value":1.0, "limits":[0.5,1.5]},
        # Dust parameters.
        "loga_max":{"fixed":True, "value":0., "limits":[0.,5.]},
        "p":{"fixed":True, "value":3.5, "limits":[2.5,4.5]},
        # Gas parameters.
        "gas_file":{"fixed":True, "value":"co.dat", "limits":[0.,0.]},
        "logabundance":{"fixed":True, "value":-4., "limits":[-6.,-2.]},
        # Viewing parameters.
        "i":{"fixed":True, "value":45., "limits":[0.,180.]},
        "pa":{"fixed":True, "value":0., "limits":[0.,360.]},
        "x0":{"fixed":True, "value":0., "limits":[-0.1,0.1]},
        "y0":{"fixed":True, "value":0., "limits":[-0.1,0.1]},
        "dpc":{"fixed":True, "value":140., "limits":[1.,1e6]},
        "Ak":{"fixed":True, "value":0., "limits":[0.,1.]},
        "v_sys":{"fixed":True, "value":5., "limits":[0.,10.]},
        }
