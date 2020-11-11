===============================================
Generating radiative transfer models with pdspy
===============================================

Now that you have pdspy installed, how do you actually use it to generate radiative transfer models? Read on to find out!

Setting up a model
""""""""""""""""""

To set up a model, first we need to import all of the relevant pdspy packages:
::

      import pdspy.modeling as modeling
      import pdspy.interferometry as uv
      import pdspy.dust as dust
      import pdspy.gas as gas

Now, we create a YSOModel object,
::

      m = modeling.YSOModel()

To run a model, we need to set up a model grid:
::

      nr, ntheta, nphi = 100, 100, 2
      rmin, rmax = 0.1, 300

      m.set_spherical_grid(rmin, rmax, nr, ntheta, nphi, code="radmc3d")

We also need to set up the dust properties. Fortunately, pdspy comes with a number of built in dust models. We'll use the diana_wice.hdf5 file, which includes dust properties based off of the DIANA project, but with water ice added:
::

      dust_gen = dust.DustGenerator(dust.__path__[0]+"/data/diana_wice.hdf5")

      a_max = 100 # microns
      p = 3.5

      d = dust_gen(a_max / 1e4, p) # dust_gen wants units of cm

Optionally, you can also include gas in your model if you want to make spectral line channel maps. pdspy also comes with data files from the LAMDA database for a number of common molecules already built in:
::

      abundance = []
      gases = []

      co = gas.Gas()
      co.set_properties_from_lambda(os.environ["HOME"]+\
              "/Documents/Projects/DiskMasses/Modeling/Gas/co.dat")
      gases.append(co)
      abundance.append(1.0e-4)

Finally, we need to set up all of the components of the model: star, disk, and envelope (if desired):
::

      m.add_star(mass=0.5, luminosity=1., temperature=4000.)
      m.add_disk(mass=0.01, rmin=0.1, rmax=50., plrho=1., h0=0.1, plh=1., \
              dust=d, gas=gases, abundance=abundance)
      m.add_ulrich_envelope(mass=0.01, rmin=0.1, rmax=3000., dust=d, \
              gas=gases, abundance=abundance)

      # The below sets up the wavelength grid for RADMC3D.
      m.grid.set_wavelength_grid(0.1, 1.0e5, 500, log=True)

Running radiative transfer models
"""""""""""""""""""""""""""""""""

Now that we have our model set up, we need to run the thermal simulation to calculate the temperature everywhere in the model grid:
::

      m.run_thermal(nphot=1e6, modified_random_walk=True, verbose=True, \
              setthreads=1, code="radmc3d")

If you want to visualize the model, the relevant information is stored within a Grid class that is part of the YSOModel class:
::

      m.grid.r # Units of au
      m.grid.theta
      m.grid.phi
      
      m.grid.density[0] # Disk density. Should be (r, theta, phi)
      m.grid.temperature[0] # Disk temperature
      m.grid.density[1] # Envelope density (if included)
      m.grid.temperature[1] # Envelope temperature (if included)

Generating synthetic observations
"""""""""""""""""""""""""""""""""

To run images or visibilities, then, we can simply run:
::

      # pixelsize is in units of arcseconds.
      m.run_image(name="870um", nphot=1e5, npix=256, pixelsize=0.01, \
              lam="870", incl=45, pa=30, dpc=140, code="radmc3d", \
              verbose=True, setthreads=2)

      m.run_visibilities(name="870um", nphot=1e5, npix=256, pixelsize=0.01, \
              lam="870", incl=45, pa=30, dpc=140, code="radmc3d", \
              verbose=True, setthreads=2)

To generate synthetic channel maps, you can either use the built in options to specify velocities:
::

      m.run_image(name="CO2-1", nphot=0, npix=256, pixelsize=0.01, \
              lam=None, imolspec=1, iline=2, widthkms=10., linenlam=400, \
              tgas_eq_tdust=True, scattering_mode_max=0, incl_dust=False, \
              incl=45, pa=30, dpc=140, code="radmc3d", verbose=True, \
              setthreads=2, writeimage_unformatted=True)

Or, you can manually specify the wavelengths that you want to use, and let RADMC3D automatically detect which spectral lines are relevant.
::

      velocities = numpy.linspace(-10., 10., 0.25) * 1.0e5 # in cm/s
      nu = 230.538 * (1. - velocities / 2.99e10)
      wave = 2.99e10 / nu

      m.set_camera_wavelength(wave)

      m.run_image(name="CO2-1", nphot=0, npix=256, pixelsize=0.01, \
              loadlambda=True, tgas_eq_tdust=True, scattering_mode_max=0, \
              incl_dust=False, incl=45, pa=30, dpc=140, code="radmc3d", \
              verbose=True, setthreads=2, writeimage_unformatted=True)

We can also generate broadband SEDs:
::

      m.set_camera_wavelength(numpy.logspace(-1, 4, 50))

      m.run_sed(name="SED", nphot=1e4, loadlambda=True, incl=45, pa=30, \
              dpc=140, code="radmc3d", verbose=True, setthreads=2)

Accessing and plotting synthetic observations
"""""""""""""""""""""""""""""""""""""""""""""

Synthetic observations generated with pdspy are stored in dictionaries within the :code:`YSOModel` class:
::

      m.images
      m.visibilities
      m.spectra

and can be accessed using the name that they were given when they were generated. For example, to plot an SED, you could run
::

      import matplotlib.pyplot as plt

      plt.loglog(m.spectra["SED"].wave, m.spectra["SED"].flux, "b-")
      plt.show()

Images are actually 4D structures, with the last two dimensions for frequency (in case of image cubes) and polarization (not yet implemented in pdspy). You can plot an image using:
::

      plt.imshow(m.images["870um"].image[:,:,0,0], origin="lower", \
              interpolation="nearest")
      plt.show()

Images also have a few additional pieces of data that may be of use:
::

      m.images["870um"].x
      m.images["870um"].y
      m.images["870um"].freq

Finally, to show the visibility data, lets azimuthally average it for ease of viewing.
::

      import pdspy.interferometry as uv

      m1d = uv.average(m.visibilities["870um"], gridsize=10000, binsize=3500, \
              radial=True)

      plt.semilogx(m1d.uvdist, m1d.amp, "-")
      
      plt.show()

A few other components of the Visibility class that you might find useful:
::

      m.visibilities["870um"].u
      m.visibilities["870um"].v
      m.visibilities["870um"].uvdist
      m.visibilities["870um"].real
      m.visibilities["870um"].imag
      m.visibilities["870um"].amp
      m.visibilities["870um"].weights
      m.visibilities["870um"].freq
