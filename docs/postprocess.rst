================================================
Post-processing radiative transfer modeling fits
================================================

Congratulations! You have a model that fits your data. Now, how do you make nice plots of it? Fortunately, pdspy has your back.

Loading in the results
""""""""""""""""""""""

To start, we need to load in the relevant files:
::

       import pdspy.utils as utils

       model_path = "path/to/where/you/ran/the/model/"

       config = utils.load_config(path=model_path)
       visibilities, images, spectra = utils.load_data(config, model="disk")
       keys, params, samples = utils.load_results(path=model_path)

This will load in all of the necessary configuration information, the data, and the results of the fit, namely the best-fit parameters. Next, you need to run the best-fit model to have model data to plot:
::

       import pdspy.modeling as modeling

       m = modeling.run_disk_model(config.visibilities, params, \
               config.parameters, plot=True, ncpus=ncpus, source=source, \
               plot_vis=False)

Great! You should now have all of the information that you need in order to make nice plots of your best-fit model!

Plotting continuum results
""""""""""""""""""""""""""

pdspy has a number of plotting utilities built right into it so that you can make nice, publication-quality plots without needing to get under the hood. All of the tools are built into the plotting package within pdspy:
::

       import pdspy.plotting as plotting

Plotting continuum images:
::

       fig, ax = plotting.plot_continuum_image(visibilities, m, \
               config.parameters, params, index=0, cmap="Blues", fontsize=15, \
               image="data", contours="model", cmap_contours="Blues")

In all of the plotting routines, you have the option to have pdspy create the figure and axes, as was done above, or you can specify figure and axes instances that you have already created:
::

       fig, ax = plt.subplots(nrows=1, ncols=3)

       plotting.plot_continuum_image(visibilities, m, config.parameters, \
               params, index=0, fig=(fig, ax[0]), cmap="Blues", \
               fontsize=15, image="data", contours="model", \
               cmap_contours="Blues")

Similar routines exist for plotting the 1D, azimuthally averaged visibilities:
::

       plotting.plot_1D_visibilities(visibilities, m, config.parameters, \
               params, index=0, fig=(fig, ax[1]), plot_disk=True, \
               color="black", markersize=8, linewidth=2.5, \
               line_color='#1f77b4', disk_only_color="gray", \
               fontsize=15)

Or for plotting the SED:
::

       plotting.plot_SED(spectra, m, SED=True, fig=(fig, ax[0,2]), \
               model_color='#1f77b4', linewidth=2.5, fontsize=15)

The SED keyword controls whether to plot the y-axis in units of Jy (:code:`SED=False`) or in units of ergs cm :sup:`-2` s :sup:`-1` (:code:`SED=True`).

Plotting spectral line results
""""""""""""""""""""""""""""""

Plotting the results from spectral lines are ever so slightly more involved, as you need to make sure you have the appropriate number of rows and columns shown. For example, to make a plot with 3 rows showing the data, model, and residuals, you could do:
::

      fig, ax = plt.subplots(nrows=3, ncols=8, sharex=True, sharey=True)

      # Make sure that the configuration is properly set up, otherwise it will
      # try to plot too many rows and columns. ind0 controls which channel to
      # start plotting at.
      config.visibilities["nrows"] = [1]
      config.visibilities["ncols"] = [8]
      config.visibilities["ind0"] = [1]

      plotting.plot_channel_maps(config.visibilities, m, \
              config.parameters, params, index=0, fig=(fig, ax[0,:]), \
              image='data', contours='data', image_cmap="BlueToRed", \
              contours_colors="k", fontsize=13, show_velocity=True)

      plotting.plot_channel_maps(config.visibilities, m, \
              config.parameters, params, index=0, fig=(fig, ax[1,:]), \
              image='model', contours='model', model_image="CLEAN", \
              maxiter=100, threshold=0.01, image_cmap="BlueToRed", \
              contours_colors="k", fontsize=13, show_velocity=False)

      plotting.plot_channel_maps(config.visibilities, m, \
              config.parameters, params, index=0, fig=(fig, ax[2,:]), \
              image='residuals', contours='residuals', model_image="CLEAN", \
              maxiter=0, threshold=0.01, image_cmap="BlueToRed", \
              contours_colors="k", fontsize=13, show_velocity=True)

You can also plot channel by channel the one-dimensional, azimuthally averaged visibility profiles:
::

      fig, ax = plt.subplots(nrows=3, ncols=8, sharex=True, sharey=True)

      config.visibilities["nrows"] = [3]
      config.visibilities["ncols"] = [8]
      config.visibilities["ind0"] = [1]

      plotting.plot_channel_maps(config.visibilities, m, \
              config.parameters, params, index=0, plot_vis=True, \
              fig=(fig, ax), image="data", contours="model", \
              fontsize=13, show_velocity=True, vis_color="k", \
              vis_model_color="g", show_xlabel=False)

And of course, keep in mind that you can always make any adjustments to the figure that you would like after the fact, because you have control over the :code:`fig` and :code:`ax` objects that store their properties.
