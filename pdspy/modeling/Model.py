import numpy
import h5py
import os
try:
    from hyperion.dust import IsotropicDust
    from hyperion.model import Model as HypModel
    from hyperion.model import ModelOutput
    from hyperion.model.helpers import find_last_iteration
    from hyperion.util.otf_hdf5 import _on_the_fly_hdf5
    import h5py
except:
    print("WARNING: Hyperion versions <= 0.9.10 will not work with astropy >= 4.0 because they depend on astropy.extern.six. Continuing without Hyperion.")
from .. import radmc3d
from .Grid import Grid
from ..imaging import Image, UnstructuredImage, imtovis
from ..interferometry import Visibilities
from ..spectroscopy import Spectrum
from ..constants.astronomy import AU, M_sun, R_sun, L_sun, Jy, arcsec, pc
from ..constants.physics import c
import time

class Model:
    r"""
    A class that can be used to set up a generic model with arbitrary coordinates and densities.

    Base Attributes:
        :attr:`grid` (Grid):
            The Grid object that contains information about the 3D spatial distribution of the material in the system.
        :attr:`images` (dict):
            A dictionary containing the radiative transfer model images that have been generated for the model.
        :attr:`spectra` (dict):
            A dictionary containing the radiative transfer model spectra that have been generated for the model.
        :attr:`visibilities` (dict):
            A dictionary containing the radiative transfer model visibilities that have been generated for the model.
    """


    def __init__(self):
        self.grid = Grid()
        self.images = {}
        self.spectra = {}
        self.visibilities = {}

    def set_camera_wavelength(self, lam):
        r"""
        Set the wavelengths that the camera should use for generating images, visibilities, and spectra.

        Args:
            :attr:`lam` (numpy.ndarray `n`):
                The array of wavelengths, in microns.
        """
        self.camera_wavelength = lam

    def run_thermal(self, nphot=1e6, code="radmc3d", **keywords):
        r"""
        Run the radiative equilibrium calculation for the model.

        Args:
            :attr:`nphot` (int, optional):
                The number of photons to use for the calculation. Default: `1e6`
            :attr:`code` (str, optional):
                Which underlying radiative transfer modeling code to use for the radiative equilibrium calculation. `radmc3d` or `hyperion`. Default: `radmc3d`
            :attr:`kwargs` (optional):
                Can be used to pass arguments to either 
                :code:`Model.run_thermal_hyperion` or 
                :code:`Model.run_thermal_radmc3d`, depending on the value of 
                :code:`code`.
        """
        if (code == "radmc3d"):
            self.run_thermal_radmc3d(nphot=nphot, **keywords)
        else:
            self.run_thermal_hyperion(nphot=nphot, **keywords)

    def run_thermal_hyperion(self, nphot=1e6, mrw=False, pda=False, \
            niterations=20, percentile=99., absolute=2.0, relative=1.02, \
            max_interactions=1e8, mpi=False, nprocesses=None, \
            sublimation_temperature=None, verbose=True, timeout=3600, \
            increase_photons_until_convergence=False):
        """
        Run the radiative equilibrium calculation using the Hyperion radiative
        transfer code. As a result, the `Model.grid.temperature` list will be
        populated with the temperatures calculated.

        Args:
            :attr:`nphot` (int, optional):
                The number of photons to use for the calculation. Default: `1e6`
            :attr:`mrw` (bool, optional):
                If using `hyperion`, whether or not to use the Modified Random 
                Walk Algorithm. Default: `False`
            :attr:`pda` (bool, optional):
                If using `hyperion`, whether or not to use the Partial Diffusion
                Approximation algorithm. Default: `False`
            :attr:`niterations` (int, optional):
                The maximum number of iterations to perform, if convergence is
                not reached, before giving up. Default: `20`
            :attr:`percentile` (`float`, optional):
                Which percentile of cells to use when calculating the 
                convergence criteria. Default: `99.`
            :attr:`absolute` (`float`, optional):
                Maximum absolute difference of the ratio between cells for 
                convergence to be reached. Default: `2.0`
            :attr:`relative` (`float`, optional):
                Relative difference between cells for convergence to be reached.
                Default: `1.02`
            :attr:`max_interactions` (`int`, optional):
                Maximum number of interactions a photon can have before it is 
                killed. Default: `1e8`
            :attr:`mpi` (bool, optional):
                If using `hyperion`, whether or not to run the model in 
                parallel with MPI. Default: `False`
            :attr:`nprocesses` (bool, optional):
                If `mpi=True`, the number of MPI threads to use. Default: `None`
            :attr:`sublimation_temperature` (`float`, optional):
                If you would like to sublimate dust above a certain temperature,
                set that here. Default: `None`
            :attr:`verbose` (`bool`, False):
                Should output be printed to the screen, or hidden. 
                Default: `False`
        """

        d = []
        for i in range(len(self.grid.dust)):
            d.append(IsotropicDust( \
                    self.grid.dust[i].nu[::-1].astype(numpy.float64), \
                    self.grid.dust[i].albedo[::-1].astype(numpy.float64), \
                    self.grid.dust[i].kext[::-1].astype(numpy.float64)))

            if sublimation_temperature != None:
                d[-1].set_sublimation_temperature('fast', \
                        temperature=sublimation_temperature)

        m = HypModel()
        if (self.grid.coordsystem == "cartesian"):
            m.set_cartesian_grid(self.grid.w1*AU, self.grid.w2*AU, \
                    self.grid.w3*AU)
        elif (self.grid.coordsystem == "cylindrical"):
            m.set_cylindrical_polar_grid(self.grid.w1*AU, self.grid.w3*AU, \
                    self.grid.w2)
        elif (self.grid.coordsystem == "spherical"):
            m.set_spherical_polar_grid(self.grid.w1*AU, self.grid.w2, \
                    self.grid.w3)

        for i in range(len(self.grid.density)):
            if (self.grid.coordsystem == "cartesian"):
                m.add_density_grid(numpy.transpose(self.grid.density[i], \
                        axes=(2,1,0)), d[i])
            if (self.grid.coordsystem == "cylindrical"):
                m.add_density_grid(numpy.transpose(self.grid.density[i], \
                        axes=(1,2,0)), d[i])
            if (self.grid.coordsystem == "spherical"):
                m.add_density_grid(numpy.transpose(self.grid.density[i], \
                        axes=(2,1,0)), d[i])

        sources = []
        for i in range(len(self.grid.stars)):
            sources.append(m.add_point_source())
            sources[i].luminosity = self.grid.stars[i].luminosity * L_sun
            #sources[i].radius = self.grid.stars[i].radius * R_sun
            sources[i].temperature = self.grid.stars[i].temperature

        m.set_mrw(mrw)
        m.set_pda(pda)
        m.set_max_interactions(max_interactions)
        m.set_n_initial_iterations(niterations)
        m.conf.output.output_density = 'last'
        m.set_convergence(True, percentile=percentile, absolute=absolute, \
                relative=relative)

        converged = False
        while not converged:
            m.set_n_photons(initial=nphot, imaging=0)
            m.write("temp.rtin")

            if verbose:
                m.run("temp.rtout", mpi=mpi, n_processes=nprocesses, \
                        overwrite=True, timeout=timeout)
            else:
                m.run("temp.rtout", mpi=mpi, n_processes=nprocesses, \
                        overwrite=True, logfile="temp.log", timeout=timeout)

            f = h5py.File("temp.rtout","r")
            if verbose:
                print("Ran ", find_last_iteration(f), "iterations")
            if increase_photons_until_convergence and \
                    find_last_iteration(f) == niterations:
                if verbose:
                    print("Convergence not reached, increasing nphot by a factor of 10 and trying again.")
                nphot *= 10
                m.use_quantities("temp.rtout", quantities=['specific_energy'])
                os.system("rm temp.rtin temp.rtout temp.log")
            else:
                converged = True
            f.close()

        n = ModelOutput("temp.rtout")

        grid = n.get_quantities()

        self.grid.temperature = []
        temperature = grid.quantities['temperature']
        density = grid.quantities['density']
        for i in range(len(temperature)):
            if (self.grid.coordsystem == "cartesian"):
                self.grid.temperature.append(numpy.transpose(temperature[i], \
                        axes=(2,1,0)))
                self.grid.density[i] = numpy.transpose(density[i], \
                        axes=(2,1,0))
            if (self.grid.coordsystem == "cylindrical"):
                self.grid.temperature.append(numpy.transpose(temperature[i], \
                        axes=(2,0,1)))
                self.grid.density[i] = numpy.transpose(density[i], \
                        axes=(2,0,1))
            if (self.grid.coordsystem == "spherical"):
                self.grid.temperature.append(numpy.transpose(temperature[i], \
                        axes=(2,1,0)))
                self.grid.density[i] = numpy.transpose(density[i], \
                        axes=(2,1,0))

        os.system("rm temp.rtin temp.rtout temp.log")

    def run_thermal_radmc3d(self, nphot=1e6, verbose=True, timelimit=7200, \
            nice=None, **keywords):
        """
        Run the radiative equilibrium calculation using the RADMC-3D radiative
        transfer code. As a result, the `Model.grid.temperature` list will be
        populated with the temperatures calculated.

        Args:
            :attr:`nphot` (int, optional):
                The number of photons to use for the calculation. Default: `1e6`
            :attr:`verbose` (`bool`, False):
                Should output be printed to the screen, or hidden. 
                Default: `False`
            :attr:`**keywords` (optional):
                This can be used to pass any options to RADMC-3D. For a list of
                all possibilities for the thermal code, check the RADMC-3D
                documentation. Two commonly used ones are described below.
            :attr:`modified_random_walk` (bool, optional):
                If using `radmc3d`, whether or not to use the Modified Random 
                Walk Algorithm. Default: `False`
            :attr:`setthreads` (int, optional):
                The number of OpenMP threads to use. Default: `1`
        """

        self.write_radmc3d(nphot_therm=nphot, **keywords)

        radmc3d.run.thermal(verbose=verbose, timelimit=timelimit, nice=nice)

        self.grid.temperature = radmc3d.read.dust_temperature()
        for i in range(len(self.grid.temperature)):
            n1, n2, n3 = self.grid.density[i].shape
            self.grid.temperature[i] = numpy.transpose( \
                    self.grid.temperature[i].reshape((n3,n2,n1)), \
                    axes=(2,1,0))

        os.system("rm *.out *.inp *.dat")

    def run_scattering(self, nphot=1e6, code="radmc3d", **keywords):
        r"""
        Run a scattering phase function calculation for the model.

        Args:
            :attr:`nphot` (int, optional):
                The number of photons to use for the calculation. Default: `1e6`
            :attr:`code` (str, optional):
                Which underlying radiative transfer modeling code to use for 
                the radiative equilibrium calculation. `radmc3d` or `hyperion`. 
                Default: `radmc3d`
            :attr:`kwargs` (optional):
                Can be used to pass arguments to either 
                :code:`Model.run_scattering_hyperion` or 
                :code:`Model.run_scattering_radmc3d`, depending on the value of 
                :code:`code`.
        """
        if (code == "radmc3d"):
            self.run_scattering_radmc3d(nphot=nphot, **keywords)
        else:
            print("Scattering phase function cannot be calculated in Hyperion!")

    def run_scattering_radmc3d(self, nphot=1e6, verbose=True, nice=None, \
            loadlambda=None, **keywords):
        """
        Run a scattering phase function calculation using the RADMC-3D radiative
        transfer code. As a result, the `Model.grid.scattering_phase` list will 
        be populated with the scattering phase functions calculated.

        Args:
            :attr:`nphot` (int, optional):
                The number of photons to use for the calculation. Default: `1e6`
            :attr:`verbose` (`bool`, False):
                Should output be printed to the screen, or hidden. 
                Default: `False`
            :attr:`loadlambda` (`bool`, optional):
                If `True`, use the :code:`Model.camera_wavelength` array as the
                list of wavelengths for the scattering phase function 
                calculation. If `None`, use the :code:`Model.grid.lam array`. 
                Default: `None`
            :attr:`**keywords` (optional):
                This can be used to pass any options to RADMC-3D. For a list of
                all possibilities for the scattering phase function code, check 
                the RADMC-3D documentation. Two commonly used ones are 
                described below.
            :attr:`setthreads` (int, optional):
                The number of OpenMP threads to use. Default: `1`
        """

        self.write_radmc3d(nphot_scat=nphot, **keywords)

        radmc3d.run.scattering(verbose=verbose, nice=nice, \
                loadlambda=loadlambda)

        self.grid.scattering_phase_freq, self.grid.scattering_phase = \
                radmc3d.read.scattering_phase()
        for i in range(len(self.grid.scattering_phase)):
            n1, n2, n3 = self.grid.density[0].shape
            self.grid.scattering_phase[i] = numpy.transpose( \
                    self.grid.scattering_phase[i].reshape((n3,n2,n1)), \
                    axes=(2,1,0))

        self.grid.scattering_phase = numpy.array(self.grid.scattering_phase)

        os.system("rm *.out *.inp *.dat")

    def run_image(self, name=None, nphot=1e6, code="radmc3d", **keywords):
        r"""
        Run an image of the model.

        Args:
            :attr:`name` (`str`, optional):
                The name of the image, to use as a key in the 
                :code:`Model.images` dictionary. Default: `None`
            :attr:`nphot` (int, optional):
                The number of photons to use for the calculation. Default: `1e6`
            :attr:`code` (str, optional):
                Which underlying radiative transfer modeling code to use for 
                the radiative equilibrium calculation. `radmc3d` or `hyperion`. 
                Default: `radmc3d`
            :attr:`kwargs` (optional):
                Can be used to pass arguments to either 
                :code:`Model.run_scattering_hyperion` or 
                :code:`Model.run_scattering_radmc3d`, depending on the value of 
                :code:`code`.
        """

        if (code == "radmc3d"):
            self.run_image_radmc3d(name=name, nphot=nphot, **keywords)
        else:
            self.run_image_hyperion(name=name, nphot=nphot, **keywords)

    def run_image_hyperion(self, nphot=1e6, mrw=False, pda=False, \
            niterations=20, percentile=99., absolute=2.0, relative=1.02, \
            max_interactions=1e8, mpi=False, nprocesses=None, \
            sublimation_temperature=None, verbose=True, incl=45, pa=45, \
            dpc=1, lam=1300., track_origin='basic', nphot_imaging=1e6, \
            name=None, npix=256, pixelsize=1.0):
        """
        Run the radiative equilibrium calculation using the Hyperion radiative
        transfer code. As a result, the `Model.grid.temperature` list will be
        populated with the temperatures calculated.

        Args:
            :attr:`name` (`str`, optional):
                The name of the spectrum, to use as a key in the 
                :code:`Model.spectra` dictionary. Default: `None`
            :attr:`nphot` (int, optional):
                The number of photons to use for the calculation. Default: `1e6`
            :attr:`mrw` (bool, optional):
                If using `hyperion`, whether or not to use the Modified Random 
                Walk Algorithm. Default: `False`
            :attr:`pda` (bool, optional):
                If using `hyperion`, whether or not to use the Partial Diffusion
                Approximation algorithm. Default: `False`
            :attr:`niterations` (int, optional):
                The maximum number of iterations to perform, if convergence is
                not reached, before giving up. Default: `20`
            :attr:`percentile` (`float`, optional):
                Which percentile of cells to use when calculating the 
                convergence criteria. Default: `99.`
            :attr:`absolute` (`float`, optional):
                Maximum absolute difference of the ratio between cells for 
                convergence to be reached. Default: `2.0`
            :attr:`relative` (`float`, optional):
                Relative difference between cells for convergence to be reached.
                Default: `1.02`
            :attr:`max_interactions` (`int`, optional):
                Maximum number of interactions a photon can have before it is 
                killed. Default: `1e8`
            :attr:`mpi` (bool, optional):
                If using `hyperion`, whether or not to run the model in 
                parallel with MPI. Default: `False`
            :attr:`nprocesses` (bool, optional):
                If `mpi=True`, the number of MPI threads to use. Default: `None`
            :attr:`sublimation_temperature` (`float`, optional):
                If you would like to sublimate dust above a certain temperature,
                set that here. Default: `None`
            :attr:`verbose` (`bool`, False):
                Should output be printed to the screen, or hidden. 
                Default: `False`
            :attr:`incl` (`float`, optional):
                The inclination of the model to use for the image. Default: `0.`
            :attr:`pa` (`float`, optional):
                The position angle to use for the image. Default: `0.`
            :attr:`dpc` (`float`, optional):
                The distance to the model in units of parsecs. Default: `1.`
            :attr:`lam` (`float`, optional):
                The wavelength, in microns, to make the image at. Default: 
                `1300.`
            :attr:`nphot_imaging` (`float`, optional):
                The number of photons to use in making the image. Default: `1e6`
            :attr:`npix` (`int`, optional):
                The number of pixels (squared) that the image should have. 
                Default: `256`
            :attr:`pixelsize` (`float`, optional):
                The size of the pixels, in arcseconds. Default: `1.`
        """

        d = []
        for i in range(len(self.grid.dust)):
            d.append(IsotropicDust( \
                    self.grid.dust[i].nu[::-1].astype(numpy.float64), \
                    self.grid.dust[i].albedo[::-1].astype(numpy.float64), \
                    self.grid.dust[i].kext[::-1].astype(numpy.float64)))

            if sublimation_temperature != None:
                d[-1].set_sublimation_temperature('fast', \
                        temperature=sublimation_temperature)

        m = HypModel()
        if (self.grid.coordsystem == "cartesian"):
            m.set_cartesian_grid(self.grid.w1*AU, self.grid.w2*AU, \
                    self.grid.w3*AU)
        elif (self.grid.coordsystem == "cylindrical"):
            m.set_cylindrical_polar_grid(self.grid.w1*AU, self.grid.w3*AU, \
                    self.grid.w2)
        elif (self.grid.coordsystem == "spherical"):
            m.set_spherical_polar_grid(self.grid.w1*AU, self.grid.w2, \
                    self.grid.w3)

        for i in range(len(self.grid.density)):
            if (self.grid.coordsystem == "cartesian"):
                m.add_density_grid(numpy.transpose(self.grid.density[i], \
                        axes=(2,1,0)), d[i])
            if (self.grid.coordsystem == "cylindrical"):
                m.add_density_grid(numpy.transpose(self.grid.density[i], \
                        axes=(1,2,0)), d[i])
            if (self.grid.coordsystem == "spherical"):
                m.add_density_grid(numpy.transpose(self.grid.density[i], \
                        axes=(2,1,0)), d[i])

        sources = []
        for i in range(len(self.grid.stars)):
            sources.append(m.add_spherical_source())
            sources[i].luminosity = self.grid.stars[i].luminosity * L_sun
            sources[i].radius = self.grid.stars[i].radius * R_sun
            sources[i].temperature = self.grid.stars[i].temperature

        m.set_raytracing(True)
        
        if npix%2 == 0:
            zoomau = [-pixelsize*dpc*AU * (npix+1)/2, \
                    pixelsize*dpc*AU * (npix-1)/2, \
                    -pixelsize*dpc*AU * (npix+1)/2, \
                    pixelsize*dpc*AU * (npix-1)/2]
        else:
            zoomau = [-pixelsize*dpc*AU * npix/2, \
                    pixelsize*dpc*AU * npix/2, \
                    -pixelsize*dpc*AU * npix/2, \
                    pixelsize*dpc*AU * npix/2]

        m.set_monochromatic(True, wavelengths=[lam])
        image = m.add_peeled_images(sed=False, image=True)
        image.set_viewing_angles([incl], [pa])
        image.set_track_origin(track_origin)
        image.set_image_size(npix, npix)
        image.set_image_limits(*tuple(zoomau))

        m.set_mrw(mrw)
        m.set_pda(pda)
        m.set_max_interactions(max_interactions)
        m.set_n_initial_iterations(niterations)
        m.set_n_photons(initial=nphot, raytracing_sources=1e4, \
                raytracing_dust=nphot_imaging, imaging_sources=nphot_imaging, \
                imaging_dust=nphot_imaging)
        m.conf.output.output_density = 'last'
        m.set_convergence(True, percentile=percentile, absolute=absolute, \
                relative=relative)

        m.write("temp.rtin")

        if verbose:
            m.run("temp.rtout", mpi=mpi, n_processes=nprocesses, overwrite=True)
        else:
            m.run("temp.rtout", mpi=mpi, n_processes=nprocesses, \
                    overwrite=True, logfile="temp.log")

        n = ModelOutput("temp.rtout")

        image = n.get_image(inclination=0, distance=dpc*pc, units='Jy')

        self.images[name] = Image(image.val.reshape(image.val.shape+(1,)), \
                wave=image.wav*1.0e-4)

        if track_origin == "detailed":
            for component in ["source_emit","source_scat","dust_emit", \
                    "dust_scat"]:
                image = n.get_image(inclination=0, distance=dpc*pc, \
                        units='Jy', component=component)

                self.images[name+"-"+component] = Image(image.val.reshape(\
                        image.val.shape+(1,)), wave=image.wav*1.0e-4)

        os.system("rm temp.rtin temp.rtout temp.log")


    def run_image_radmc3d(self, name=None, nphot=1e6, npix=256, pixelsize=1.0, \
            lam="1300", loadlambda=False, imolspec=None, iline=None,  \
            widthkms=None, vkms=None, linenlam=None, doppcatch=False, \
            incl=0, pa=0, phi=0, dpc=1, verbose=True, nice=None, \
            unstructured=False, nostar=False, **keywords):
        self.write_radmc3d(nphot_scat=nphot, **keywords)

        if npix%2 == 0:
            zoomau = [-pixelsize*dpc * (npix+1)/2, pixelsize*dpc * (npix-1)/2, \
                    -pixelsize*dpc * (npix+1)/2, pixelsize*dpc * (npix-1)/2]
        else:
            zoomau = [-pixelsize*dpc * npix/2, pixelsize*dpc * npix/2, \
                    -pixelsize*dpc * npix/2, pixelsize*dpc * npix/2]

        radmc3d.run.image(npix=npix, zoomau=zoomau, lam=lam, \
                loadlambda=loadlambda, imolspec=imolspec, iline=iline, \
                widthkms=widthkms, vkms=vkms, linenlam=linenlam, \
                doppcatch=doppcatch, incl=incl, posang=pa, phi=phi, \
                verbose=verbose, nice=nice, circ=unstructured, \
                nostar=nostar)

        if unstructured:
            if 'writeimage_unformatted' in keywords:
                image, r, phi, lam = radmc3d.read.circimage(\
                        binary=keywords["writeimage_unformatted"])
            else:
                image, r, phi, lam = radmc3d.read.circimage()

            r, phi = numpy.meshgrid(r, phi)
            
            r = r / (dpc*pc) / arcsec

            x = -r*numpy.cos(phi)
            y = r*numpy.sin(phi)

            x = numpy.concatenate(([x[0,0]], x[:,1:].reshape((x[:,1:].size,))))
            y = numpy.concatenate(([y[0,0]], y[:,1:].reshape((y[:,1:].size,))))
            image = numpy.concatenate((image[0:1,0,:,0], image[:,1:,:,0].\
                    reshape((image[:,1:,0,0].size,image.shape[2]))))

            image = image / Jy

            self.images[name] = UnstructuredImage(image, x=x, y=y, \
                    wave=lam*1.0e-4)
        else:
            if 'writeimage_unformatted' in keywords:
                image, x, y, lam = radmc3d.read.image(\
                        binary=keywords["writeimage_unformatted"])
            else:
                image, x, y, lam = radmc3d.read.image()

            image = image / Jy * ((x[1] - x[0]) / (dpc * pc)) * \
                    ((y[1] - y[0]) / (dpc * pc))

            x = (x - x[int(npix/2)]) * pixelsize / (x[1] - x[0])
            y = (y - y[int(npix/2)]) * pixelsize / (y[1] - y[0])

            self.images[name] = Image(image, x=x, y=y, wave=lam*1.0e-4)

        os.system("rm *.out *.inp *.dat")
        if 'writeimage_unformatted' in keywords:
            if keywords['writeimage_unformatted']:
                os.system("rm *.bout")

    def run_sed(self, name=None, nphot=1e6, code="radmc3d", **keywords):
        r"""
        Run a spectrum of the model.

        Args:
            :attr:`name` (`str`, optional):
                The name of the spectrum, to use as a key in the 
                :code:`Model.spectra` dictionary. Default: `None`
            :attr:`nphot` (int, optional):
                The number of photons to use for the calculation. Default: `1e6`
            :attr:`code` (str, optional):
                Which underlying radiative transfer modeling code to use for 
                the radiative equilibrium calculation. `radmc3d` or `hyperion`. 
                Default: `radmc3d`
            :attr:`kwargs` (optional):
                Can be used to pass arguments to either 
                :code:`Model.run_sed_hyperion` or 
                :code:`Model.run_sed_radmc3d`, depending on the value of 
                :code:`code`.
        """

        if (code == "radmc3d"):
            self.run_sed_radmc3d(name=name, nphot=nphot, **keywords)
        else:
            self.run_sed_hyperion(name=name, nphot=nphot, **keywords)

    def run_sed_hyperion(self, name=None, nphot=1e6):
        return

    def run_sed_radmc3d(self, name=None, nphot=1e6, incl=0, pa=0, \
            phi=0, dpc=1, loadlambda=False, verbose=True, nice=None, \
            **keywords):
        r"""
        Run a spectrum of the model with RADMC-3D. As a result, the 
        :code:`Model.spectra` dictionary will have a 
        :code:`pdspy.spectroscopy.Spectrum` added with key `name`.

        Args:
            :attr:`name` (`str`, optional):
                The name of the spectrum, to use as a key in the 
                :code:`Model.spectra` dictionary. Default: `None`
            :attr:`nphot` (int, optional):
                The number of photons to use for the calculation. Default: `1e6`
            :attr:`incl` (`float`, optional):
                The inclination of the model to use for the image. Default: `0.`
            :attr:`loadlambda` (`bool`, optional):
                If `True`, use the :code:`Model.camera_wavelength` array as the
                list of wavelengths for the scattering phase function 
                calculation. If `False`, use the :code:`Model.grid.lam array`. 
                Default: `False`
            :attr:`code` (str, optional):
                Which underlying radiative transfer modeling code to use for 
                the radiative equilibrium calculation. `radmc3d` or `hyperion`. 
                Default: `radmc3d`
            :attr:`kwargs` (optional):
                Can be used to pass arguments to RADMC-3D.
                This can be used to pass any options to RADMC-3D. For a list of
                all possibilities for spectra, check the RADMC-3D documentation.
        """

        self.write_radmc3d(nphot_spec=nphot, **keywords)

        radmc3d.run.sed(incl=incl, posang=pa, phi=phi, noline=True, \
                loadlambda=loadlambda, verbose=verbose, nice=nice)

        flux, lam = radmc3d.read.spectrum()

        flux = flux / Jy * (1. / dpc)**2

        self.spectra[name] = Spectrum(wave=lam, flux=flux)

        os.system("rm *.out *.inp *.dat")

    def run_visibilities(self, name=None, nphot=1e6, code="radmc3d", \
            **keywords):
        r"""
        Run visibilities for the model.

        Args:
            :attr:`name` (`str`, optional):
                The name of the visibilities, to use as a key in the 
                :code:`Model.visibilities` dictionary. Default: `None`
            :attr:`nphot` (int, optional):
                The number of photons to use for the calculation. Default: `1e6`
            :attr:`code` (str, optional):
                Which underlying radiative transfer modeling code to use for 
                the radiative equilibrium calculation. `radmc3d` or `hyperion`. 
                Default: `radmc3d`
            :attr:`kwargs` (optional):
                Can be used to pass arguments to either 
                :code:`Model.run_scattering_hyperion` or 
                :code:`Model.run_scattering_radmc3d`, depending on the value of 
                :code:`code`.
        """

        if (code == "radmc3d"):
            self.run_visibilities_radmc3d(name=name, nphot=nphot, **keywords)
        else:
            self.run_visibilities_hyperion(name=name, nphot=nphot, **keywords)

    def run_visibilities_hyperion(self, name=None, nphot=1e6):
        return

    def run_visibilities_radmc3d(self, name=None, nphot=1e6, npix=256, \
            pixelsize=1.0, lam="1300", loadlambda=False, imolspec=None, \
            iline=None,  widthkms=None, vkms=None, linenlam=None, \
            doppcatch=False, incl=0, pa=0, phi=0, dpc=1, verbose=True, \
            nice=None, nostar=False, **keywords):
        self.write_radmc3d(nphot_scat=nphot, **keywords)

        if npix%2 == 0:
            zoomau = [-pixelsize*dpc * (npix+1)/2, pixelsize*dpc * (npix-1)/2, \
                    -pixelsize*dpc * (npix+1)/2, pixelsize*dpc * (npix-1)/2]
        else:
            zoomau = [-pixelsize*dpc * npix/2, pixelsize*dpc * npix/2, \
                    -pixelsize*dpc * npix/2, pixelsize*dpc * npix/2]

        radmc3d.run.image(npix=npix, zoomau=zoomau, lam=lam, \
                loadlambda=loadlambda, imolspec=imolspec, iline=iline, \
                widthkms=widthkms, vkms=vkms, linenlam=linenlam, \
                doppcatch=doppcatch, incl=incl, posang=pa, phi=phi, \
                verbose=verbose, nostar=nostar, nice=nice)

        if 'writeimage_unformatted' in keywords:
            image, x, y, lam = radmc3d.read.image(\
                    binary=keywords["writeimage_unformatted"])
        else:
            image, x, y, lam = radmc3d.read.image()

        image = image / Jy * ((x[1] - x[0]) / (dpc * pc)) * \
                ((y[1] - y[0]) / (dpc * pc))

        x = x * pixelsize / (x[1] - x[0])
        y = y * pixelsize / (y[1] - y[0])

        im = Image(image, x=x, y=y, wave=lam*1.0e-4)

        self.visibilities[name] = imtovis(im)

        os.system("rm *.out *.inp *.dat")
        if 'writeimage_unformatted' in keywords:
            if keywords['writeimage_unformatted']:
                os.system("rm *.bout")

    def write_radmc3d(self, **keywords):
        radmc3d.write.control(**keywords)

        mstar = []
        rstar = []
        xstar = []
        ystar = []
        zstar = []
        tstar = []

        for i in range(len(self.grid.stars)):
            mstar.append(self.grid.stars[i].mass*M_sun)
            rstar.append(self.grid.stars[i].radius*R_sun)
            xstar.append(self.grid.stars[i].x*AU)
            ystar.append(self.grid.stars[i].y*AU)
            zstar.append(self.grid.stars[i].z*AU)
            tstar.append(self.grid.stars[i].temperature)

        radmc3d.write.stars(rstar, mstar, self.grid.lam, xstar, ystar, zstar, \
                tstar=tstar)

        radmc3d.write.wavelength_micron(self.grid.lam)
        if hasattr(self, "camera_wavelength"):
            radmc3d.write.camera_wavelength_micron(self.camera_wavelength)

        if (self.grid.coordsystem == "cartesian"):
            radmc3d.write.amr_grid(self.grid.w1*AU, self.grid.w2*AU, \
                    self.grid.w3*AU, coordsystem=self.grid.coordsystem)
        elif(self.grid.coordsystem == "cylindrical"):
            radmc3d.write.amr_grid(self.grid.w1*AU, self.grid.w2, \
                    self.grid.w3*AU, coordsystem=self.grid.coordsystem)
        elif(self.grid.coordsystem == "spherical"):
            radmc3d.write.amr_grid(self.grid.w1*AU, self.grid.w2, \
                    self.grid.w3, coordsystem=self.grid.coordsystem)

        radmc3d.write.dust_density(self.grid.density)
        if len(self.grid.temperature) > 0:
            density = numpy.array(self.grid.density)
            temperature = numpy.array(self.grid.temperature)

            density[density == 0] = 1.0e-30

            temperature = (density * temperature).sum(axis=0) / \
                    density.sum(axis=0)
            temperature = [temperature for i in range(len(self.grid.density))]

            radmc3d.write.dust_temperature(temperature)

        dustopac = []
        for i in range(len(self.grid.dust)):
            dustopac.append("dustkappa_{0:d}.inp".format(i))
            radmc3d.write.dustkappa("{0:d}".format(i), \
                    self.grid.dust[i].lam*1.0e4, self.grid.dust[i].kabs, \
                    ksca=self.grid.dust[i].ksca)

        radmc3d.write.dustopac(dustopac)

        if len(self.grid.gas) > 0:
            gas = []
            inpstyle = []
            colpartners = []
            for i in range(len(self.grid.gas)):
                gas.append("{0:d}".format(i))
                inpstyle.append("leiden")
                colpartners.append([])
                radmc3d.write.molecule(self.grid.gas[i], gas[i])
                radmc3d.write.numberdens(self.grid.number_density[i], gas[i])

            radmc3d.write.line(gas, inpstyle, colpartners)

            number_density = numpy.array(self.grid.number_density)
            velocity = numpy.array(self.grid.velocity)
            vx = velocity[:,0,:,:,:]
            vy = velocity[:,1,:,:,:]
            vz = velocity[:,2,:,:,:]
            velocity = numpy.zeros(self.grid.velocity[0].shape)

            nx, ny, nz = self.grid.number_density[0].shape

            number_density[number_density == 0] = 1.0e-50

            velocity[0,:,:,:] = (number_density * vx).sum(axis=0) / \
                    number_density.sum(axis=0)
            velocity[1,:,:,:] = (number_density * vy).sum(axis=0) / \
                    number_density.sum(axis=0)
            velocity[2,:,:,:] = (number_density * vz).sum(axis=0) / \
                    number_density.sum(axis=0)

            radmc3d.write.gas_velocity(velocity)

            if len(self.grid.gas_temperature) > 0:
                gas_temperature = numpy.array(self.grid.gas_temperature)
                gas_temperature = (number_density * gas_temperature).\
                        sum(axis=0) / number_density.sum(axis=0)

                radmc3d.write.gas_temperature(gas_temperature)

            if len(self.grid.microturbulence) > 0:
                microturbulence = numpy.array(self.grid.microturbulence)
                microturbulence = (number_density * microturbulence).\
                        sum(axis=0) / number_density.sum(axis=0)

                radmc3d.write.microturbulence(microturbulence)

    def read(self, filename=None, usefile=None):
        r"""
        Read a model in from an HDF5 model file.

        Args:
            :attr:`filename` (`str`, optional):
                The filename of the file storing the model to read in. 
                Default: `None`
            :attr:`usefile` (`h5py.File`, optional):
                If :code:`filename = None`, then use this keyword to provide an
                instance of an :code:`h5py.File` that has already be opened 
                that the model can be read from.
        """

        if (usefile == None):
            f = h5py.File(filename, "r")
        else:
            f = usefile

        if ('Grid' in f):
            self.grid = Grid()
            self.grid.read(usefile=f['Grid'])

        if ('Images' in f):
            for image in f['Images']:
                try:
                    self.images[image] = Image()
                    self.images[image].read(usefile=f['Images'][image])
                except:
                    self.images[image] = UnstructuredImage()
                    self.images[image].read(usefile=f['Images'][image])

        if ('Spectra' in f):
            for spectrum in f['Spectra']:
                self.spectra[spectrum] = Spectrum()
                self.spectra[spectrum].read(usefile=f['Spectra'][spectrum])

        if ('Visibilities' in f):
            for visibility in f['Visibilities']:
                self.visibilities[visibility] = Visibilities()
                self.visibilities[visibility].read( \
                        usefile=f['Visibilities'][visibility])

        if (usefile == None):
            f.close()

    def write(self, filename=None, usefile=None):
        r"""
        Write a model to an HDF5 model file.

        Args:
            :attr:`filename` (`str`, optional):
                The filename of the file that will be written out with the 
                model. Default: `None`
            :attr:`usefile` (`h5py.File`, optional):
                If :code:`filename = None`, then use this keyword to provide an
                instance of an :code:`h5py.File` that has already be opened 
                that the model can be written to.
        """

        if (usefile == None):
            f = h5py.File(filename, "w")
        else:
            f = usefile

        grid = f.create_group("Grid")
        if hasattr(self, 'grid'):
            self.grid.write(usefile=grid)

        spectra = f.create_group("Spectra")
        for spectrum in self.spectra:
            spectra.create_group(spectrum)
            self.spectra[spectrum].write(usefile=spectra[spectrum])

        images = f.create_group("Images")
        for image in self.images:
            images.create_group(image)
            self.images[image].write(usefile=images[image])

        visibilities = f.create_group("Visibilities")
        for visibility in self.visibilities:
            visibilities.create_group(visibility)
            self.visibilities[visibility].write( \
                    usefile=visibilities[visibility])

        if (usefile == None):
            f.close()
