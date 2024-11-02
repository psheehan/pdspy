import numpy
import h5py
from .Model import Model
from .Star import Star
from ..constants.physics import c
import tempfile
from ..interferometry import interpolate_model
import os

class RMLModel(Model):
    r"""
    A Model that specifically represents a young star, including a star, disk, and envelope.
    """

    inclination = 0
    pa = 0

    star_luminosity = 1

    def add_star(self, luminosity=1):
        self.star_luminosity = luminosity

        self.grid.add_star(Star(mass=0.5, luminosity=luminosity, \
                temperature=4000.))

    def set_cartesian_grid(self, xmin, xmax, nx):
        x = numpy.linspace(xmin, xmax, nx)
        y = numpy.linspace(xmin, xmax, nx)
        z = numpy.linspace(xmin, xmax, nx)

        self.grid.set_cartesian_grid(x, y, z)

    def set_cylindrical_grid(self, rmin, rmax, nr, nz, nphi):
        r = numpy.linspace(rmin, rmax, nr)
        phi = numpy.linspace(0.0, 2*numpy.pi, nphi)
        z = numpy.linspace(0.,rmax, nz)

        self.grid.set_cylindrical_grid(r, phi, z)

    def set_spherical_grid(self, rmin, rmax, nr, ntheta, nphi, log=True, \
            code="radmc3d"):
        if log:
            r = numpy.logspace(numpy.log10(rmin), numpy.log10(rmax), nr)
        else:
            r = numpy.linspace(rmin, rmax, nr)
        if (code == "hyperion"):
            r = numpy.hstack([0.0,r])

        if (code == "radmc3d"):
            theta = numpy.linspace(0.0, numpy.pi/2, ntheta)
        elif (code == "hyperion"):
            theta = numpy.linspace(0.0, numpy.pi, ntheta)

        phi = numpy.linspace(0.0, 2*numpy.pi, nphi)

        self.grid.set_spherical_grid(r, theta, phi)

    def add_dust_layer(self, dust):
        density = numpy.ones((self.grid.w1.size-1, self.grid.w2.size-1, self.grid.w3.size-1))*1e-20

        self.grid.add_density(density, dust)

    def set_parameter_vector(self, p, nphotons=1e6, nprocesses=1, verbose=False):
        self.inclination = p[0]
        self.pa = p[1]
        self.star_luminosity = p[2]

        self.grid.stars = [Star(mass=0.5, luminosity=self.star_luminosity, \
                temperature=4000.)]

        self.grid.density = list(p[3:].reshape((len(self.grid.density),)+self.grid.density[0].shape))

        self.run_thermal(code="radmc3d", nphot=nphotons, \
                modified_random_walk=True,\
                mrw_gamma=2, mrw_tauthres=10, mrw_count_trigger=100, \
                verbose=verbose, setthreads=nprocesses)

    def get_parameter_vector(self):
        return numpy.concatenate(([self.inclination, self.pa, self.star_luminosity], numpy.array(self.grid.density).flatten()))

    def make_hyperion_symmetric(self):
        for i in range(len(self.grid.temperature)):
            ntheta = len(self.grid.theta)
            upper = self.grid.temperature[i][:,0:int(ntheta/2),:]
            lower = self.grid.temperature[i][:,int(ntheta/2):,:][:,::-1,:]
            average = 0.5 * (upper + lower)

            self.grid.temperature[i][:,0:int(ntheta/2),:] = average
            self.grid.temperature[i][:,int(ntheta/2):,:] =  average[:,::-1,:]

    def convert_hyperion_to_radmc3d(self):
        self.grid.r = self.grid.r[1:]
        self.grid.w1 = self.grid.w1[1:]

        ntheta = len(self.grid.theta)
        self.grid.theta = self.grid.theta[0:int(ntheta/2)]
        self.grid.w2 = self.grid.w2[0:int(ntheta/2+1)]
        
        for i in range(len(self.grid.density)):
            self.grid.density[i] = self.grid.density[i][1:,0:int(ntheta/2),:]
        for i in range(len(self.grid.temperature)):
            self.grid.temperature[i] = self.grid.temperature[i][1:,0:int(ntheta/2),:]
        for i in range(len(self.grid.number_density)):
            self.grid.number_density[i] = self.grid.number_density[i][1:,0:int(ntheta/2),:]
        for i in range(len(self.grid.microturbulence)):
            self.grid.microturbulence[i] = self.grid.microturbulence[i][1:,0:int(ntheta/2),:]
        for i in range(len(self.grid.velocity)):
            self.grid.velocity[i] = self.grid.velocity[i][:,1:,0:int(ntheta/2),:]

    # Define a likelihood function.

    def log_likelihood(self, visibilities, images, spectra, \
            nprocesses=1, source="ObjName", nice=19, verbose=False, \
            ftcode="galario"):

        original_dir = os.environ["PWD"]
        temp_dir = tempfile.TemporaryDirectory()
        os.chdir(temp_dir.name)

        for j in range(len(visibilities["file"])):
            # Set the wavelengths for RADMC3D to use.

            wave = c / visibilities["data"][j].freq / 1.0e-4
            self.set_camera_wavelength(wave)

            if ftcode == "galario":
                self.run_image(name=visibilities["lam"][j], nphot=1e5, \
                        npix=visibilities["npix"][j], \
                        pixelsize=visibilities["pixelsize"][j], \
                        lam=None, loadlambda=True, incl=self.inclination, \
                        pa=self.pa, dpc=140., code="radmc3d", \
                        mc_scat_maxtauabs=5, verbose=verbose,setthreads=nprocesses,\
                        writeimage_unformatted=True, nice=nice)
            else:
                self.run_image(name=visibilities["lam"][j], nphot=1e5, \
                        lam=None, loadlambda=True, incl=self.inclination, \
                        pa=self.pa, dpc=140, code="radmc3d", \
                        mc_scat_maxtauabs=5, verbose=verbose,setthreads=nprocesses,\
                        writeimage_unformatted=True, nice=nice, unstructured=True,\
                        camera_circ_nrphiinf=visibilities["nphi"][j], \
                        camera_circ_dbdr=visibilities["nr"][j])

            # Account for the flux calibration uncertainties.

            self.visibilities[visibilities["lam"][j]] = interpolate_model(\
                    visibilities["data"][j].u, visibilities["data"][j].v, \
                    visibilities["data"][j].freq, self.images[visibilities["lam"][j]],\
                    dRA=0., dDec=0., nthreads=nprocesses, code=ftcode, \
                    nxy=visibilities["npix"][j], dxy=visibilities["pixelsize"][j])

        # Run the images.

        for j in range(len(images["file"])):
            self.run_image(name=images["lam"][j], nphot=1e5, \
                    npix=images["npix"][j], pixelsize=images["pixelsize"][j], \
                    lam=images["lam"][j], incl=self.inclination, \
                    pa=self.pa, dpc=140, code="radmc3d", \
                    mc_scat_maxtauabs=5, verbose=verbose, setthreads=nprocesses, \
                    nice=nice)

            # Convolve with the beam.

            x, y = numpy.meshgrid(numpy.linspace(-256,255,512), \
                    numpy.linspace(-256,255,512))

            beam = misc.gaussian2d(x, y, 0., 0., images["bmaj"][j]/2.355/\
                    images["pixelsize"][j], images["bmin"][j]/2.355/\
                    image["pixelsize"][j], (90-images["bpa"][j])*numpy.pi/180., 1.0)

            m.images[images["lam"][j]].image = scipy.signal.fftconvolve(\
                    self.images[images["lam"][j]].image[:,:,0,0], beam, mode="same").\
                    reshape(self.images[images["lam"][j]].image.shape)

        # Run the SED.

        if "total" in spectra:
            self.set_camera_wavelength(spectra["total"].wave)

            self.run_sed(name="SED", nphot=1e4, loadlambda=True, incl=self.inclination,\
                    pa=self.pa, dpc=140, code="radmc3d", \
                    camera_scatsrc_allfreq=True, mc_scat_maxtauabs=5, \
                    verbose=verbose, setthreads=nprocesses, nice=nice)

        os.chdir(original_dir)

        # A list to put all of the chisq into.

        chisq = []

        # Calculate the chisq for the visibilities.

        for j in range(len(visibilities["file"])):
            good = visibilities["data"][j].weights > 0

            chisq.append(-0.5*numpy.sum((visibilities["data"][j].real - \
                    self.visibilities[visibilities["lam"][j]].real)**2 * \
                    visibilities["data"][j].weights) - \
                    numpy.sum(numpy.log(visibilities["data"][j].weights[good]/ \
                    (2*numpy.pi))) + \
                    -0.5*numpy.sum((visibilities["data"][j].imag - \
                    self.visibilities[visibilities["lam"][j]].imag)**2 * \
                    visibilities["data"][j].weights) - \
                    numpy.sum(numpy.log(visibilities["data"][j].weights[good]/ \
                    (2*numpy.pi))))


        # Calculate the chisq for all of the images.

        for j in range(len(images["file"])):
            chisq.append(-0.5 * (numpy.sum((images["data"][j].image - \
                    self.images[images["lam"][j]].image)**2 / \
                    images["data"][j].unc**2)))

        # Calculate the chisq for the SED.

        if "total" in spectra:
            chisq.append(-0.5 * (numpy.sum((spectra["total"].flux - \
                    self.spectra["SED"].flux)**2 / spectra["total"].unc**2)))

        # Return the sum of the chisq.

        return numpy.array(chisq).sum()

