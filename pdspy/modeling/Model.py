import numpy
import h5py
import os
try:
    from hyperion.dust import IsotropicDust
    from hyperion.model import Model as HypModel
    from hyperion.model import ModelOutput
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

    def __init__(self):
        self.grid = Grid()
        self.images = {}
        self.spectra = {}
        self.visibilities = {}

    def set_camera_wavelength(self, lam):
        self.camera_wavelength = lam

    def run_thermal(self, nphot=1e6, code="radmc3d", **keywords):
        if (code == "radmc3d"):
            self.run_thermal_radmc3d(nphot=nphot, **keywords)
        else:
            self.run_thermal_hyperion(nphot=nphot, **keywords)

    def run_thermal_hyperion(self, nphot=1e6, mrw=False, pda=False, \
            niterations=20, percentile=99., absolute=2.0, relative=1.02, \
            max_interactions=1e8, mpi=False, nprocesses=None, \
            sublimation_temperature=None, verbose=True):
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
        m.set_n_photons(initial=nphot, imaging=0)
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
        if (code == "radmc3d"):
            self.run_scattering_radmc3d(nphot=nphot, **keywords)
        else:
            print("Scattering phase function cannot be calculated in Hyperion!")

    def run_scattering_radmc3d(self, nphot=1e6, verbose=True, nice=None, \
            loadlambda=None, **keywords):
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
        if (code == "radmc3d"):
            self.run_sed_radmc3d(name=name, nphot=nphot, **keywords)
        else:
            self.run_sed_hyperion(name=name, nphot=nphot, **keywords)

    def run_sed_hyperion(self, name=None, nphot=1e6):
        return

    def run_sed_radmc3d(self, name=None, nphot=1e6, incl=0, pa=0, \
            phi=0, dpc=1, loadlambda=False, verbose=True, nice=None, \
            **keywords):
        self.write_radmc3d(nphot_spec=nphot, **keywords)

        radmc3d.run.sed(incl=incl, posang=pa, phi=phi, noline=True, \
                loadlambda=loadlambda, verbose=verbose, nice=nice)

        flux, lam = radmc3d.read.spectrum()

        flux = flux / Jy * (1. / dpc)**2

        self.spectra[name] = Spectrum(wave=lam, flux=flux)

        os.system("rm *.out *.inp *.dat")

    def run_visibilities(self, name=None, nphot=1e6, code="radmc3d", \
            **keywords):
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
