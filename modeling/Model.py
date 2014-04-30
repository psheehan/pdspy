import numpy
import h5py
import os
from hyperion.dust import IsotropicDust
from hyperion.model import Model as HypModel
from hyperion.model import ModelOutput
from .. import radmc3d
from .Grid import Grid
from ..imaging import Image, imtovis
from ..interferometry import Visibilities
from ..spectroscopy import Spectrum
from ..constants.astronomy import AU, M_sun, R_sun, L_sun, Jy, arcsec, pc
from ..constants.physics import c

class Model:

    def __init__(self):
        self.grid = Grid()
        self.images = {}
        self.spectra = {}
        self.visibilities = {}

    def run_thermal(self, nphot=1e6, code="radmc3d", **keywords):
        if (code == "radmc3d"):
            self.run_thermal_radmc3d(nphot=nphot, **keywords)
        else:
            self.run_thermal_hyperion(nphot=nphot, **keywords)

    def run_thermal_hyperion(self, nphot=1e6, mrw=False, pda=False, \
            niterations=20, percentile=99., absolute=2.0, relative=1.02, \
            max_interactions=1e8, mpi=False, nprocesses=None):
        d = []
        for i in range(len(self.grid.dust)):
            d.append(IsotropicDust( \
                    self.grid.dust[i].nu[::-1].astype(numpy.float64), \
                    self.grid.dust[i].albedo[::-1].astype(numpy.float64), \
                    self.grid.dust[i].kext[::-1].astype(numpy.float64)))

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

        m.set_mrw(mrw)
        m.set_pda(pda)
        m.set_max_interactions(max_interactions)
        m.set_n_initial_iterations(niterations)
        m.set_n_photons(initial=nphot, imaging=0)
        m.set_convergence(True, percentile=percentile, absolute=absolute, \
                relative=relative)

        m.write("temp.rtin")

        m.run("temp.rtout", mpi=mpi, n_processes=nprocesses)

        n = ModelOutput("temp.rtout")

        grid = n.get_quantities()

        self.grid.temperature = []
        temperature = grid.quantities['temperature']
        for i in range(len(temperature)):
            if (self.grid.coordsystem == "cartesian"):
                self.grid.temperature.append(numpy.transpose(temperature[i], \
                        axes=(2,1,0)))
            if (self.grid.coordsystem == "cylindrical"):
                self.grid.temperature.append(numpy.transpose(temperature[i], \
                        axes=(2,0,1)))
            if (self.grid.coordsystem == "spherical"):
                self.grid.temperature.append(numpy.transpose(temperature[i], \
                        axes=(2,1,0)))

        os.system("rm temp.rtin temp.rtout")

    def run_thermal_radmc3d(self, nphot=1e6, **keywords):
        self.write_radmc3d(nphot_therm=nphot, **keywords)

        radmc3d.run.thermal()

        self.grid.temperature = radmc3d.read.dust_temperature()
        for i in range(len(self.grid.temperature)):
            n1, n2, n3 = self.grid.density[i].shape
            self.grid.temperature[i] = numpy.transpose( \
                    self.grid.temperature[i].reshape((n3,n2,n1)), \
                    axes=(2,1,0))

        os.system("rm *.out *.inp *.dat")

    def run_image(self, name=None, nphot=1e6, code="radmc3d", **keywords):
        if (code == "radmc3d"):
            self.run_image_radmc3d(name=name, nphot=nphot, **keywords)
        else:
            self.run_image_hyperion(name=name, nphot=nphot, **keywords)

    def run_image_hyperion(self, name=None, nphot=1e6):
        return

    def run_image_radmc3d(self, name=None, nphot=1e6, npix=256, pixelsize=1.0, \
            lam="1300", imolspec=None, iline=None,  widthkms=None, \
            linenlam=None, doppcatch=False, incl=0, pa=0, phi=0, dpc=1, \
            **keywords):
        self.write_radmc3d(nphot_scat=nphot, **keywords)

        if npix%2 == 0:
            zoomau = [-pixelsize*dpc * (npix+1)/2, pixelsize*dpc * (npix-1)/2, \
                    -pixelsize*dpc * (npix+1)/2, pixelsize*dpc * (npix-1)/2]
        else:
            zoomau = [-pixelsize*dpc * npix/2, pixelsize*dpc * npix/2, \
                    -pixelsize*dpc * npix/2, pixelsize*dpc * npix/2]

        sizeau=pixelsize*npix*dpc

        if iline != None:
            radmc3d.run.image(npix=npix, sizeau=sizeau, lam=lam, \
                    imolspec=imolspec, iline=iline, widthkms=widthkms, \
                    linenlam=linenlam, doppcatch=doppcatch, incl=incl, \
                    posang=pa, phi=phi)
        else:
            radmc3d.run.image(npix=npix, zoomau=zoomau, lam=lam, \
                    imolspec=imolspec, iline=iline, widthkms=widthkms, \
                    linenlam=linenlam, doppcatch=doppcatch, incl=incl, \
                    posang=pa, phi=phi)

        image, x, y, lam = radmc3d.read.image()

        image = image / Jy * ((x[1] - x[0]) / (dpc * pc)) * \
                ((y[1] - y[0]) / (dpc * pc))
        if iline != None:
            x = x / (dpc * pc) / arcsec
            y = y / (dpc * pc) / arcsec
        else:
            x = (x - x[int(npix/2)]) / (dpc * pc) / arcsec
            y = (y - y[int(npix/2)]) / (dpc * pc) / arcsec

        self.images[name] = Image(image, x=x, y=y, wave=lam*1.0e-4)

        os.system("rm *.out *.inp *.dat")

    def run_sed(self, name=None, nphot=1e6, code="radmc3d", **keywords):
        if (code == "radmc3d"):
            self.run_sed_radmc3d(name=name, nphot=nphot, **keywords)
        else:
            self.run_sed_hyperion(name=name, nphot=nphot, **keywords)

    def run_sed_hyperion(self, name=None, nphot=1e6):
        return

    def run_sed_radmc3d(self, name=None, nphot=1e6, incl=0, pa=0, \
            phi=0, dpc=1, **keywords):
        self.write_radmc3d(nphot_spec=nphot, **keywords)

        radmc3d.run.sed(incl=incl, posang=pa, phi=phi, noline=True)

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
            pixelsize=1.0, lam="1300", imolspec=None, iline=None,  \
            widthkms=None, linenlam=None, doppcatch=False, incl=0, pa=0, \
            phi=0, dpc=1, **keywords):
        self.write_radmc3d(nphot_scat=nphot, **keywords)

        if npix%2 == 0:
            zoomau = [-pixelsize*dpc * (npix+1)/2, pixelsize*dpc * (npix-1)/2, \
                    -pixelsize*dpc * (npix+1)/2, pixelsize*dpc * (npix-1)/2]
        else:
            zoomau = [-pixelsize*dpc * npix/2, pixelsize*dpc * npix/2, \
                    -pixelsize*dpc * npix/2, pixelsize*dpc * npix/2]

        radmc3d.run.image(npix=npix, zoomau=zoomau, lam=lam, imolspec=imolspec,\
                iline=iline, widthkms=widthkms, linenlam=linenlam, \
                doppcatch=doppcatch, incl=incl, posang=pa, phi=phi)

        image, x, y, lam = radmc3d.read.image()

        image = image / Jy * ((x[1] - x[0]) / (dpc * pc)) * \
                ((y[1] - y[0]) / (dpc * pc))
        x = x / (dpc * pc) / arcsec
        y = y / (dpc * pc) / arcsec

        im = Image(image, x=x, y=y, wave=lam*1.0e-4)

        self.visibilities[name] = imtovis(im)

        os.system("rm *.out *.inp *.dat")

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
            radmc3d.write.dust_temperature(self.grid.temperature)

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

            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        index = numpy.where(number_density[:,i,j,k] == \
                                number_density[:,i,j,k].max())[0][0]
                        velocity[0,i,j,k] = vx[index,i,j,k]
                        velocity[1,i,j,k] = vy[index,i,j,k]
                        velocity[2,i,j,k] = vz[index,i,j,k]

            radmc3d.write.gas_velocity(velocity)

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
                self.images[image] = Image()
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
