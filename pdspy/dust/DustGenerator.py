import numpy
import scipy
import h5py
import os
from .Dust import Dust
from .mix_dust import mix_dust

class DustGenerator:
    recipes = ["draine","pollack","diana","diana_wice"]

    def __init__(self, dust, with_dhs=False, fmax=0.8, nf=50, singlesize=False):
        if type(dust) == str:
            if dust in self.recipes:
                recipe_dict = {
                    "draine":{
                        "dust":["astronomical_silicates","graphite_parallel_0.01","graphite_perpendicular_0.01"],
                        "format":["draine","draine","draine"],
                        "density":numpy.array([3.3,2.24,2.24]),
                        "abundance":numpy.array([0.65,0.35*1./3,0.35*2./3])
                    },
                    "dsharp":{
                        "dust":["astronomical_silicates","troilite","organics","water_ice"],
                        "format":["draine","henn","henn","henn"],
                        "density":numpy.array([3.3,4.83,1.5,0.92]),
                        "abundance":numpy.array([0.1670,0.0258,0.4430,0.3642])
                    },
                    "pollack":{
                        "dust":["astronomical_silicates","troilite","organics","water_ice"],
                        "format":["draine","henn","henn","henn"],
                        "density":numpy.array([3.3,4.83,1.5,0.92]),
                        "mass_fraction":numpy.array([3.41e-3,7.68e-4,4.13e-3,5.55e-3]),
                    },
                    "diana":{
                        "dust":["amorphous_silicates_extrapolated","amorphous_carbon_zubko1996_extrapolated"],
                        "format":["henn","henn"],
                        "density":numpy.array([3.3,1.0]),
                        "abundance":numpy.array([0.8,0.2]),
                        "filling":0.75,
                        "with_dhs":True,
                    },
                    "diana_wice":{
                        "dust":["amorphous_silicates_extrapolated","amorphous_carbon_zubko1996_extrapolated","water_ice"],
                        "format":["henn","henn","henn"],
                        "density":numpy.array([3.3,1.0,0.92]),
                        "abundance":numpy.array([0.8,0.2,0.5]),
                        "filling":0.75,
                        "with_dhs":True,
                    },
                }

                water_ice = Dust()
                water_ice.set_optical_constants_from_henn(os.path.dirname(__file__)+"/optical_constants/water_ice.txt")

                species = []
                for i in range(len(recipe_dict[dust])):
                    species.append(Dust())
                    if recipe_dict[dust]["format"][i] == "henn":
                        species[-1].set_optical_constants_from_henn(os.path.dirname(__file__)+"/optical_constants/"+recipe_dict[dust]["dust"][i]+".txt")
                        if "extrapolated" in recipe_dict[dust]["dust"][i]:
                            species[-1].calculate_optical_constants_on_wavelength_grid(water_ice.lam)
                    elif recipe_dict[dust]["format"][i] == "draine":
                        species[-1].set_optical_constants_from_draine(os.path.dirname(__file__)+"/optical_constants/graphite_parallel_0.01.txt")
                        species[-1].calculate_optical_constants_on_wavelength_grid(water_ice.lam)

                    species[-1].set_density(recipe_dict[dust]["density"][i])

                if "mass_fraction" in recipe_dict[dust]:
                    abundances = (recipe_dict[dust]["mass_fraction"]/recipe_dict[dust]["density"])/(recipe_dict[dust]["mass_fraction"]/recipe_dict[dust]["density"]).sum()
                else:
                    abundances = recipe_dict[dust]["abundance"] / recipe_dict[dust]["abundance"].sum()

                if "filling" not in recipe_dict[dust]:
                    recipe_dict[dust]["filling"] = 1.

                if "with_dhs" in recipe_dict[dust]:
                    with_dhs = recipe_dict[dust]["with_dhs"]

                dust = mix_dust(species, abundances, filling=recipe_dict[dust]["filling"])
            else:
                self.read(dust)
                return
        
        self.rho = dust.rho

        self.amax = numpy.logspace(-4.,1.,60)
        self.p = numpy.linspace(2.5,4.5,11)
        self.lam = dust.lam

        self.kabs = []
        self.ksca = []
        self.kext = []
        self.albedo = []

        a = numpy.logspace(numpy.log10(0.05e-4),1.,500)
        if singlesize:
            self.amax = a
        kabsgrid = numpy.zeros((self.lam.size, a.size))
        kscagrid = numpy.zeros((self.lam.size, a.size))

        for i in range(a.size):
            if with_dhs:
                dust.calculate_dhs_opacity(a[i], fmax=fmax, nf=nf, nang=1)
            else:
                dust.calculate_opacity(a[i], coat_volume_fraction=0.0, \
                        nang=1)

            kabsgrid[:,i] = dust.kabs
            kscagrid[:,i] = dust.ksca

        for p in self.p:
            kabs_temp = []
            ksca_temp = []
            kext_temp = []
            albedo_temp = []

            for amax in self.amax:
                if singlesize:
                    normfunc = numpy.zeros(a.size)
                    normfunc[a == amax] = 1.
                else:
                    normfunc = a**(3-p)
                    normfunc[a > amax] = 0.

                norm = scipy.integrate.trapz(normfunc, x=a)

                kabs_temp.append(scipy.integrate.trapz(kabsgrid*normfunc,\
                        x=a, axis=1)/norm)
                ksca_temp.append(scipy.integrate.trapz(kscagrid*normfunc,\
                        x=a, axis=1)/norm)
                kext_temp.append(kabs_temp[-1] + ksca_temp[-1])
                albedo_temp.append( ksca_temp[-1] / kext_temp[-1])

            self.kabs.append(kabs_temp)
            self.ksca.append(ksca_temp)
            self.kext.append(kext_temp)
            self.albedo.append(albedo_temp)

        self.kabs = numpy.array(self.kabs)
        self.ksca = numpy.array(self.ksca)
        self.kext = numpy.array(self.kext)
        self.albedo = numpy.array(self.albedo)

    def __call__(self, amax, p=None):
        if self.old:
            f_kabs = scipy.interpolate.interp2d(self.lam, self.amax, \
                    numpy.log10(self.kabs))
            f_ksca = scipy.interpolate.interp2d(self.lam, self.amax, \
                    numpy.log10(self.ksca))

            kabs = 10.**f_kabs(self.lam, amax)
            ksca = 10.**f_ksca(self.lam, amax)

            d = Dust()
            d.set_properties(self.lam, kabs, ksca)
        else:
            f_kabs = scipy.interpolate.RegularGridInterpolator(\
                    (self.p, self.amax, self.lam), numpy.log10(self.kabs))
            f_ksca = scipy.interpolate.RegularGridInterpolator(\
                    (self.p, self.amax, self.lam), numpy.log10(self.ksca))

            pts = numpy.array([[p, amax, lam] for lam in self.lam])

            kabs = 10.**f_kabs(pts)
            ksca = 10.**f_ksca(pts)

            d = Dust()
            d.set_properties(self.lam, kabs, ksca)

        return d

    def read(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "r")
        else:
            f = usefile

        self.lam = f['lam'][...]
        self.amax = f['amax'][...]

        if ('p' in f):
            self.p = f['p'][...]
            self.old = False
        else:
            self.old = True

        if ('kabs' in f):
            self.kabs = f['kabs'][...]
        if ('ksca' in f):
            self.ksca = f['ksca'][...]
        if (hasattr(self, 'kabs') and hasattr(self, 'ksca')):
            self.kext = self.kabs + self.ksca
            self.albedo = self.ksca / self.kext

        if ('rho' in f):
            self.rho = f['rho'][...][0]

        if (usefile == None):
            f.close()

    def write(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "w")
        else:
            f = usefile

        if hasattr(self, 'amax'):
            amax_dset = f.create_dataset("amax", (self.amax.size,), dtype='f')
            amax_dset[...] = self.amax
        if hasattr(self, 'p'):
            amax_dset = f.create_dataset("p", (self.p.size,), dtype='f')
            amax_dset[...] = self.p
        if hasattr(self, 'lam'):
            lam_dset = f.create_dataset("lam", (self.lam.size,), dtype='f')
            lam_dset[...] = self.lam
        
        if hasattr(self, 'kabs'):
            kabs_dset = f.create_dataset("kabs", self.kabs.shape, dtype='f')
            kabs_dset[...] = self.kabs
        if hasattr(self, 'ksca'):
            ksca_dset = f.create_dataset("ksca", self.ksca.shape, dtype='f')
            ksca_dset[...] = self.ksca
        if hasattr(self, 'g'):
            g_dset = f.create_dataset("g", self.g.shape, dtype='f')
            g_dset[...] = self.g

        if hasattr(self, 'rho'):
            rho_dset = f.create_dataset("rho", (1,), dtype='f')
            rho_dset[...] = [self.rho]

        if (usefile == None):
            f.close()
