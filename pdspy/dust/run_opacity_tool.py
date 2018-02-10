from .Dust import Dust
import os
import numpy

def run_opacity_tool(amin=0.05, amax=3000., apow=3.5, na=50, fcarbon=0.13, \
        Vcarbon=0.15, porosity=0.25, fmax=0.8, lmin=0.05, lmax=5000, nlam=300, \
        filename=None, verbose=False):

    original_dir = os.environ["PWD"]
    os.mkdir("/tmp/temp_opacitytool")
    os.chdir("/tmp/temp_opacitytool")

    try:
        command = "OpacityTool -amin {0:f} -amax {1:f} -apow {2:f} -na {3:d} "\
                "-fcarbon {4:f} -Vcarbon {5:f} -porosity {6:f} -fmax {7:f} "\
                "-lmin {8:f} -lmax {9:f} -nlam {10:d}".format(amin, amax, apow,\
                na, fcarbon, Vcarbon, porosity, fmax, lmin, lmax, nlam)

        if not verbose:
            os.system(command + " > log.txt")
        else:
            os.system(command)
        
        data = numpy.loadtxt("particle.dat")

        d = Dust()
        d.set_properties(data[:,0], data[:,1], data[:,2])

        if not verbose:
            os.system("rm particle.dat log.txt")
        else:
            os.system("rm particle.dat")
        os.chdir(original_dir)
        os.rmdir("/tmp/temp_opacitytool")

        if filename != None:
            d.write(filename)
        else:
            return d
    except:
        os.system("rm particle.dat log.txt")
        os.chdir(original_dir)
        os.rmdir("/tmp/temp_opacitytool")
