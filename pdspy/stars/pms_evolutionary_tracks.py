import scipy.interpolate
import astropy.table
import numpy
import time

############################################################################
#
# Get Mass from Teff and Lstar
#
############################################################################

def pms_get_mstar(temperature=None, luminosity=None, tracks="BHAC15"):

    # Load in the data for the appropriate set of evolutionary tracks.

    path = '/'.join(__file__.split("/")[0:-1])+"/evolutionary_tracks/"
    
    if tracks == "BHAC15":
        f = open(path+"BHAC15_tracks+structure.txt","r")
        lines = f.readlines()
        f.close()

        colnames = lines[46].split()[1:]

        data = numpy.loadtxt(path+"BHAC15_tracks+structure.txt", comments="!", \
                skiprows=45)

    # Make the data into a table.

    table = astropy.table.Table(data, names=colnames)

    # Now do the 2D interpolation.

    Mstar = scipy.interpolate.LinearNDInterpolator((table["Teff"], \
            table["L/Ls"]), table["M/Ms"])

    # Finally, get the stellar mass.

    if isinstance(temperature,float) and isinstance(luminosity,float):
        xi = numpy.array([[temperature, numpy.log10(luminosity)]])
    else:
        xi = numpy.array([[temperature[i],numpy.log10(luminosity[i])] for i in \
                range(len(temperature))])

    return Mstar(xi)

############################################################################
#
# Get Age from Teff and Lstar
#
############################################################################

def pms_get_age(temperature=None, luminosity=None, tracks="BHAC15"):

    # Load in the data for the appropriate set of evolutionary tracks.

    path = '/'.join(__file__.split("/")[0:-1])+"/evolutionary_tracks/"
    
    if tracks == "BHAC15":
        f = open(path+"BHAC15_tracks+structure.txt","r")
        lines = f.readlines()
        f.close()

        colnames = lines[46].split()[1:]

        data = numpy.loadtxt(path+"BHAC15_tracks+structure.txt", comments="!", \
                skiprows=45)

    # Make the data into a table.

    table = astropy.table.Table(data, names=colnames)

    # Now do the 2D interpolation.

    Age = scipy.interpolate.LinearNDInterpolator((table["Teff"], \
            table["L/Ls"]), table["log_t(yr)"])

    # Finally, get the stellar mass.

    if isinstance(temperature,float) and isinstance(luminosity,float):
        xi = numpy.array([[temperature, numpy.log10(luminosity)]])
    else:
        xi = numpy.array([[temperature[i],numpy.log10(luminosity[i])] for i in \
                range(len(temperature))])

    return 10.**Age(xi)

############################################################################
#
# Get Teff from Mass and Age.
#
############################################################################

def pms_get_teff(mass=1.0, age=1.0e6, tracks="BHAC15"):

    # Load in the data for the appropriate set of evolutionary tracks.

    path = '/'.join(__file__.split("/")[0:-1])+"/evolutionary_tracks/"
    
    if tracks == "BHAC15":
        f = open(path+"BHAC15_tracks+structure.txt","r")
        lines = f.readlines()
        f.close()

        colnames = lines[46].split()[1:]

        data = numpy.loadtxt(path+"BHAC15_tracks+structure.txt", comments="!", \
                skiprows=45)

    # Make the data into a table.

    table = astropy.table.Table(data, names=colnames)

    # Now do the 2D interpolation.

    Teff = scipy.interpolate.LinearNDInterpolator((table["M/Ms"], \
            table["log_t(yr)"]), table["Teff"])

    # Finally, get the stellar mass.

    if isinstance(age,float) and isinstance(mass,float):
        xi = numpy.array([[mass, numpy.log10(age)]])
    elif isinstance(age,float):
        xi = numpy.array([[mass[i],numpy.log10(age)] for i in range(len(mass))])
    else:
        xi = numpy.array([[mass,numpy.log10(age[i])] for i in range(len(age))])

    return Teff(xi)

############################################################################
#
# Get Lstar from Mass and Age.
#
############################################################################

def pms_get_luminosity(mass=1.0, age=1.0e6, tracks="BHAC15"):

    # Load in the data for the appropriate set of evolutionary tracks.

    path = '/'.join(__file__.split("/")[0:-1])+"/evolutionary_tracks/"
    
    if tracks == "BHAC15":
        f = open(path+"BHAC15_tracks+structure.txt","r")
        lines = f.readlines()
        f.close()

        colnames = lines[46].split()[1:]

        data = numpy.loadtxt(path+"BHAC15_tracks+structure.txt", comments="!", \
                skiprows=45)

    # Make the data into a table.

    table = astropy.table.Table(data, names=colnames)

    # Now do the 2D interpolation.

    Lstar = scipy.interpolate.LinearNDInterpolator((table["M/Ms"],\
            table["log_t(yr)"]), table["L/Ls"])

    # Finally, get the stellar mass.

    if isinstance(age,float) and isinstance(mass,float):
        xi = numpy.array([[mass, numpy.log10(age)]])
    elif isinstance(age,float):
        xi = numpy.array([[mass[i],numpy.log10(age)] for i in range(len(mass))])
    else:
        xi = numpy.array([[mass,numpy.log10(age[i])] for i in range(len(age))])

    return 10.**Lstar(xi)
