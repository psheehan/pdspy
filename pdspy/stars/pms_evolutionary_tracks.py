import scipy.interpolate
import astropy.table
import numpy

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

    tck = scipy.interpolate.bisplrep(table["Teff"], table["L/Ls"], \
            table["M/Ms"], kx=5, ky=5)

    # Finally, get the stellar mass.

    if isinstance(temperature,float):
        return scipy.interpolate.bisplev(temperature, luminosity, tck)
    else:
        return numpy.array([scipy.interpolate.bisplev(temperature[i], \
                luminosity[i], tck) for i in range(len(temperature))])

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

    tck = scipy.interpolate.bisplrep(table["Teff"], table["L/Ls"], \
            table["log_t(yr)"], kx=5, ky=5)

    # Finally, get the stellar mass.

    if isinstance(temperature,float):
        return 10.**scipy.interpolate.bisplev(temperature, luminosity, tck)
    else:
        return numpy.array([10.**scipy.interpolate.bisplev(temperature[i], \
                luminosity[i], tck) for i in range(len(temperature))])

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

    tck = scipy.interpolate.bisplrep(table["M/Ms"], table["log_t(yr)"], \
            table["Teff"], kx=5, ky=5, s=0)

    # Finally, get the stellar mass.

    if isinstance(age,float) and isinstance(mass,float):
        return scipy.interpolate.bisplev(mass, numpy.log10(age), tck)
    elif isinstance(age,float):
        return numpy.array([scipy.interpolate.bisplev(mass[i], \
                numpy.log10(age), tck) for i in range(len(mass))])
    else:
        return numpy.array([scipy.interpolate.bisplev(mass, \
                numpy.log10(age[i]), tck) for i in range(len(age))])

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

    tck = scipy.interpolate.bisplrep(table["M/Ms"], table["log_t(yr)"], \
            table["L/Ls"], kx=5, ky=5, s=0)

    # Finally, get the stellar mass.

    if isinstance(age,float) and isinstance(mass,float):
        return 10.**scipy.interpolate.bisplev(mass, numpy.log10(age), tck)
    elif isinstance(age,float):
        return numpy.array([10.**scipy.interpolate.bisplev(mass[i], \
                numpy.log10(age), tck) for i in range(len(mass))])
    else:
        return numpy.array([10.**scipy.interpolate.bisplev(mass, \
                numpy.log10(age[i]), tck) for i in range(len(age))])
