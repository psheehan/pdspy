import scipy.interpolate
import astropy.table
import numpy

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

    if type(temperature) == float:
        return scipy.interpolate.bisplev(temperature, luminosity, tck)
    else:
        return numpy.array([scipy.interpolate.bisplev(temperature[i], \
                luminosity[i], tck) for i in range(len(temperature))])

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

    if type(temperature) == float:
        return 10.**scipy.interpolate.bisplev(temperature, luminosity, tck)
    else:
        return numpy.array([10.**scipy.interpolate.bisplev(temperature[i], \
                luminosity[i], tck) for i in range(len(temperature))])
