from pdspy.constants.astronomy import R_sun, pc
import scipy.interpolate
import astropy.table
import numpy

def btsettl_photometry(Teff=5770., Logg=4.4, Rstar=1.0, dpc=140., \
        system="Johnson", filters="B", AB=False):

    # Create the filename.

    path = '/'.join(__file__.split("/")[0:-1])+"/btsettl_data/"

    basename = "colmag.BT-Settl.server."

    filename = basename + system + "."

    if AB:
        filename += "AB.txt"
    else:
        filename += "Vega.txt"

    # Get the column names.

    f = open(path+filename, "r")
    lines = f.readlines()
    f.close()

    for i, line in enumerate(lines):
        if line[0] != "!":
            break

    colnames = lines[i-1].split()[1:]

    # Get the data.

    data = numpy.loadtxt(path+filename, comments='!')

    # Make the data into a table.

    table = astropy.table.Table(data[1:,0:16], names=colnames)

    # Make sure we only include the Teff's with all the log(g)'s.

    """
    teff = numpy.unique(table["Teff"])
    logg = numpy.unique(table["Logg"])

    for T in teff:
        if len(table["Teff"][table["Teff"] == T]) < len(logg):
            table.remove_rows(numpy.where(table["Teff"] == T)[0])
    """

    # Now loop through the filters and get the photometry.

    """
    teff = numpy.array(numpy.unique(table["Teff"]))
    logg = numpy.array(numpy.unique(table["Logg"]))
    """

    mag = []

    for filter in filters.split(','):
        #data = numpy.array(table[filter]).reshape((teff.size, logg.size))

        # Now do the 2D interpolation.

        #f = scipy.interpolate.interp2d(logg, teff, data)

        f = scipy.interpolate.LinearNDInterpolator((table["Teff"], \
                table["Logg"]), table[filter], rescale=False)

        # Calculate the photometry now.

        mag.append(f([[Teff,Logg]])[0])

    # Turn the photometry into an array.

    mag = numpy.array(mag)

    # Turn into an absolute magnitude.

    M = mag - 5 * numpy.log10(Rstar*R_sun / pc) + 5.

    # Now get the apparent magnitude.

    photometry = M + 5*(numpy.log10(dpc) - 1.)

    return photometry
