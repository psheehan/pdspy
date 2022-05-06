from ..constants.physics import c
from .libinterferometry import Visibilities
import astropy.table
import casatools
import numpy

def readms(filename, spw='all', tolerance=0.01, time_tolerance=0., \
        datacolumn="corrected", corr=["I"]):
    """Testing

    :param filename: The name of the MS file that you would like to read in.
    :param spw: The list of spectral windows to read in.
    """

    # Load the MS file.

    ms = casatools.ms()

    ms.open(filename)

    # If we want all of the spw, then get them all.

    if spw == 'all':
        tb = casatools.table()

        tb.open(filename)
        spw = tb.getcol('DATA_DESC_ID')
        tb.close()

        spw = list(numpy.unique(spw))

    # Loop through all of the DATA_DESC_ID values and collect the relevant data.

    if datacolumn == "corrected":
        prefix = "corrected_"
    else:
        prefix = ""

    i = 0                                                               
    ms.reset()                                                          
    data = []
    for i in spw:
        ms.selectinit(datadescid=i)
        ms.selectpolarization(corr)
        data.append(ms.getdata(items=["u","v",prefix+"real",prefix+\
                "imaginary","weight","flag","axis_info","uvdist",\
                "antenna1","antenna2","time"]))
        ms.reset()

    # We are done with the data now, so close it.

    ms.close()

    # Test how many of the spectral windows are identical to each other.

    matching_spw = [0]

    for i in range(1,len(data)):
        matched = False

        for j in range(i):
            if i == j:
                continue

            # If they don't have the same number of frequencies, they can't be
            # the same spectral window.

            if data[i]["axis_info"]["freq_axis"]["chan_freq"].shape != \
                    data[j]["axis_info"]['freq_axis']['chan_freq'].shape:
                continue

            # Test how different they are and compare with some threshold.

            diff = (data[i]["axis_info"]['freq_axis']['chan_freq'] - \
                    data[j]["axis_info"]['freq_axis']['chan_freq']) / \
                    data[j]["axis_info"]["freq_axis"]['resolution']

            if numpy.abs(diff).max() < tolerance:
                matching_spw.append(j)
                matched = True
                break

        # If we didn't find a match, add as a new spectral window.

        if not matched:
            matching_spw.append(i)

    # Loop through all of the unique spectral windows that we found.

    for i in numpy.unique(matching_spw):
        new_freq = data[i]["axis_info"]["freq_axis"]["chan_freq"]

        # Collect all of the DATA_DESC_ID's that match that spectral window.

        new_u, new_v, new_real, new_imag, new_weights, new_flags, \
                new_uvdist, new_antenna1, new_antenna2, new_time = [], [], [], \
                [], [], [], [], [], [], []

        for j in range(len(data)):
            if matching_spw[j] == i:
                new_u.append(data[j]["u"])
                new_v.append(data[j]["v"])
                new_uvdist.append(data[j]["uvdist"])
                new_real.append(data[j][prefix+"real"])
                new_imag.append(data[j][prefix+"imaginary"])
                new_weights.append(data[j]["weight"])
                new_flags.append(data[j]["flag"])
                new_antenna1.append(data[j]["antenna1"])
                new_antenna2.append(data[j]["antenna2"])
                new_time.append(data[j]["time"])

        # Concatenate them all along the uvdist axis.

        new_u = numpy.concatenate(new_u)
        new_v = numpy.concatenate(new_v)
        new_uvdist = numpy.concatenate(new_uvdist)
        new_real = numpy.concatenate(new_real, axis=2)
        new_imag = numpy.concatenate(new_imag, axis=2)
        new_weights = numpy.concatenate(new_weights, axis=1)
        new_flags = numpy.concatenate(new_flags, axis=2)
        new_antenna1 = numpy.concatenate(new_antenna1)
        new_antenna2 = numpy.concatenate(new_antenna2)
        new_time = numpy.concatenate(new_time)

        # Adjust the weights for flags and make the right shape.
        
        new_weights = ((new_flags == False) * new_weights.reshape((\
                new_weights.shape[0],1,new_weights.shape[1])))

        # Average out the two different polarizations.

        new_real = (new_real * new_weights).sum(axis=0)
        new_imag = (new_imag * new_weights).sum(axis=0)
        new_weights = new_weights.sum(axis=0)

        new_real[new_weights != 0] /= new_weights[new_weights != 0]
        new_imag[new_weights != 0] /= new_weights[new_weights != 0]

        # Transpose to make the right shape.

        new_real = numpy.transpose(new_real)
        new_imag = numpy.transpose(new_imag)
        new_weights = numpy.transpose(new_weights)

        # Trim the autocorrelation data.

        good = new_uvdist != 0

        new_u = new_u[good]
        new_v = new_v[good]
        new_uvdist = new_uvdist[good]
        new_real = new_real[good,:]
        new_imag = new_imag[good,:]
        new_weights = new_weights[good,:]
        new_antenna1 = new_antenna1[good]
        new_antenna2 = new_antenna2[good]
        new_time = new_time[good]

        # Now do some fancy stuff to merge the multiple spectral windows. Since 
        # it is possible that even for the same observation, there'll be a 
        # different number of UV points because of flagging, we need to match 
        # the uv points that exist for both, but add zeros when that point 
        # doesn't exist for one or the other.

        if i == 0:
            freq = new_freq

            # Make the first column uvdist.

            table = astropy.table.Table([new_antenna1, \
                    new_antenna2, new_time, new_u, new_v, new_real, new_imag, \
                    new_weights, new_uvdist], names=["antenna1","antenna2",\
                    "time"]+[string+"_"+str(i) for string in ['u','v','real',\
                    'imag','weights','uvdist']])
        else:
            freq = numpy.concatenate((freq, new_freq))

            # Make the first column uvdist.

            new_table = astropy.table.Table([new_antenna1, \
                    new_antenna2, new_time, new_u, new_v, new_real, new_imag, \
                    new_weights, new_uvdist], names=["antenna1","antenna2",\
                    "time"]+[string+"_"+str(i) for string in ['u','v','real',\
                    'imag','weights','uvdist']])

            # Do the fancy merging.

            if time_tolerance == 0:
                table = astropy.table.join(table, new_table, join_type='outer',\
                        keys=["antenna1","antenna2","time"])
            else:
                table = astropy.table.join(table, new_table, join_type='outer',\
                        keys=["antenna1","antenna2","time"], \
                        join_funcs={'time':astropy.table.\
                        join_distance(time_tolerance)})

                for key in ["time"]:
                    table[key] = numpy.concatenate([\
                            table[key+'_1'].data[:,numpy.newaxis], \
                            table[key+'_2'].data[:,numpy.newaxis]], axis=1)

                    wt = numpy.where(table[key].data == 0, 0, 1)

                    table[key] = (table[key].data*wt).sum(axis=1)/\
                            wt.sum(axis=1)
                
                    table.remove_columns([key+'_1',key+'_2',key+'_id'])

    # Make sure masked colums are filled appropriately.

    for colname in table.colnames:
        table[colname].fill_value = 0.
    table = table.filled()

    # Get the info out of the table.

    u = numpy.concatenate([table['u_{0:d}'.format(i)].data[:,numpy.newaxis] \
            for i in numpy.unique(matching_spw)], axis=1)
    v = numpy.concatenate([table['v_{0:d}'.format(i)].data[:,numpy.newaxis] \
            for i in numpy.unique(matching_spw)], axis=1)

    freq = freq[:,0]

    real = numpy.concatenate([table['real_{0:d}'.format(i)].data for i in \
            numpy.unique(matching_spw)], axis=1)
    imag = numpy.concatenate([table['imag_{0:d}'.format(i)].data for i in \
            numpy.unique(matching_spw)], axis=1)
    weights = numpy.concatenate([table['weights_{0:d}'.format(i)].data for i \
            in numpy.unique(matching_spw)], axis=1)

    # Trim down u and v.

    uwt = numpy.where(numpy.logical_and(u == 0, v == 0), 0, 1)
    u = (u*uwt).sum(axis=1) / uwt.sum(axis=1)
    v = (v*uwt).sum(axis=1) / uwt.sum(axis=1)

    # Include the complex conjugate.

    scale = 100 * freq.mean() / c

    u = numpy.concatenate((u, -u))*scale
    v = numpy.concatenate((v, -v))*scale
    real = numpy.concatenate((real, real))
    imag = -1*numpy.concatenate((imag, -imag))
    weights = numpy.concatenate((weights, weights))
    
    return Visibilities(u, v, freq, real, imag, weights)
