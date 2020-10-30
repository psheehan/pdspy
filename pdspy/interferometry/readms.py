from ..constants.physics import c
from .libinterferometry import Visibilities
import casatools
import numpy

def readms(filename, spw=[0], tolerance=0.01, datacolumn="corrected"):

    # Load the MS file.

    ms = casatools.ms()

    ms.open(filename)

    # Loop through all of the DATA_DESC_ID values and collect the relevant data.

    if datacolumn == "corrected":
        prefix = "corrected_"
    else:
        prefix = ""

    i = 0                                                               
    ms.reset()                                                          
    data = []
    while ms.selectinit(datadescid=i):
        if int(list(ms.getspectralwindowinfo().keys())[0]) in spw:
            data.append(ms.getdata(items=["u","v",prefix+"real",prefix+\
                    "imaginary","weight","flag","axis_info","uvdist"]))
        ms.reset()
        i += 1

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
                new_uvdist = [], [], [], [], [], [], []

        for j in range(len(data)):
            if matching_spw[j] == i:
                new_u.append(data[j]["u"])
                new_v.append(data[j]["v"])
                new_uvdist.append(data[j]["uvdist"])
                new_real.append(data[j]["real"])
                new_imag.append(data[j]["imaginary"])
                new_weights.append(data[j]["weight"])
                new_flags.append(data[j]["flag"])

        # Concatenate them all along the uvdist axis.

        new_u = numpy.concatenate(new_u)
        new_v = numpy.concatenate(new_v)
        new_uvdist = numpy.concatenate(new_uvdist)
        new_real = numpy.concatenate(new_real, axis=2)
        new_imag = numpy.concatenate(new_imag, axis=2)
        new_weights = numpy.concatenate(new_weights, axis=1)
        new_flags = numpy.concatenate(new_flags, axis=2)

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

        # Now do some fancy stuff to merge the multiple spectral windows. Since 
        # it is possible that even for the same observation, there'll be a 
        # different number of UV points because of flagging, we need to match 
        # the uv points that exist for both, but add zeros when that point 
        # doesn't exist for one or the other.

        if i == 0:
            freq = new_freq

            # Make the first column uvdist.

            u = numpy.hstack((new_uvdist.reshape((-1,1)),new_u.reshape((-1,1))))
            v = numpy.hstack((new_uvdist.reshape((-1,1)),new_v.reshape((-1,1))))
            real = numpy.hstack((new_uvdist.reshape((-1,1)),new_real))
            imag = numpy.hstack((new_uvdist.reshape((-1,1)),new_imag))
            weights = numpy.hstack((new_uvdist.reshape((-1,1)),new_weights))
        else:
            freq = numpy.concatenate((freq, new_freq))

            # Make the first column uvdist.

            new_u = numpy.hstack((new_uvdist.reshape((-1,1)), \
                    new_u.reshape((-1,1))))
            new_v = numpy.hstack((new_uvdist.reshape((-1,1)), \
                    new_v.reshape((-1,1))))
            new_real = numpy.hstack((new_uvdist.reshape((-1,1)),new_real))
            new_imag = numpy.hstack((new_uvdist.reshape((-1,1)),new_imag))
            new_weights = numpy.hstack((new_uvdist.reshape((-1,1)),new_weights))

            # Do the fancy merging.

            u = merge_arrays(u, new_u)
            v = merge_arrays(v, new_v)

            real = merge_arrays(real, new_real)
            imag = merge_arrays(imag, new_imag)
            weights = merge_arrays(weights, new_weights)

            # Since for u and v, the two columns are duplicates of each other 
            # (except when there's a uv point that exists in one but not the 
            # other, trim it down to just one column.

            u[:,1] = numpy.where(u[:,1] != 0, u[:,1], u[:,2])
            u = u[:,0:2]
            v[:,1] = numpy.where(v[:,1] != 0, v[:,1], v[:,2])
            v = v[:,0:2]

    # Finally, chop off the uvdist column as it is not needed any more.

    u = u[:,1]
    v = v[:,1]
    freq = freq[:,0]
    real = real[:, 1:]
    imag = imag[:, 1:]
    weights = weights[:, 1:]

    # Include the complex conjugate.

    scale = 100 * freq.mean() / c

    u = numpy.concatenate((u, -u))*scale
    v = numpy.concatenate((v, -v))*scale
    real = numpy.concatenate((real, real))
    imag = numpy.concatenate((imag, -imag))
    weights = numpy.concatenate((weights, weights))
    
    return Visibilities(u, v, freq, real, imag, weights)

# A little function to merge two arrays that may share the same first 
# column.

def merge_arrays(a, b):
    d = numpy.union1d(a[:, 0],b[:, 0]).reshape((-1,1))
    z = numpy.zeros((d.shape[0],a.shape[1]-1 + b.shape[1]-1),dtype=int)
    c = numpy.hstack((d,z))
    mask = numpy.in1d(c[:, 0], a[:, 0])
    c[mask,1:a.shape[1]] = a[:, 1:]
    mask = numpy.in1d(c[:, 0], b[:, 0])
    c[mask,a.shape[1]:] = b[:, 1:]

    return c
