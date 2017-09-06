import numpy
import astropy
import astropy.coordinates

#def update_catalog(catalog, *lists, tol=0.3, column_names=['1']):
def update_catalog(catalog, tol=0.3, column_names=['1'], *lists):

    coord_catalog = astropy.coordinates.SkyCoord(catalog["ra"].tolist(), \
            catalog["dec"].tolist(), frame='icrs')
    catalog["id"] = numpy.arange(len(catalog))

    for i, list1 in enumerate(lists):
        coord1 = astropy.coordinates.SkyCoord(list1["ra"].tolist(), \
                list1['dec'].tolist(), frame='icrs')

        idx, d2d, d3d = astropy.coordinates.match_coordinates_sky(coord1, \
                coord_catalog)

        ids, counts = numpy.unique(idx[d2d.arcsec < tol], return_counts=True)
        for j in range(len(ids)):
            if counts[j] > 1:
                min_d2d = min(d2d[idx == ids[j]].arcsec)
                d2d[(d2d.arcsec != min_d2d) & (idx == ids[j])] = \
                        astropy.coordinates.Angle("0d00m{0:f}s".format(2*tol))

        for j, k in enumerate(idx):
            if d2d.arcsec[j] < tol:
                for key in list1.colnames:
                    catalog[key+"_{0}".format(column_names[i])][k] = \
                            list1[key][j]

    del catalog["id"]

    return catalog
